# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict, Any, Type
import logging

from .data_operations import SocialData
from .recommender import SocialRecommender
from .utils import (
    load_config,
    load_action_space,
    get_action_info,
    set_logger,
    load_init_observation,
    load_start_time,
    load_end_time,
    str_to_datetime,
)
from .basic import Action

logger = logging.getLogger(__name__)


class Environment(ABC):
    """环境基类"""

    @abstractmethod
    def to_text_prompt(self) -> str:
        """转换为文本提示"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """重置环境"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action, action_args: Dict[str, Any], agent=None):
        """执行动作"""
        raise NotImplementedError


class SocialEnvironment(Environment):
    """
    社交环境类
    
    参考 BotSim 框架的设计模式，提供统一的社交网络环境接口
    负责管理数据、动作空间、状态转换等
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_data: Optional[SocialData] = None
        self.env_action_space: Optional[Dict[str, Any]] = None
        self.env_action_info: Optional[List[Dict[str, Any]]] = None
        self.logger: Optional[logging.Logger] = None
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.current_time: Optional[datetime] = None
        self.platform: str = config.get("platform", "twitter")
        self.action = None  # 社交行为执行器
        self.recommender: Optional[SocialRecommender] = None
        self.clock_factor: float = config.get("simulation", {}).get("clock_factor", 60)

    def reset(self):
        """重置环境状态"""
        # 初始化数据管理器
        self.env_data = SocialData(self.config)
        
        # 加载动作空间
        self.env_action_space = load_action_space(self.config)
        self.env_action_info = get_action_info(self.env_action_space)
        
        # 设置日志
        self.logger = set_logger(self.config)
        
        # 加载时间配置
        self.start_time = load_start_time(self.config)
        self.end_time = load_end_time(self.config)
        self.current_time = str_to_datetime(self.start_time)

        # 初始化推荐系统（TwHIN-BERT）
        self.recommender = SocialRecommender(self.config)
        
        # 生成初始观察
        init_observation = load_init_observation(
            env_action_info=self.env_action_info,
            platform=self.platform,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        
        logger.info("Environment reset completed")
        return init_observation

    def get_env_data(self) -> SocialData:
        """获取数据管理器"""
        return self.env_data

    def get_env_action_space(self) -> Dict[str, Any]:
        """获取动作空间"""
        return self.env_action_space

    def get_env_action_info(self) -> List[Dict[str, Any]]:
        """获取动作信息"""
        return self.env_action_info

    def step(self, action: Action, action_args: Dict[str, Any], agent=None):
        """执行动作"""
        if not action.enable:
            action_res = ""
            observation = "Can't do this action now, Please choose another one."
        else:
            try:
                action_res = action(action_args, self, agent)
                observation = action_res
                
                # 记录日志
                if agent:
                    self.logger.info(
                        f"agent id: {getattr(agent, 'social_agent_id', 'unknown')}, "
                        f"action name: {action.name}, "
                        f"action args: {action_args}, "
                        f"result: {observation}"
                    )
                else:
                    self.logger.info(
                        f"action name: {action.name}, "
                        f"action args: {action_args}, "
                        f"result: {observation}"
                    )
                    
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                observation = f"Action execution failed: {str(e)}"
        
        return observation

    def to_text_prompt(self) -> str:
        """转换为文本提示"""
        # 获取当前环境状态
        stats = self.env_data.get_statistics() if self.env_data else {}
        
        env_info = f"""
Current Environment Status:
- Platform: {self.platform}
- Total Users: {stats.get('total_users', 0)}
- Total Posts: {stats.get('total_posts', 0)}
- Total Comments: {stats.get('total_comments', 0)}
- Bot Users: {stats.get('bot_users', 0)}
- Human Users: {stats.get('human_users', 0)}

Available Actions:
"""
        
        if self.env_action_info:
            for action_info in self.env_action_info:
                env_info += f"- {action_info['ActionName']}: {action_info['Description']}\n"
        
        return env_info

    def tick(self, real_seconds: float = 1.0):
        """推进环境时间（基于时钟加速因子）"""
        if not self.current_time:
            self.current_time = datetime.utcnow()
        delta = real_seconds * float(self.clock_factor)
        self.current_time = self.current_time + timedelta(seconds=delta)
        return self.current_time

    def get_recommended_posts_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户推荐帖子"""
        if not self.env_data:
            return []

        user = self.env_data.get_user(user_id)
        if not user:
            return []

        posts = self.env_data.get_all_posts()
        comments = self.env_data.get_all_comments()

        post_dicts = [
            {
                "post_id": p.post_id,
                "author_id": p.author_id,
                "author_name": p.author_name,
                "content": p.content,
                "created_time": p.created_time,
                "num_likes": p.num_likes,
                "num_comments": p.num_comments,
                "num_dislikes": p.num_dislikes,
                "num_reposts": p.num_reposts,
                "tags": p.tags,
                "content_type": "post",
            }
            for p in posts
        ]

        # Treat comments as repost candidates (paper requirement)
        comment_dicts = [
            {
                "post_id": c.post_id,
                "comment_id": c.comment_id,
                "author_id": c.author_id,
                "author_name": c.author_name,
                "content": c.content,
                "created_time": c.created_time,
                "num_likes": c.num_likes,
                "num_comments": 0,
                "num_dislikes": c.num_dislikes,
                "num_reposts": 0,
                "tags": [],
                "content_type": "comment",
            }
            for c in comments
        ]

        candidates = post_dicts + comment_dicts

        if self.recommender:
            now = self.current_time or datetime.utcnow()
            ranked = self.recommender.recommend(user.to_dict(), candidates, now=now, top_k=limit)
        else:
            ranked = self._fallback_recommend(user_id, limit)

        # Normalize created_time for output
        for p in ranked:
            if isinstance(p.get("created_time"), datetime):
                p["created_time"] = p["created_time"].isoformat()
        return ranked

    def _fallback_recommend(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """回退推荐：基于关注者帖子 + 热门帖子"""
        posts = self.env_data.get_recommended_posts(user_id, limit)
        return [
            {
                "post_id": post.post_id,
                "author_name": post.author_name,
                "content": post.content,
                "created_time": post.created_time,
                "num_likes": post.num_likes,
                "num_comments": post.num_comments,
                "num_dislikes": post.num_dislikes,
                "tags": post.tags,
            }
            for post in posts
        ]

    def get_user_feed(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        """获取用户信息流"""
        if not self.env_data:
            return {"posts": [], "groups": [], "recommendations": []}
        
        user = self.env_data.get_user(user_id)
        if not user:
            return {"posts": [], "groups": [], "recommendations": []}
        
        # 获取推荐帖子
        recommended_posts = self.get_recommended_posts_for_user(user_id, limit)
        
        # 获取用户加入的群组
        user_groups = self.env_data.get_user_groups(user_id)
        groups_info = [
            {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "description": group.description,
                "member_count": group.member_count
            }
            for group in user_groups
        ]
        
        # 获取热门帖子
        all_posts = list(self.env_data.posts.values())
        hot_posts = sorted(all_posts, key=lambda x: x.num_likes, reverse=True)[:limit//2]
        hot_posts_info = [
            {
                "post_id": post.post_id,
                "author_name": post.author_name,
                "content": post.content,
                "created_time": post.created_time.isoformat(),
                "num_likes": post.num_likes,
                "num_comments": post.num_comments
            }
            for post in hot_posts
        ]
        
        return {
            "posts": recommended_posts,
            "groups": groups_info,
            "hot_posts": hot_posts_info,
            "user_info": {
                "user_id": user.id,
                "screen_name": user.profile.screen_name,
                "name": user.profile.name,
                "followers_count": user.profile.followers_count,
                "friends_count": user.profile.friends_count,
                "statuses_count": user.profile.statuses_count
            }
        }

    def format_user_feed_context(self, user_id: str, limit: int = 10) -> str:
        """将用户信息流格式化为上下文文本"""
        feed = self.get_user_feed(user_id, limit=limit)
        parts = []
        posts = feed.get("posts", [])
        hot_posts = feed.get("hot_posts", [])
        if posts:
            parts.append("Recommended posts:")
            for p in posts[:limit]:
                parts.append(f"- ({p.get('post_id')}) {p.get('author_name')}: {p.get('content')[:120]}")
        if hot_posts:
            parts.append("Hot posts:")
            for p in hot_posts[:max(1, limit // 2)]:
                parts.append(f"- ({p.get('post_id')}) {p.get('author_name')}: {p.get('content')[:120]}")
        return "\n".join(parts) if parts else "No feed content available."

    def search_content(self, query: str, search_type: str = "posts", limit: int = 10) -> List[Dict[str, Any]]:
        """搜索内容"""
        if not self.env_data:
            return []
        
        if search_type == "posts":
            results = self.env_data.search_posts(query, limit)
            return [
                {
                    "post_id": post.post_id,
                    "author_name": post.author_name,
                    "content": post.content,
                    "created_time": post.created_time.isoformat(),
                    "num_likes": post.num_likes,
                    "num_comments": post.num_comments
                }
                for post in results
            ]
        elif search_type == "users":
            results = self.env_data.search_users(query, limit)
            return [
                {
                    "user_id": user.id,
                    "screen_name": user.profile.screen_name,
                    "name": user.profile.name,
                    "description": user.profile.description,
                    "followers_count": user.profile.followers_count,
                    "statuses_count": user.profile.statuses_count
                }
                for user in results
            ]
        else:
            return []

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户档案（符合 str.md 格式）"""
        if not self.env_data:
            return None
        
        user = self.env_data.get_user(user_id)
        if not user:
            return None
        
        return user.to_dict()

    def exit(self):
        """退出环境"""
        logger.info("Environment exiting")
        # 可以在这里添加清理逻辑
        pass 
