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

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

from .entity import SocialUser, SocialPost, SocialComment, Domain, TwitterProfile
# from .entity import SocialGroup  # 暂时注释掉
from .utils import generate_id, str_to_datetime, check_time

logger = logging.getLogger(__name__)


class SocialData:
    """
    社交数据管理类
    
    负责管理社交网络中的所有数据，包括用户、帖子、评论、群组等
    提供数据的增删改查、推荐、搜索等功能
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.users: Dict[str, SocialUser] = {}
        self.posts: Dict[str, SocialPost] = {}
        self.comments: Dict[str, SocialComment] = {}
        # self.groups: Dict[str, SocialGroup] = {}  # 暂时注释掉
        self.domains: Dict[str, Domain] = {}  # 新增：领域管理
        
        # 索引
        self.user_name_to_id: Dict[str, str] = {}
        self.post_author_index: Dict[str, List[str]] = {}  # author_id -> post_ids
        self.domain_name_to_id: Dict[str, str] = {}  # 新增：领域名称索引
        
        # 配置参数
        self.page_size = config.get("constants", {}).get("page_size", 20)
        self.platform = config.get("platform", "twitter")
        
        # 加载数据
        self._load_data(config)

    def _load_data(self, config: Dict[str, Any]):
        """加载所有数据"""
        try:
            paths = config.get("paths", {})
            
            # 加载用户数据
            if "users" in paths and os.path.exists(paths["users"]):
                self._load_users(paths["users"])
            
            # 加载帖子数据
            if "posts" in paths and os.path.exists(paths["posts"]):
                self._load_posts(paths["posts"])
            
            # 加载评论数据
            if "comments" in paths and os.path.exists(paths["comments"]):
                self._load_comments(paths["comments"])
            
            # 加载群组数据
            # if "groups" in paths and os.path.exists(paths["groups"]):
            #     self._load_groups(paths["groups"])
            
            # 加载领域数据
            if "domains" in paths and os.path.exists(paths["domains"]):
                self._load_domains(paths["domains"])
            else:
                # 如果没有领域文件，从用户数据中提取领域信息
                self._extract_domains_from_users()
                
            logger.info(f"Data loaded: {len(self.users)} users, {len(self.posts)} posts, {len(self.domains)} domains")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def _load_users(self, file_path: str):
        """加载用户数据（支持 str.md 格式）"""
        try:
            # 检查文件扩展名
            if file_path.endswith('.json'):
                # 加载 JSON 格式（str.md 格式）
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            user_data = json.loads(line.strip())
                            user = SocialUser.from_dict(user_data)
                            # 只有当user.id不为空时才添加到用户字典
                            if user.id:
                                self.users[user.id] = user
                                # 只有当profile和screen_name都不为空时才添加到索引
                                if user.profile and user.profile.screen_name:
                                    self.user_name_to_id[user.profile.screen_name] = user.id
            else:
                # 加载 CSV 格式（兼容旧格式）
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # 创建 TwitterProfile
                    profile = TwitterProfile(
                        id=str(row.get("user_id", "")),
                        followers_count=int(row.get("followers_count", 0)),
                        friends_count=int(row.get("friends_count", 0)),
                        listed_count=int(row.get("listed_count", 0)),
                        favourites_count=int(row.get("favourites_count", 0)),
                        statuses_count=int(row.get("statuses_count", 0)),
                        age=row.get("age"),
                        name=row.get("name", ""),
                        screen_name=row.get("screen_name", ""),
                        location=row.get("location", ""),
                        description=row.get("description", ""),
                        url=row.get("url", ""),
                        created_at=row.get("created_at", ""),
                        lang=row.get("lang", "en"),
                        gender=row.get("gender", ""),
                        mbit=row.get("mbit", ""),
                        profile_image_url=row.get("profile_image_url", ""),
                        pinned_tweet_id=row.get("pinned_tweet_id"),
                        verified=bool(row.get("verified", False)),
                        protected=bool(row.get("protected", False)),
                        default_profile_image=bool(row.get("default_profile_image", False)),
                        geo_enabled=bool(row.get("geo_enabled", False)),
                        has_extended_profile=bool(row.get("has_extended_profile", False))
                    )
                    
                    user = SocialUser(
                        id=str(row.get("user_id", "")),
                        profile=profile,
                        tweets=json.loads(row.get("tweets", "[]")),
                        neighbor=json.loads(row.get("neighbor", '{"following": [], "follower": []}')),
                        domains=json.loads(row.get("domains", "[]")),
                        label=int(row.get("label", 0))
                    )
                    # 只有当user.id不为空时才添加到用户字典
                    if user.id:
                        self.users[user.id] = user
                        # 只有当profile和screen_name都不为空时才添加到索引
                        if user.profile and user.profile.screen_name:
                            self.user_name_to_id[user.profile.screen_name] = user.id
                    
        except Exception as e:
            logger.error(f"Failed to load users from {file_path}: {e}")

    def _load_posts(self, file_path: str):
        """加载帖子数据"""
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                post = SocialPost(
                    post_id=str(row.get("post_id", "")),
                    author_id=str(row.get("author_id", "")),
                    author_name=row.get("author_name", ""),
                    content=row.get("content", ""),
                    created_time=str_to_datetime(row.get("created_time", datetime.now().isoformat())),
                    platform=row.get("platform", self.platform),
                    score=int(row.get("score", 0)),
                    num_comments=int(row.get("num_comments", 0)),
                    num_likes=int(row.get("num_likes", 0)),
                    num_dislikes=int(row.get("num_dislikes", 0)),
                    num_reposts=int(row.get("num_reposts", 0)),
                    tags=json.loads(row.get("tags", "[]")),
                    metadata=json.loads(row.get("metadata", "{}"))
                )
                # 只有当post.post_id不为空时才添加到帖子字典
                if post.post_id:
                    self.posts[post.post_id] = post
                    
                    # 更新索引（只有当author_id不为空时才更新）
                    if post.author_id:
                        if post.author_id not in self.post_author_index:
                            self.post_author_index[post.author_id] = []
                        self.post_author_index[post.author_id].append(post.post_id)
                
        except Exception as e:
            logger.error(f"Failed to load posts from {file_path}: {e}")

    def _load_comments(self, file_path: str):
        """加载评论数据"""
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                comment = SocialComment(
                    comment_id=str(row.get("comment_id", "")),
                    post_id=str(row.get("post_id", "")),
                    author_id=str(row.get("author_id", "")),
                    author_name=row.get("author_name", ""),
                    content=row.get("content", ""),
                    created_time=str_to_datetime(row.get("created_time", datetime.now().isoformat())),
                    parent_id=row.get("parent_id"),
                    score=int(row.get("score", 0)),
                    num_likes=int(row.get("num_likes", 0)),
                    num_dislikes=int(row.get("num_dislikes", 0)),
                    platform=row.get("platform", self.platform),
                    metadata=json.loads(row.get("metadata", "{}"))
                )
                if comment.comment_id:
                    self.comments[comment.comment_id] = comment
        except Exception as e:
            logger.error(f"Failed to load comments from {file_path}: {e}")

    # def _load_groups(self, file_path: str):
    #     """加载群组数据"""
    #     try:
    #         df = pd.read_csv(file_path)
    #         for _, row in df.iterrows():
    #             group = SocialGroup(
    #                 group_id=str(row.get("group_id", "")),
    #                 group_name=row.get("group_name", ""),
    #                 description=row.get("description", ""),
    #                 created_time=str_to_datetime(row.get("created_time", datetime.now().isoformat())),
    #                 creator_id=str(row.get("creator_id", "")),
    #                 platform=row.get("platform", self.platform),
    #                 member_count=int(row.get("member_count", 0)),
    #                 is_private=bool(row.get("is_private", False)),
    #                 members=json.loads(row.get("members", "[]")),
    #                 messages=json.loads(row.get("messages", "[]")),
    #                 metadata=json.loads(row.get("metadata", "{}"))
    #             )
    #             self.groups[group.group_id] = group
                
    #     except Exception as e:
    #         logger.error(f"Failed to load groups from {file_path}: {e}")

    # 用户相关操作
    def add_user(self, user: SocialUser) -> str:
        """添加用户"""
        # 只有当user.id不为空时才添加到用户字典
        if not user.id:
            raise ValueError("User ID cannot be empty")
        self.users[user.id] = user
        # 只有当profile和screen_name都不为空时才添加到索引
        if user.profile and user.profile.screen_name:
            self.user_name_to_id[user.profile.screen_name] = user.id
        # 检查路径是否存在，如果不存在则不保存到文件
        if "paths" in self.config and "users" in self.config["paths"]:
            user.save_to_file(self.config["paths"]["users"])
        return user.id

    def get_user(self, user_id: str) -> Optional[SocialUser]:
        """获取用户"""
        return self.users.get(user_id)

    def get_user_by_name(self, user_name: str) -> Optional[SocialUser]:
        """根据用户名获取用户"""
        user_id = self.user_name_to_id.get(user_name)
        return self.users.get(user_id) if user_id else None

    def get_all_users(self) -> List[SocialUser]:
        """获取所有用户"""
        return list(self.users.values())

    def get_user_by_screen_name(self, screen_name: str) -> Optional[SocialUser]:
        """根据 screen_name 获取用户"""
        user_id = self.user_name_to_id.get(screen_name)
        return self.users.get(user_id) if user_id else None

    # 帖子相关操作
    def add_post(self, post: SocialPost) -> str:
        """添加帖子"""
        # 只有当post.post_id不为空时才添加到帖子字典
        if not post.post_id:
            raise ValueError("Post ID cannot be empty")
        self.posts[post.post_id] = post
        # 只有当author_id不为空时才更新索引
        if post.author_id:
            if post.author_id not in self.post_author_index:
                self.post_author_index[post.author_id] = []
            self.post_author_index[post.author_id].append(post.post_id)
        # 检查路径是否存在，如果不存在则不保存到文件
        if "paths" in self.config and "posts" in self.config["paths"]:
            post.save_to_file(self.config["paths"]["posts"])
        return post.post_id

    def delete_post(self, post_id: str) -> bool:
        """删除帖子"""
        post = self.posts.pop(post_id, None)
        if not post:
            return False
        if post.author_id and post.author_id in self.post_author_index:
            self.post_author_index[post.author_id] = [
                pid for pid in self.post_author_index[post.author_id] if pid != post_id
            ]
        return True

    def get_post(self, post_id: str) -> Optional[SocialPost]:
        """获取帖子"""
        return self.posts.get(post_id)

    def get_all_posts(self) -> List[SocialPost]:
        """获取所有帖子"""
        return list(self.posts.values())

    def get_posts_by_author(self, author_id: str) -> List[SocialPost]:
        """获取作者的所有帖子"""
        post_ids = self.post_author_index.get(author_id, [])
        return [self.posts[post_id] for post_id in post_ids if post_id in self.posts]

    def get_recommended_posts(self, user_id: str, limit: int = 10) -> List[SocialPost]:
        """获取推荐帖子"""
        # 简单的推荐算法：基于用户关注的人和热门帖子
        user = self.get_user(user_id)
        if not user:
            return []
        
        # 获取关注用户的帖子
        following_posts = []
        for following_id in user.neighbor["following"]:
            following_posts.extend(self.get_posts_by_author(following_id))
        
        # 获取热门帖子（按点赞数排序）
        all_posts = list(self.posts.values())
        hot_posts = sorted(all_posts, key=lambda x: x.num_likes, reverse=True)
        
        # 合并并去重
        recommended = following_posts + hot_posts[:limit//2]
        recommended = list({post.post_id: post for post in recommended}.values())
        
        return recommended[:limit]

    def search_posts(self, query: str, limit: int = 10) -> List[SocialPost]:
        """搜索帖子"""
        results = []
        query_lower = query.lower()
        
        for post in self.posts.values():
            if (query_lower in post.content.lower() or 
                query_lower in post.author_name.lower() or
                any(query_lower in tag.lower() for tag in post.tags)):
                results.append(post)
        
        return results[:limit]

    def search_users(self, query: str, limit: int = 10) -> List[SocialUser]:
        """搜索用户"""
        results = []
        query_lower = query.lower()
        
        for user in self.users.values():
            if (query_lower in user.profile.screen_name.lower() or 
                query_lower in user.profile.name.lower() or
                query_lower in user.profile.description.lower()):
                results.append(user)
        
        return results[:limit]

    # 评论相关操作
    def add_comment(self, comment: SocialComment) -> str:
        """添加评论"""
        if not comment.comment_id:
            raise ValueError("Comment ID cannot be empty")
        self.comments[comment.comment_id] = comment
        if comment.post_id in self.posts:
            self.posts[comment.post_id].num_comments += 1
        if "paths" in self.config and "comments" in self.config["paths"]:
            comment.save_to_file(self.config["paths"]["comments"])
        return comment.comment_id

    def delete_comment(self, comment_id: str) -> bool:
        """删除评论"""
        comment = self.comments.pop(comment_id, None)
        if comment and comment.post_id in self.posts:
            if self.posts[comment.post_id].num_comments > 0:
                self.posts[comment.post_id].num_comments -= 1
        return comment is not None

    def like_comment(self, comment_id: str) -> bool:
        comment = self.comments.get(comment_id)
        if not comment:
            return False
        comment.like()
        return True

    def unlike_comment(self, comment_id: str) -> bool:
        comment = self.comments.get(comment_id)
        if not comment:
            return False
        comment.unlike()
        return True

    def dislike_comment(self, comment_id: str) -> bool:
        comment = self.comments.get(comment_id)
        if not comment:
            return False
        comment.dislike()
        return True

    def undislike_comment(self, comment_id: str) -> bool:
        comment = self.comments.get(comment_id)
        if not comment:
            return False
        comment.undislike()
        return True

    def get_comments_by_post(self, post_id: str) -> List[SocialComment]:
        """获取帖子的评论"""
        return [comment for comment in self.comments.values() if comment.post_id == post_id]

    def get_all_comments(self) -> List[SocialComment]:
        """获取所有评论"""
        return list(self.comments.values())

    # 群组相关操作
    # def add_group(self, group: SocialGroup) -> str:
    #     """添加群组"""
    #     self.groups[group.group_id] = group
    #     group.save_to_file(self.config["paths"]["groups"])
    #     return group.group_id

    # def get_group(self, group_id: str) -> Optional[SocialGroup]:
    #     """获取群组"""
    #     return self.groups.get(group_id)

    # def get_user_groups(self, user_id: str) -> List[SocialGroup]:
    #     """获取用户加入的群组"""
    #     return [group for group in self.groups.values() if user_id in group.members]

    # 统计信息
    def _load_domains(self, file_path: str):
        """加载领域数据"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            domain_data = json.loads(line.strip())
                            domain = Domain.from_dict(domain_data)
                            # 只有当domain.domain_id不为空时才添加到领域字典
                            if domain.domain_id:
                                self.domains[domain.domain_id] = domain
                                # 只有当domain_name不为空时才添加到索引
                                if domain.domain_name:
                                    self.domain_name_to_id[domain.domain_name] = domain.domain_id
            else:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    domain = Domain(
                        domain_id=str(row.get("domain_id", "")),
                        domain_name=row.get("domain_name", ""),
                        description=row.get("description", ""),
                        category=row.get("category", "general"),
                        popularity=int(row.get("popularity", 0)),
                        metadata=json.loads(row.get("metadata", "{}"))
                    )
                    # 只有当domain.domain_id不为空时才添加到领域字典
                    if domain.domain_id:
                        self.domains[domain.domain_id] = domain
                        # 只有当domain_name不为空时才添加到索引
                        if domain.domain_name:
                            self.domain_name_to_id[domain.domain_name] = domain.domain_id
        except Exception as e:
            logger.error(f"Failed to load domains from {file_path}: {e}")

    def _extract_domains_from_users(self):
        """从用户数据中提取领域信息"""
        try:
            domain_counter = {}
            
            # 统计所有用户的领域
            for user in self.users.values():
                for domain_name in user.domains:
                    # 跳过空的领域名称
                    if not domain_name:
                        continue
                    if domain_name not in domain_counter:
                        domain_counter[domain_name] = {
                            'users': [],
                            'posts': [],
                            'count': 0
                        }
                    domain_counter[domain_name]['users'].append(user.id)
                    domain_counter[domain_name]['count'] += 1
            
            # 创建领域对象
            for domain_name, info in domain_counter.items():
                domain_id = f"domain_{len(self.domains) + 1}"
                domain = Domain(
                    domain_id=domain_id,
                    domain_name=domain_name,
                    description=f"Domain for {domain_name}",
                    category="general",
                    popularity=info['count'],
                    metadata={"extracted_from_users": True},
                    users=info['users']
                )
                # 只有当domain.domain_id不为空时才添加到领域字典
                if domain.domain_id:
                    self.domains[domain.domain_id] = domain
                    # 只有当domain_name不为空时才添加到索引
                    if domain.domain_name:
                        self.domain_name_to_id[domain.domain_name] = domain.domain_id
                
            logger.info(f"Extracted {len(self.domains)} domains from user data")
        except Exception as e:
            logger.error(f"Failed to extract domains from users: {e}")

    def add_domain(self, domain: Domain) -> str:
        """添加领域"""
        # 只有当domain.domain_id不为空时才添加到领域字典
        if not domain.domain_id:
            raise ValueError("Domain ID cannot be empty")
        self.domains[domain.domain_id] = domain
        # 只有当domain_name不为空时才添加到索引
        if domain.domain_name:
            self.domain_name_to_id[domain.domain_name] = domain.domain_id
        return domain.domain_id

    def get_domain(self, domain_id: str) -> Optional[Domain]:
        """获取领域"""
        return self.domains.get(domain_id)

    def get_domain_by_name(self, domain_name: str) -> Optional[Domain]:
        """根据名称获取领域"""
        domain_id = self.domain_name_to_id.get(domain_name)
        return self.domains.get(domain_id) if domain_id else None

    def get_all_domains(self) -> List[Domain]:
        """获取所有领域"""
        return list(self.domains.values())

    def get_users_by_domain(self, domain_name: str) -> List[SocialUser]:
        """获取对特定领域感兴趣的用户"""
        domain = self.get_domain_by_name(domain_name)
        if not domain:
            return []
        
        return [self.users[user_id] for user_id in domain.users if user_id in self.users]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_users": len(self.users),
            "total_posts": len(self.posts),
            "total_comments": len(self.comments),
            # "total_groups": len(self.groups),  # 暂时注释掉
            "total_domains": len(self.domains),
            "bot_users": len([u for u in self.users.values() if u.label == 1]),
            "human_users": len([u for u in self.users.values() if u.label == 0])
        }
