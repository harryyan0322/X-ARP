#!/usr/bin/env python3
"""
新的图结构社交网络仿真脚本

基于重构后的 myagent 框架，支持：
1. LLM agent的批量生成与注册
2. 用户档案生成和调整
3. 社交关系建立
4. 大规模LLM agent网络实验
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import random
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from yaml import safe_load

# 导入新的myagent框架
from myagent.agent import SocialAgent, ALL_SOCIAL_ACTIONS
from myagent.environment import SocialEnvironment
from myagent.utils import load_config, generate_id
from myagent.entity import SocialPost

# 导入模型管理器
from model_manager import MultiModelManager

# 日志设置
log = logging.getLogger("new_graph_llm_simulation")
log.setLevel("INFO")

# 清除所有现有的处理器，避免重复
for handler in log.handlers[:]:
    log.removeHandler(handler)

# 添加新的处理器
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(message)s"))
log.addHandler(handler)

# 防止日志传播到根日志器
log.propagate = False

parser = argparse.ArgumentParser(description="New Graph LLM Agent Simulation")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)


class NewGraphLLMSimulation:
    """新的图结构LLM仿真类
    
    主要功能：
    1. 智能体创建和管理
    2. 用户档案生成和验证
    3. 社交关系建立
    4. 数据保存和映射管理
    """
    
    def __init__(self, config_path: str):
        """初始化仿真环境"""
        self.config = load_config(config_path)
        self.environment = None
        self.agents = []
        self.agent_id_to_user_id = {}
        self.user_id_to_agent_id = {}
        self.original_users = []
        
        # Graph-prompt settings (paper §3.2)
        sim_cfg = self.config.get("simulation", {})
        self.graph_prompt_top_k = sim_cfg.get("graph_prompt_top_k", 10)
        self.graph_prompt_max_users = sim_cfg.get("graph_prompt_max_users", 10)
        self.graph_prompt_max_posts_per_user = sim_cfg.get("graph_prompt_max_posts_per_user", 3)
        self.score_eta = sim_cfg.get("score_eta", 100)
        self.score_lambda = sim_cfg.get("score_lambda", 0.017)
        
        # 并发控制参数
        self.max_concurrent_agents = self.config.get("simulation", {}).get("max_concurrent_agents", 10)
        log.info(f"并发控制: 最大同时处理智能体数量 = {self.max_concurrent_agents}")
        
    # ==================== 环境初始化 ====================
    
    async def initialize_environment(self, db_path: str, user_path: str):
        """初始化社交网络环境"""
        log.info("初始化社交网络环境...")
        
        # 创建环境
        self.environment = SocialEnvironment(self.config)
        self.environment.reset()
        
        # 创建并添加SocialAction (使用通用ID)
        from myagent.actions import SocialAction
        self.environment.action = SocialAction(agent_id=0)  # 使用0作为环境级别的action
        
        # 验证模型配置
        model_manager = MultiModelManager(self.config)
        if not model_manager.validate_config():
            log.warning("模型配置验证失败，将使用降级模型")
        else:
            log.info("模型配置验证成功")
        
        # 加载用户数据
        if os.path.exists(user_path):
            with open(user_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            log.info(f"加载用户数据: {len(user_data)} 条记录")
            
            # 将用户数据添加到环境
            for user_dict in user_data:
                if isinstance(user_dict, dict) and 'id' in user_dict:
                    from myagent.entity import SocialUser
                    try:
                        user = SocialUser.from_dict(user_dict)
                        self.environment.get_env_data().add_user(user)
                    except Exception as e:
                        log.warning(f"无法创建用户对象: {e}")
        else:
            log.warning(f"用户数据文件不存在: {user_path}")
        
        log.info("环境初始化完成")
    
    # ==================== 智能体管理 ====================
    
    async def create_agents_from_user_data(self, user_data: List[Dict]):
        """从用户数据创建智能体（异步并发版本）"""
        self.original_users = user_data or []
        # 根据加载的用户数量计算智能体数量：用户数量 * 0.14/0.86
        total_users = len(user_data)
        calculated_agents = int(total_users * 0.14 / 0.86)
        
        log.info(f"加载用户数量: {total_users}")
        log.info(f"计算智能体数量: {calculated_agents} (用户数量 * 0.14/0.86)")
        
        # 预先随机选择用户数据，避免重复
        selected_users = random.sample(user_data, min(calculated_agents, len(user_data)))
        log.info(f"已随机选择 {len(selected_users)} 个用户数据用于智能体创建")
        
        # 创建智能体的异步函数
        async def create_single_agent(agent_id: int) -> Optional[SocialAgent]:
            """创建单个智能体（基于预选用户数据）"""
            try:
                # 使用预选的用户数据
                selected_user_dict = selected_users[agent_id - 1]
                
                # 创建 SocialUser 对象
                from myagent.entity import SocialUser
                user = SocialUser.from_dict(selected_user_dict)
                
                # 创建模型管理器
                model_manager = MultiModelManager(self.config)
                model = model_manager.create_model(agent_id)
                
                # 创建智能体（传入环境）
                agent = SocialAgent(
                    agent_id=agent_id,
                    user_info=user,
                    model=model
                )
                
                # 设置环境引用
                agent.env = self.environment
                
                # 设置智能体图引用（用于环境交互）
                agent.agent_graph = self
                
                # 不在这里建立ID映射关系，等待profile生成时建立
                
                log.info(f"智能体 {agent_id} 创建完成")
                return agent
                
            except Exception as e:
                log.error(f"创建智能体 {agent_id} 失败: {e}")
                return None
        
        # 并发创建智能体
        log.info(f"开始并发创建 {calculated_agents} 个智能体（随机选择用户数据）...")
        start_time = datetime.now()
        
        # 创建任务列表
        tasks = []
        for i in range(calculated_agents):
            agent_id = i + 1
            task = create_single_agent(agent_id)
            tasks.append(task)
        
        log.info(f"任务列表创建完成，共 {len(tasks)} 个任务")
        
        # 并发执行所有任务（添加超时控制）
        try:
            agents_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5分钟超时
            )
        except asyncio.TimeoutError:
            log.error("智能体创建超时，可能网络或API响应较慢")
            # 取消所有未完成的任务
            for task in tasks:
                task.cancel()
            raise
        
        # 处理结果
        agents = []
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(agents_results):
            if isinstance(result, Exception):
                log.error(f"智能体 {i+1} 创建异常: {result}")
                failed_count += 1
            elif result is not None:
                agents.append(result)
                successful_count += 1
            else:
                failed_count += 1
        
        self.agents = agents
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log.info(f"智能体创建完成: 成功 {successful_count} 个，失败 {failed_count} 个，总数: {len(agents)}，耗时: {duration:.2f}秒")
        return agents
    
    def get_agents(self):
        """获取所有智能体"""
        return [(agent.social_agent_id, agent) for agent in self.agents]

    # ==================== Graph Prompt (Paper §3.2) ====================

    def _summarize_user(self, user_dict: Dict) -> str:
        """简要摘要用户信息用于图结构提示"""
        if not isinstance(user_dict, dict):
            return "Unknown user"
        profile = user_dict.get("profile", user_dict)
        name = profile.get("name", "") or profile.get("screen_name", "")
        uid = profile.get("id", user_dict.get("id", ""))
        desc = profile.get("description", "")
        followers = profile.get("followers_count", 0)
        friends = profile.get("friends_count", 0)
        domains = user_dict.get("domains", [])
        domains_text = ", ".join(domains[:3]) if isinstance(domains, list) else ""
        return f"User {uid} ({name}) | followers={followers}, friends={friends} | interests={domains_text} | bio={desc[:120]}"

    def _collect_candidate_posts(self, candidate_users: List[Dict], max_posts_per_user: int) -> List[Dict]:
        """从候选用户中收集帖子，构造候选内容池"""
        posts = []
        for user in candidate_users:
            if not isinstance(user, dict):
                continue
            profile = user.get("profile", user)
            author_id = profile.get("id", user.get("id", ""))
            author_name = profile.get("name", profile.get("screen_name", ""))
            tweets = user.get("tweets", []) or []
            for idx, content in enumerate(tweets[:max_posts_per_user]):
                posts.append({
                    "author_id": str(author_id),
                    "author_name": str(author_name),
                    "content": str(content),
                    "num_likes": profile.get("followers_count", 0),
                    "created_idx": idx,
                })
        return posts

    def _score_posts(self, user_desc: str, posts: List[Dict]) -> List[Dict]:
        """基于论文公式近似打分：cos(sim) * log(eta*(C+1)) * exp(-lambda*delta_t)."""
        if not user_desc or not posts:
            return [{**p, "score": 0.0} for p in posts]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            return [{**p, "score": 0.0} for p in posts]

        corpus = [user_desc] + [p.get("content", "") for p in posts]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
        user_vec = tfidf[0:1]
        post_vecs = tfidf[1:]
        sims = cosine_similarity(post_vecs, user_vec).flatten()

        scored = []
        for i, post in enumerate(posts):
            sim = float(sims[i]) if i < len(sims) else 0.0
            c_c = float(post.get("num_likes", 0))
            delta_t = float(post.get("created_idx", 0))
            score = sim * math.log(self.score_eta * (c_c + 1) + 1e-6) * math.exp(-self.score_lambda * delta_t)
            scored.append({**post, "score": score})
        return scored

    def _build_graph_prompt(self, user_dict: Dict, candidate_users: List[Dict]) -> str:
        """构建图结构提示，包含候选邻居与Top-K内容。"""
        if not candidate_users:
            return "No graph context available."

        # User summary
        user_summary = self._summarize_user(user_dict)

        # Candidate users summary
        candidate_summaries = []
        for user in candidate_users[: self.graph_prompt_max_users]:
            candidate_summaries.append(self._summarize_user(user))

        # Candidate posts (Top-K)
        posts = self._collect_candidate_posts(candidate_users, self.graph_prompt_max_posts_per_user)
        scored_posts = self._score_posts(user_summary, posts)
        top_posts = sorted(scored_posts, key=lambda x: x.get("score", 0.0), reverse=True)[: self.graph_prompt_top_k]

        post_lines = []
        for idx, p in enumerate(top_posts, 1):
            content = p.get("content", "")
            post_lines.append(
                f"{idx}. [{p.get('author_id')}] {content[:160]} (score={p.get('score', 0.0):.4f})"
            )

        prompt = (
            "Target user:\\n"
            f"{user_summary}\\n\\n"
            "Candidate neighbors:\\n"
            + "\\n".join(f"- {s}" for s in candidate_summaries)
            + "\\n\\n"
            "Top-K candidate posts:\\n"
            + ("\\n".join(post_lines) if post_lines else "No candidate posts.")
        )
        return prompt
    
    # ==================== 用户档案生成 ====================
    
    async def generate_profile_with_retry(self, agent: SocialAgent, max_retries: int = 3):
        """为智能体生成档案，支持重试，确保与str.md格式完全对应"""
        for attempt in range(max_retries):
            try:
                log.info(f"Agent {agent.social_agent_id} 第 {attempt + 1} 次尝试生成档案...")
                
                if hasattr(agent, 'generate_new_user_profile_by_llm'):
                    profile_json_str = await agent.generate_new_user_profile_by_llm()
                    log.info(f"Agent {agent.social_agent_id} LLM返回内容: {profile_json_str[:200]}...")
                    
                    # 解析和验证profile JSON
                    profile = self._parse_and_validate_profile(profile_json_str, agent.social_agent_id)
                    if profile:
                        log.info(f"Agent {agent.social_agent_id} 解析后的profile: name={profile.get('name', 'N/A')}, description={profile.get('description', 'N/A')[:50]}...")
                        return True, profile, ""
                    else:
                        log.error(f"Agent {agent.social_agent_id} Profile解析失败，原始内容: {profile_json_str}")
                        return False, None, "Profile解析或验证失败"
                else:
                    return False, None, "智能体不支持档案生成"
                    
            except Exception as e:
                error_msg = f"生成档案失败: {str(e)}"
                log.warning(f"Agent {agent.social_agent_id} 第 {attempt + 1} 次失败: {error_msg}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    log.error(f"Agent {agent.social_agent_id} 多次生成失败")
                    return False, None, error_msg
        
        return False, None, "未知错误"
    
    def _parse_and_validate_profile(self, profile_json_str: str, agent_id: int) -> dict:
        """解析和验证profile JSON，确保与str.md格式完全对应"""
        import json
        import re
        
        try:
            # 清理和解析JSON
            profile_json_str = profile_json_str.strip()
            match = re.search(r'\{.*\}', profile_json_str, re.DOTALL)
            if match:
                profile_json_str = match.group(0)
            

            
            profile = json.loads(profile_json_str)
            
            # 定义str.md中profile字段的完整规范
            profile_fields = [
                ("id", str, ""),
                ("followers_count", int, 0),
                ("friends_count", int, 0),
                ("listed_count", int, 0),
                ("favourites_count", int, 0),
                ("statuses_count", int, 0),
                ("age", int, None),
                ("name", str, ""),
                ("screen_name", str, ""),
                ("location", str, ""),
                ("description", str, ""),
                ("url", str, ""),
                ("created_at", str, ""),
                ("lang", str, "en"),
                ("gender", str, ""),
                ("mbit", str, ""),
                ("profile_image_url", str, ""),
                ("pinned_tweet_id", (str, type(None)), None),
                ("verified", bool, False),
                ("protected", bool, False),
                ("default_profile_image", bool, False),
                ("geo_enabled", bool, False),
                ("has_extended_profile", bool, False),
            ]
            
            # 验证和补全字段
            validated_profile = {}
            for field_name, field_type, default_value in profile_fields:
                if field_name not in profile:
                    validated_profile[field_name] = default_value
                else:
                    value = profile[field_name]
                    # 类型校验和转换
                    if not isinstance(value, field_type):
                        if isinstance(field_type, tuple) and type(None) in field_type and value is None:
                            validated_profile[field_name] = value
                        else:
                            # 尝试类型转换
                            try:
                                if field_type == int:
                                    validated_profile[field_name] = int(value) if value is not None else default_value
                                elif field_type == str:
                                    validated_profile[field_name] = str(value) if value is not None else default_value
                                elif field_type == bool:
                                    validated_profile[field_name] = bool(value) if value is not None else default_value
                                else:
                                    validated_profile[field_name] = value
                            except (ValueError, TypeError):
                                validated_profile[field_name] = default_value
                    else:
                        validated_profile[field_name] = value
            
            # 确保ID唯一性
            if not validated_profile["id"] or validated_profile["id"] == "":
                validated_profile["id"] = f"u{agent_id}"
            
            # 更新ID映射关系，确保与生成的profile ID一致
            generated_user_id = validated_profile["id"]
            
            # 最终检查ID唯一性，如果有重复直接修改
            if generated_user_id in self.user_id_to_agent_id:
                # 生成新的唯一ID
                counter = 1
                while f"{generated_user_id}_{counter}" in self.user_id_to_agent_id:
                    counter += 1
                validated_profile["id"] = f"{generated_user_id}_{counter}"
                generated_user_id = validated_profile["id"]
            
            self.agent_id_to_user_id[agent_id] = generated_user_id
            self.user_id_to_agent_id[generated_user_id] = agent_id
            
            log.info(f"Profile验证成功，包含 {len(validated_profile)} 个字段")
            return validated_profile
            
        except Exception as e:
            log.error(f"Profile解析失败: {e}")
            return None
    
    async def generate_complete_user_data(self, agent: SocialAgent) -> dict:
        """生成完整的用户数据，包括profile、tweets、domains等所有字段"""
        try:
            # 1. 生成profile
            success, profile, error = await self.generate_profile_with_retry(agent)
            if not success:
                log.error(f"Agent {agent.social_agent_id} 生成profile失败: {error}")
                return None
            
            # 2. 构建基础用户字典
            user_dict = {
                "id": profile["id"],
                "profile": profile,
                "tweets": [],
                "neighbor": {"following": [], "follower": []},
                "domains": [],
                "label": 1,  # 默认为机器人
            }
            
            # 3. 生成兴趣领域(domains)
            try:
                if hasattr(agent, 'generate_user_interests_by_llm'):
                    topics = await agent.generate_user_interests_by_llm()
                    if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                        user_dict["domains"] = topics
                        log.info(f"Agent {agent.social_agent_id} 生成domains: {topics}")
                    else:
                        user_dict["domains"] = []
                else:
                    user_dict["domains"] = []
            except Exception as e:
                log.warning(f"Agent {agent.social_agent_id} 生成domains失败: {e}")
                user_dict["domains"] = []
            
            # 4. 生成推文(tweets)
            try:
                if hasattr(agent, 'generate_user_tweets_by_llm'):
                    # 从配置文件获取最大推文数量
                    max_tweets = self.config.get("simulation", {}).get("max_tweets_per_user", 100)
                    # 构建图结构提示（使用原始用户作为候选池）
                    candidate_pool_size = self.config.get("simulation", {}).get("candidate_pool_size", 50)
                    candidate_users = []
                    if self.original_users:
                        sample_size = min(candidate_pool_size, len(self.original_users))
                        candidate_users = random.sample(self.original_users, sample_size)
                    graph_prompt = self._build_graph_prompt(user_dict, candidate_users) if candidate_users else ""

                    tweets_json = await agent.generate_user_tweets_by_llm(
                        user_dict,
                        max_tweets=max_tweets,
                        graph_prompt=graph_prompt
                    )
                    tweets = json.loads(tweets_json)
                    if isinstance(tweets, list) and all(isinstance(t, str) for t in tweets):
                        user_dict["tweets"] = tweets
                        log.info(f"Agent {agent.social_agent_id} 生成 {len(tweets)} 条推文")
                        # 将推文写入环境帖子池（用于推荐系统）
                        if self.environment and self.environment.get_env_data():
                            env_data = self.environment.get_env_data()
                            author_id = user_dict.get("id", f"u{agent.social_agent_id}")
                            author_name = profile.get("name", "") if isinstance(profile, dict) else ""
                            for tweet in tweets:
                                try:
                                    post_id = generate_id(list(env_data.posts.keys()))
                                    post = SocialPost(
                                        post_id=post_id,
                                        author_id=str(author_id),
                                        author_name=str(author_name),
                                        content=tweet,
                                        created_time=datetime.now(),
                                        platform="twitter",
                                    )
                                    env_data.add_post(post)
                                except Exception as e:
                                    log.warning(f"写入环境帖子失败: {e}")
                    else:
                        user_dict["tweets"] = []
                else:
                    user_dict["tweets"] = []
            except Exception as e:
                log.warning(f"Agent {agent.social_agent_id} 生成tweets失败: {e}")
                user_dict["tweets"] = []
            
            log.info(f"Agent {agent.social_agent_id} 完整用户数据生成成功")
            return user_dict
            
        except Exception as e:
            log.error(f"Agent {agent.social_agent_id} 生成完整用户数据失败: {e}")
            return None
    
    async def adjust_profile_with_explanations(self, agent: SocialAgent, explanations_path: str):
        """使用解释文件调整智能体档案"""
        try:
            if hasattr(agent, 'adjust_and_regenerate_user_profile'):
                adjusted_user_dict = await agent.adjust_and_regenerate_user_profile(
                    explanations_path=explanations_path,
                    ori_data_template=agent.user_info.to_dict() if agent.user_info else None,
                    agent_user_id_mapping=self.agent_id_to_user_id
                )
                return True, adjusted_user_dict, ""
            else:
                return False, None, "智能体不支持档案调整"
        except Exception as e:
            return False, None, f"档案调整失败: {str(e)}"
    
    # ==================== 社交关系生成 ====================
    
    async def generate_social_relationships(self, agents: List[SocialAgent], user_path: str, sample_size: int = None):
        """生成社交关系，候选池包含原始用户和新用户，保证1:1比例"""
        log.info("开始生成社交关系...")
        
        # 从配置文件获取候选池大小
        if sample_size is None:
            sample_size = self.config.get("simulation", {}).get("candidate_pool_size", 50)
        
        # 加载原始用户数据
        original_users = self._load_original_users(user_path)
        log.info(f"加载原始用户数据: {len(original_users)} 个用户")
        
        for agent in agents:
            try:
                # 获取其他新智能体作为候选
                other_agents = [a for a in agents if a.social_agent_id != agent.social_agent_id]
                
                # 计算候选池大小，确保1:1比例
                max_new_agents = min(sample_size // 2, len(other_agents))
                max_original_users = sample_size - max_new_agents
                
                # 随机选择新智能体候选
                new_agent_candidates = []
                if other_agents:
                    new_agent_candidates = random.sample(other_agents, max_new_agents)
                
                # 随机选择原始用户候选
                original_user_candidates = []
                if original_users:
                    original_user_candidates = random.sample(original_users, max_original_users)
                
                # 合并候选列表
                all_candidates = []
                
                # 添加新智能体候选
                for candidate in new_agent_candidates:
                    if hasattr(candidate, 'user_info') and candidate.user_info:
                        all_candidates.append(candidate.user_info.to_dict())
                
                # 添加原始用户候选
                all_candidates.extend(original_user_candidates)
                
                log.info(f"Agent {agent.social_agent_id} 候选池: {len(new_agent_candidates)} 个新智能体 + {len(original_user_candidates)} 个原始用户")
                
                # 使用智能体的关注决策方法
                if hasattr(agent, 'decide_user_following_by_llm') and all_candidates:
                    current_user_dict = agent.user_info.to_dict()
                    graph_prompt = self._build_graph_prompt(current_user_dict, all_candidates)
                    await agent.decide_user_following_by_llm(
                        current_user_dict, 
                        all_candidates, 
                        min_friends=0,
                        graph_prompt=graph_prompt
                    )
                
            except Exception as e:
                log.error(f"为智能体 {agent.social_agent_id} 生成社交关系失败: {e}")
    
    async def generate_social_relationships_async(self, agents: List[SocialAgent], user_path: str, sample_size: int = None):
        """生成社交关系（异步并发版本），候选池包含原始用户和新用户，保证1:1比例"""
        log.info("开始并发生成社交关系...")
        
        # 从配置文件获取候选池大小
        if sample_size is None:
            sample_size = self.config.get("simulation", {}).get("candidate_pool_size", 50)
        
        # 加载原始用户数据
        original_users = self._load_original_users(user_path)
        log.info(f"加载原始用户数据: {len(original_users)} 个用户")
        
        # 生成单个智能体社交关系的异步函数
        async def generate_single_agent_relationships(agent: SocialAgent) -> Dict:
            """为单个智能体生成社交关系"""
            try:
                # 获取其他新智能体作为候选
                other_agents = [a for a in agents if a.social_agent_id != agent.social_agent_id]
                
                # 计算候选池大小，确保1:1比例
                max_new_agents = min(sample_size // 2, len(other_agents))
                max_original_users = sample_size - max_new_agents
                
                # 随机选择新智能体候选
                new_agent_candidates = []
                if other_agents:
                    new_agent_candidates = random.sample(other_agents, max_new_agents)
                
                # 随机选择原始用户候选
                original_user_candidates = []
                if original_users:
                    original_user_candidates = random.sample(original_users, max_original_users)
                
                # 合并候选列表
                all_candidates = []
                
                # 添加新智能体候选
                for candidate in new_agent_candidates:
                    if hasattr(candidate, 'user_info') and candidate.user_info:
                        all_candidates.append(candidate.user_info.to_dict())
                
                # 添加原始用户候选
                all_candidates.extend(original_user_candidates)
                
                log.info(f"Agent {agent.social_agent_id} 候选池: {len(new_agent_candidates)} 个新智能体 + {len(original_user_candidates)} 个原始用户")
                
                # 使用智能体的关注决策方法
                if hasattr(agent, 'decide_user_following_by_llm') and all_candidates:
                    current_user_dict = agent.user_info.to_dict()
                    graph_prompt = self._build_graph_prompt(current_user_dict, all_candidates)
                    await agent.decide_user_following_by_llm(
                        current_user_dict, 
                        all_candidates, 
                        min_friends=0,
                        graph_prompt=graph_prompt
                    )
                
                return {
                    "agent_id": agent.social_agent_id,
                    "status": "success",
                    "candidates_count": len(all_candidates)
                }
                
            except Exception as e:
                log.error(f"为智能体 {agent.social_agent_id} 生成社交关系失败: {e}")
                return {
                    "agent_id": agent.social_agent_id,
                    "status": "failed",
                    "error": str(e)
                }
        
        # 使用信号量控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        
        async def limited_relationship_generation(agent: SocialAgent):
            """限制并发的社交关系生成函数"""
            async with semaphore:
                return await generate_single_agent_relationships(agent)
        
        # 创建限制并发的任务列表
        limited_tasks = [limited_relationship_generation(agent) for agent in agents]
        
        # 并发执行所有社交关系生成任务（带并发限制）
        relationship_results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # 处理结果
        successful_relationships = 0
        failed_relationships = 0
        
        for i, result in enumerate(relationship_results):
            if isinstance(result, Exception):
                log.error(f"智能体 {agents[i].social_agent_id} 社交关系生成异常: {result}")
                failed_relationships += 1
            else:
                if result["status"] == "success":
                    successful_relationships += 1
                else:
                    failed_relationships += 1
        
        log.info(f"社交关系生成完成: 成功 {successful_relationships} 个，失败 {failed_relationships} 个")

    # ==================== 时间步模拟 ====================

    async def simulate_time_steps(self, agents: List[SocialAgent], steps: int = 1, feed_k: int = 10):
        """按时间步模拟代理行为，环境推送Top-K内容"""
        if not self.environment:
            return
        log.info(f"开始时间步模拟: steps={steps}, top_k={feed_k}")
        for step in range(steps):
            self.environment.tick(1.0)
            for agent in agents:
                try:
                    user_id = agent.user_info.profile.id if agent.user_info else f"u{agent.social_agent_id}"
                    context = self.environment.format_user_feed_context(user_id, limit=feed_k)
                    env_data = {
                        "current_time": self.environment.current_time.isoformat() if self.environment.current_time else "",
                        "feed": context,
                    }
                    await agent.perform_action_by_llm(
                        context=context,
                        environment_data=env_data,
                        available_actions=ALL_SOCIAL_ACTIONS
                    )
                except Exception as e:
                    log.warning(f"时间步{step}代理{agent.social_agent_id}执行失败: {e}")
    
    def _load_original_users(self, user_path: str) -> List[Dict]:
        """加载原始用户数据"""
        try:
            with open(user_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            # 确保数据是列表格式
            if isinstance(user_data, list):
                log.info(f"加载原始用户数据: {len(user_data)} 个用户")
                return user_data
            elif isinstance(user_data, dict) and 'agent_data' in user_data:
                # 如果是完整结果格式，提取用户数据
                extracted_users = [item.get('user_info', {}) for item in user_data['agent_data'] if item.get('user_info')]
                log.info(f"从完整结果格式提取原始用户数据: {len(extracted_users)} 个用户")
                return extracted_users
            elif isinstance(user_data, dict) and 'user_data' in user_data:
                # 如果是另一种完整结果格式
                extracted_users = user_data['user_data']
                log.info(f"从user_data格式提取原始用户数据: {len(extracted_users)} 个用户")
                return extracted_users
            else:
                log.warning(f"无法识别的用户数据格式: {type(user_data)}")
                return []
                
        except Exception as e:
            log.error(f"加载原始用户数据失败: {e}")
            return []
    
    # ==================== 数据保存 ====================
    
    async def save_results(self, output_path: str, mode: str = "generation", config: dict = None, generated_data: list = None):
        """保存结果到JSON文件"""
        log.info(f"保存结果到: {output_path}")
        
        # 从配置中获取文件后缀
        data_params = config.get("data", {}) if config else {}
        full_result_suffix = data_params.get("full_result_suffix", "_full_result_with_metadata")
        mapping_suffix = data_params.get("mapping_suffix", "_agent_user_mapping")
        
        # 收集所有智能体的数据
        agent_data = []
        str_md_data = []  # 符合str.md格式的数据
        
        # 如果有生成的数据，优先使用生成的数据
        if generated_data:
            # 创建生成数据的映射
            generated_data_map = {}
            for item in generated_data:
                if item.get("status") == "success" and "user_dict" in item:
                    generated_data_map[item["agent_id"]] = item["user_dict"]
            
            for agent in self.agents:
                if hasattr(agent, 'user_info') and agent.user_info:
                    # 优先使用生成的数据，如果没有则使用智能体的数据
                    if agent.social_agent_id in generated_data_map:
                        user_dict = generated_data_map[agent.social_agent_id]
                        log.info(f"使用生成的数据保存 Agent {agent.social_agent_id}")
                    else:
                        user_dict = agent.user_info.to_dict()
                        log.warning(f"Agent {agent.social_agent_id} 没有生成数据，使用原始数据")
                    
                    agent_data.append({
                        "agent_id": agent.social_agent_id,
                        "user_info": user_dict,
                        "user_id": user_dict.get("id", "")
                    })
                    
                    # 添加到str.md格式数据
                    str_md_data.append(user_dict)
        else:
            # 如果没有生成数据，使用智能体的原始数据
            for agent in self.agents:
                if hasattr(agent, 'user_info') and agent.user_info:
                    # 转换为字典格式
                    user_dict = agent.user_info.to_dict()
                    
                    agent_data.append({
                        "agent_id": agent.social_agent_id,
                        "user_info": user_dict,
                        "user_id": user_dict.get("id", "")
                    })
                    
                    # 添加到str.md格式数据
                    str_md_data.append(user_dict)
        
        # 构建完整结果数据
        result_data = {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.agents),
            "agent_data": agent_data,
            "agent_id_to_user_id": self.agent_id_to_user_id,
            "user_id_to_agent_id": self.user_id_to_agent_id
        }
        
        # 保存完整结果到JSON文件（包含元数据）
        full_result_path = output_path.replace('.json', f'{full_result_suffix}.json')
        os.makedirs(os.path.dirname(full_result_path), exist_ok=True)
        with open(full_result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 保存str.md格式数据到主文件（纯用户数据）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(str_md_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"结果已保存，包含 {len(agent_data)} 个智能体数据")
        log.info(f"完整结果文件（含元数据）: {full_result_path}")
        log.info(f"用户数据文件（str.md格式）: {output_path}")
        
        return full_result_path
    
    async def save_agent_user_mapping(self, output_path: str):
        """保存agent与用户id的映射关系，用于后续调整模块"""
        log.info(f"保存agent与用户id映射关系到: {output_path}")
        
        # 只保存必要的映射字典
        mapping_data = {
            "agent_id_to_user_id": self.agent_id_to_user_id,
            "user_id_to_agent_id": self.user_id_to_agent_id
        }
        
        # 保存映射文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"映射关系已保存，包含 {len(self.agent_id_to_user_id)} 个映射记录")
        
        return mapping_data
    
    # ==================== 运行模式 ====================
    
    async def run_generation_mode(self, user_path: str, output_path: str = "../../data/new_bot_data_complete.json"):
        """运行生成模式：创建智能体并生成完整的用户数据"""
        log.info("🆕 开始生成模式...")
        
        # 1. 加载用户数据
        with open(user_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        # 2. 创建智能体
        agents = await self.create_agents_from_user_data(user_data)
        
        # 3. 生成完整的用户数据（异步并发版本）
        log.info(f"开始并发生成 {len(agents)} 个智能体的完整用户数据...")
        
        # 生成用户数据的异步函数
        async def generate_single_agent_data(agent: SocialAgent) -> Dict:
            """为单个智能体生成完整用户数据"""
            try:
                log.info(f"开始生成 Agent {agent.social_agent_id} 的完整用户数据...")
                
                user_dict = await self.generate_complete_user_data(agent)
                if user_dict:
                    # 更新智能体的用户信息
                    if hasattr(agent, 'user_info') and agent.user_info:
                        if hasattr(agent.user_info, 'profile'):
                            # 确保profile是TwitterProfile对象而不是字典
                            from myagent.entity import TwitterProfile
                            if isinstance(user_dict["profile"], dict):
                                agent.user_info.profile = TwitterProfile.from_dict(user_dict["profile"])
                            else:
                                agent.user_info.profile = user_dict["profile"]
                        agent.user_info.tweets = user_dict.get("tweets", [])
                        agent.user_info.domains = user_dict.get("domains", [])
                        agent.user_info.neighbor = user_dict.get("neighbor", {"following": [], "follower": []})
                        agent.user_info.label = user_dict.get("label", 1)
                    
                    log.info(f"Agent {agent.social_agent_id} 生成成功: profile={len(user_dict['profile'])}字段, tweets={len(user_dict['tweets'])}条, domains={len(user_dict['domains'])}个")
                    
                    return {
                        "agent_id": agent.social_agent_id,
                        "user_dict": user_dict,
                        "status": "success"
                    }
                else:
                    log.error(f"Agent {agent.social_agent_id} 生成失败")
                    return {
                        "agent_id": agent.social_agent_id,
                        "error": "完整用户数据生成失败",
                        "status": "failed"
                    }
                    
            except Exception as e:
                log.error(f"Agent {agent.social_agent_id} 生成异常: {e}")
                return {
                    "agent_id": agent.social_agent_id,
                    "error": str(e),
                    "status": "failed"
                }
        
        # 使用信号量控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        
        async def limited_generation(agent: SocialAgent):
            """限制并发的生成函数"""
            async with semaphore:
                return await generate_single_agent_data(agent)
        
        # 创建限制并发的任务列表
        limited_tasks = [limited_generation(agent) for agent in agents]
        
        # 并发执行所有生成任务（带并发限制）
        generated_data_results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # 处理生成结果
        generated_data = []
        successful_generations = 0
        failed_generations = 0
        
        for i, result in enumerate(generated_data_results):
            if isinstance(result, Exception):
                log.error(f"智能体 {agents[i].social_agent_id} 生成异常: {result}")
                generated_data.append({
                    "agent_id": agents[i].social_agent_id,
                    "error": str(result),
                    "status": "failed"
                })
                failed_generations += 1
            else:
                generated_data.append(result)
                if result["status"] == "success":
                    successful_generations += 1
                else:
                    failed_generations += 1
        
        log.info(f"用户数据生成完成: 成功 {successful_generations} 个，失败 {failed_generations} 个")
        
        # 4. 生成社交关系（异步并发版本）
        log.info("开始并发生成社交关系...")
        # 从配置文件获取候选池大小
        candidate_pool_size = self.config.get("simulation", {}).get("candidate_pool_size", 50)
        max_tweets = self.config.get("simulation", {}).get("max_tweets_per_user", 100)
        log.info(f"配置参数 - 候选池大小: {candidate_pool_size}, 最大推文数量: {max_tweets}")
        await self.generate_social_relationships_async(agents, user_path, candidate_pool_size)

        # 4.5 时间步模拟（论文时序推送）
        time_steps = self.config.get("simulation", {}).get("time_steps", 1)
        feed_k = self.config.get("simulation", {}).get("recsys_top_k", 10)
        await self.simulate_time_steps(agents, steps=time_steps, feed_k=feed_k)

        # 5. 保存结果
        await self.save_results(output_path, "generation", self.config, generated_data)
        
        # 6. 保存映射关系
        mapping_path = output_path.replace('.json', f'{self.config.get("data", {}).get("mapping_suffix", "_agent_user_mapping")}.json')
        await self.save_agent_user_mapping(mapping_path)
        
        # 7. 统计结果
        successful_generations = len([d for d in generated_data if d["status"] == "success"])
        failed_generations = len([d for d in generated_data if d["status"] == "failed"])
        
        log.info(f"生成模式完成: 成功 {successful_generations} 个，失败 {failed_generations} 个")
        
        # 清理生成数据，减少内存占用
        cleaned_generated_data = []
        for item in generated_data:
            if item.get("status") == "success":
                # 对于成功的数据，只保留基本信息，不保留完整的user_dict
                cleaned_item = {
                    "agent_id": item.get("agent_id"),
                    "status": "success",
                    "profile_fields": len(item.get("user_dict", {}).get("profile", {})),
                    "tweets_count": len(item.get("user_dict", {}).get("tweets", [])),
                    "domains_count": len(item.get("user_dict", {}).get("domains", [])),
                    "following_count": len(item.get("user_dict", {}).get("neighbor", {}).get("following", []))
                }
            else:
                # 对于失败的数据，保留错误信息
                cleaned_item = {
                    "agent_id": item.get("agent_id"),
                    "status": "failed",
                    "error": item.get("error", "未知错误")
                }
            cleaned_generated_data.append(cleaned_item)
        
        return {
            "mode": "generation",
            "total_agents": len(agents),
            "successful_generations": successful_generations,
            "failed_generations": failed_generations,
            "generated_data": cleaned_generated_data
        }
    
    async def run_adjustment_mode(self, user_path: str, explanations_path: str, agent_ids_to_adjust: List[int] = None, output_path: str = "../../data/new_bot_data_complete.json"):
        """运行调整模式：根据解释文件调整智能体档案"""
        log.info("🔧 开始调整模式...")
        
        # 1. 加载用户数据
        with open(user_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        # 2. 创建智能体
        agents = await self.create_agents_from_user_data(user_data)
        
        # 3. 确定需要调整的智能体
        if agent_ids_to_adjust:
            target_agents = [a for a in agents if a.social_agent_id in agent_ids_to_adjust]
            log.info(f"目标调整模式: {len(target_agents)} 个智能体")
        else:
            # 全局调整模式：随机抽样
            sample_size = math.ceil(len(agents) * 0.14 / 0.86)
            target_agents = random.sample(agents, min(sample_size, len(agents)))
            log.info(f"全局调整模式: {len(target_agents)} 个智能体")
        
        # 4. 调整档案
        adjusted_data = []
        for i, agent in enumerate(target_agents):
            log.info(f"调整档案 {i+1}/{len(target_agents)}: Agent {agent.social_agent_id}")
            
            success, user_dict, error = await self.adjust_profile_with_explanations(agent, explanations_path)
            if success:
                # 更新智能体数据
                try:
                    if hasattr(agent, 'user_info') and agent.user_info and isinstance(user_dict, dict):
                        from myagent.entity import TwitterProfile
                        if isinstance(user_dict.get("profile"), dict):
                            agent.user_info.profile = TwitterProfile.from_dict(user_dict["profile"])
                        agent.user_info.tweets = user_dict.get("tweets", [])
                        agent.user_info.domains = user_dict.get("domains", [])
                        agent.user_info.neighbor = user_dict.get("neighbor", {"following": [], "follower": []})
                        agent.user_info.label = user_dict.get("label", 1)
                except Exception as e:
                    log.warning(f"更新智能体数据失败: {e}")
                adjusted_data.append({
                    "agent_id": agent.social_agent_id,
                    "user_dict": user_dict,
                    "status": "success"
                })
            else:
                adjusted_data.append({
                    "agent_id": agent.social_agent_id,
                    "error": error,
                    "status": "failed"
                })
        
        # 5. 保存结果（使用调整后的数据）
        await self.save_results(output_path, "adjustment", self.config, generated_data=adjusted_data)
        
        # 6. 保存映射关系
        mapping_path = output_path.replace('.json', f'{self.config.get("data", {}).get("mapping_suffix", "_agent_user_mapping")}.json')
        await self.save_agent_user_mapping(mapping_path)
        
        return {
            "mode": "adjustment",
            "total_agents": len(agents),
            "target_agents": len(target_agents),
            "successful_adjustments": len([d for d in adjusted_data if d["status"] == "success"]),
            "failed_adjustments": len([d for d in adjusted_data if d["status"] == "failed"])
        }


async def main(
    config_path: str,
    user_path: str = None,
    explanations_path: str = None,
    agent_ids_to_adjust: List[int] = None,
    output_path: str = None
):
    """主函数"""
    log.info("🎉 开始新的图结构LLM仿真...")
    
    # 加载配置文件
    with open(config_path, "r") as f:
        cfg = safe_load(f)
    
    # 从配置文件读取参数，如果函数参数为None则使用配置文件中的值
    data_params = cfg.get("data", {})
    simulation_params = cfg.get("simulation", {})
    inference_params = cfg.get("inference", {})
    
    # 设置路径参数
    user_path = user_path or data_params.get("user_path", "")
    output_path = output_path or data_params.get("output_path", "../../data/new_bot_data_complete.json")
    db_path = data_params.get("db_path", "../../data/new_graph_simulation.db")
    explanations_path = explanations_path or data_params.get("explanations_path")
    
    # 设置环境变量
    if "openai_api_key" in inference_params:
        os.environ["OPENAI_API_KEY"] = inference_params["openai_api_key"]
    if "openai_base_url" in inference_params:
        os.environ["OPENAI_BASE_URL"] = inference_params["openai_base_url"]
    
    # 创建仿真实例
    simulation = NewGraphLLMSimulation(config_path)
    
    # 初始化环境
    await simulation.initialize_environment(db_path, user_path)
    
    # 根据参数选择运行模式
    if explanations_path and os.path.exists(explanations_path):
        log.info(f"🔧 使用调整模式，解释文件: {explanations_path}")
        result = await simulation.run_adjustment_mode(
            user_path, explanations_path, agent_ids_to_adjust, output_path
        )
    else:
        log.info("🆕 使用生成模式")
        result = await simulation.run_generation_mode(
            user_path, output_path
        )
    
    log.info("🎉 仿真完成！")
    
    # 清理结果数据，减少内存占用
    if isinstance(result, dict) and "generated_data" in result:
        # 清理生成数据中的详细内容，只保留统计信息
        cleaned_result = {
            "mode": result.get("mode"),
            "total_agents": result.get("total_agents"),
            "successful_generations": result.get("successful_generations"),
            "failed_generations": result.get("failed_generations"),
            "generated_data_count": len(result.get("generated_data", [])),
            "status": "completed"
        }
        log.info(f"结果已清理，保留统计信息: {cleaned_result}")
        return cleaned_result
    
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.config_path or not os.path.exists(args.config_path):
        print("请通过--config_path指定有效的yaml配置文件！")
        exit(1)
    
    asyncio.run(
        main(config_path=args.config_path),
        debug=True,
    ) 
