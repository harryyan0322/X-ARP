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

from datetime import datetime
from typing import List, Optional, Callable, Dict, Any, Type
import logging
from camel.messages import BaseMessage

from .basic import Action
from .entity import SocialUser, SocialPost, SocialComment, TwitterProfile
from .utils import generate_id, str_to_datetime, check_time, format_time

logger = logging.getLogger(__name__)


class SocialAction:
    """
    社交动作基类
    
    参考 agent_action.py 的设计，提供统一的动作执行接口和提示词管理。
    所有提示词都直接嵌入到动作方法中，简化架构。
    """
    
    # 类级别的缓存，所有实例共享
    _cached_function_tools = None
    
    def __init__(self, agent_id: int, channel=None, agent=None):
        """
        初始化社交动作
        
        Args:
            agent_id: 智能体ID
            channel: 通信通道（可选）
            agent: 智能体实例（可选）
        """
        self.agent_id = agent_id
        self.channel = channel
        self.agent = agent
    
    def get_openai_function_list(self) -> list:
        """获取OpenAI函数列表（带缓存优化）"""
        # 使用类级别的缓存，避免重复创建 FunctionTool
        if SocialAction._cached_function_tools is None:
            SocialAction._cached_function_tools = [
                func for func in [
                    self.register_account,
                    self.create_post,
                    self.delete_post,
                    self.repost,
                    self.revoke_repost,
                    self.like_post,
                    self.unlike_post,
                    self.dislike_post,
                    self.undislike_post,
                    self.follow_user,
                    self.unfollow_user,
                    self.create_comment,
                    self.delete_comment,
                    self.like_comment,
                    self.unlike_comment,
                    self.dislike_comment,
                    self.undislike_comment,
                    self.search_posts,
                    self.search_users,
                    self.get_recommended_posts,
                    self.write_to_memory,
                    self.adjust_profile,
                    self.do_nothing,
                    self.smart_action_selector,
                    self.content_generation
                ]
            ]
        return SocialAction._cached_function_tools
    
    # =========== Prompt Manager 功能方法 ===========
    
    def create_user_description(self, profile: Dict[str, Any]) -> str:
        """
        创建用户描述
        
        Args:
            profile: 用户档案字典
            
        Returns:
            str: 格式化的用户描述
        """
        if not profile:
            return "A new user with no profile information."
        
        description_parts = []
        
        # 基本信息
        if profile.get('name'):
            description_parts.append(f"Name: {profile['name']}")
        if profile.get('screen_name'):
            description_parts.append(f"Screen name: @{profile['screen_name']}")
        if profile.get('description'):
            description_parts.append(f"Bio: {profile['description']}")
        if profile.get('location'):
            description_parts.append(f"Location: {profile['location']}")
        
        # 个人特征
        if profile.get('age'):
            description_parts.append(f"Age: {profile['age']}")
        if profile.get('gender'):
            description_parts.append(f"Gender: {profile['gender']}")
        if profile.get('mbit'):
            description_parts.append(f"MBTI: {profile['mbit']}")
        
        # 领域信息
        if profile.get('domains'):
            domains = profile['domains']
            if isinstance(domains, list):
                description_parts.append(f"Interests: {', '.join(domains)}")
            elif isinstance(domains, str):
                description_parts.append(f"Interests: {domains}")
        
        # 其他信息
        other_info = profile.get('other_info', {})
        if other_info:
            if other_info.get('tweet'):
                tweets = other_info['tweet']
                if isinstance(tweets, list) and tweets:
                    description_parts.append(f"Sample tweets: {'; '.join(tweets[:2])}")
                elif isinstance(tweets, str):
                    description_parts.append(f"Sample tweet: {tweets}")
        
        return " | ".join(description_parts) if description_parts else "A new user with basic profile."
    
    def format_existing_ids(self, existing_ids: List[str]) -> str:
        """
        格式化现有ID列表
        
        Args:
            existing_ids: 现有ID列表
            
        Returns:
            str: 格式化的ID字符串
        """
        if not existing_ids:
            return "No existing IDs"
        # 显示所有现有ID，确保唯一性
        return ", ".join([str(uid) for uid in existing_ids])
    
    def create_candidate_users_description(self, candidate_users: List[Dict[str, Any]]) -> str:
        """
        创建候选用户描述
        
        Args:
            candidate_users: 候选用户列表
            
        Returns:
            str: 格式化的候选用户描述
        """
        if not candidate_users:
            return "No candidate users available."
        
        descriptions = []
        for i, user in enumerate(candidate_users[:5]):  # 只显示前5个
            user_desc = self.create_user_description(user)
            user_id = user.get('profile.id', user.get('id', f'user_{i}'))
            descriptions.append(f"User {user_id}: {user_desc}")
        
        return "\n".join(descriptions)
    
    def create_example_tweets_text(self, tweet_examples: List[str]) -> str:
        """
        创建示例推文文本
        
        Args:
            tweet_examples: 示例推文列表
            
        Returns:
            str: 格式化的示例推文文本
        """
        if not tweet_examples:
            return "No example tweets available."
        
        if isinstance(tweet_examples, list):
            return "\n".join([f"- {tweet}" for tweet in tweet_examples[:3]])  # 只显示前3条
        elif isinstance(tweet_examples, str):
            return f"- {tweet_examples}"
        else:
            return "No example tweets available."
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """
        格式化提示词模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化的提示词
        """
        templates = {
            "generate_profile": self._get_generate_profile_prompt(),
            "generate_interests": self._get_generate_interests_prompt(),
            "decide_following": self._get_decide_following_prompt(),
            "generate_tweets": self._get_generate_tweets_prompt()
        }
        
        template = templates.get(template_name, "")
        if not template:
            logger.warning(f"Unknown prompt template: {template_name}")
            return ""
        
        try:
            class _SafeDict(dict):
                def __missing__(self, key):
                    return ""
            return template.format_map(_SafeDict(**kwargs))
        except KeyError as e:
            logger.error(f"Missing required parameter for template {template_name}: {e}")
            return template
    
    def _get_generate_profile_prompt(self) -> str:
        """获取生成档案的提示词模板"""
        return """You are a social media user profile generator. Generate a complete user profile following the str.md format.

User description: {user_description}
Existing IDs: {existing_ids}

IMPORTANT: Return ONLY valid JSON without any comments, explanations, or markdown formatting.

Please generate a complete user profile in JSON format with the following structure:
{{
    "id": "u123",
    "followers_count": 150,
    "friends_count": 200,
    "listed_count": 10,
    "favourites_count": 500,
    "statuses_count": 300,
    "age": 25,
    "name": "Full Name",
    "screen_name": "username",
    "location": "City, Country",
    "description": "Bio description about the user's interests and personality",
    "url": "https://example.com or empty string",
    "created_at": "2020-01-01T00:00:00Z",
    "lang": "en",
    "gender": "Male/Female/Other",
    "mbit": "INTJ/ENFP/ISTP/etc",
    "profile_image_url": "https://example.com/avatar.jpg or empty string",
    "pinned_tweet_id": null,
    "verified": true/false,
    "protected": true/false,
    "default_profile_image": true/false,
    "geo_enabled": true/false,
    "has_extended_profile": true/false
}}

CRITICAL REQUIREMENTS:
1. Fill in ALL fields with realistic, diverse values based on the user description
2. DO NOT use the example values above - create unique, personalized content
3. Generate realistic follower counts (50-2000), friend counts (100-500), etc.
4. Create unique names, screen names, and descriptions that match the user's personality
5. Use realistic locations, ages, and MBTI types
6. **CRITICAL: The user ID MUST be unique and different from ALL existing IDs listed above. ID format must be "u" followed by numbers only (e.g., "u123", "u456"). You MUST check the existing IDs list and ensure your generated ID is NOT in that list. If you see an ID like "u123" in the existing list, do NOT generate "u123" again.**
7. For boolean fields (verified, protected, default_profile_image, geo_enabled, has_extended_profile), make intelligent decisions based on the user's profile and social media behavior patterns
8. For URL fields (url, profile_image_url), decide whether the user would have a personal website or custom avatar based on their profile and interests
9. Do not leave any fields empty or null unless specifically required

Generate the profile (JSON only, no comments):"""
    
    def _get_generate_interests_prompt(self) -> str:
        """获取生成兴趣的提示词模板"""
        return """You are generating user interests for a social media profile.

User description: {user_description}
Available topics: {topic_choices}
Maximum topics: {max_topics}

IMPORTANT: Return ONLY valid JSON array without any comments, explanations, or markdown formatting.

Please select up to {max_topics} topics that best match the user's profile and interests.
Return the result as a JSON array of strings.

CRITICAL REQUIREMENTS:
1. Select topics that genuinely match the user's personality and interests
2. Choose diverse topics that reflect the user's background and preferences
3. Do not randomly select topics - make thoughtful choices based on the user description
4. Ensure the selection is realistic and coherent

Generate interests (JSON array only, no comments):"""
    
    def _get_decide_following_prompt(self) -> str:
        """获取决定关注的提示词模板"""
        return """You are deciding which users to follow on social media.

Your profile: {user_description}
Candidate users:
{candidate_users}

Graph-structured context (neighbors + posts):
{graph_prompt}

Available user IDs: {available_ids}

IMPORTANT: Return ONLY valid JSON array without any comments, explanations, or markdown formatting.

Please select user IDs to follow based on your interests and the candidate users' profiles.
Return the result as a JSON array of user IDs (strings).

CRITICAL REQUIREMENTS:
1. Select users whose interests and personality align with your profile
2. Choose users you would genuinely want to follow based on shared interests
3. Do not randomly select users - make thoughtful choices
4. Ensure the selection reflects realistic social media behavior
5. You can follow any number of users (0 to all available) based on your preferences
6. Only include user IDs that are in the available user IDs list

Choose users to follow (JSON array only, no comments):"""
    
    def _get_generate_tweets_prompt(self) -> str:
        """获取生成推文的提示词模板"""
        return """You are generating tweets for a social media user.

User profile: {user_description}
Example tweets:
{example_tweets}
Maximum tweets: {max_tweets}

Graph-structured context (neighbors + posts):
{graph_prompt}

IMPORTANT: Return ONLY valid JSON array without any comments, explanations, or markdown formatting.

Please generate up to {max_tweets} tweets (minimum 1, maximum 100) that reflect the user's personality, interests, and writing style.
Return the result as a JSON array of strings.

CRITICAL REQUIREMENTS:
1. Create unique, original tweets that match the user's personality and interests
2. Do not copy or closely mimic the example tweets - create fresh content
3. Ensure tweets reflect the user's writing style, tone, and interests
4. Make tweets engaging and authentic to the user's character
5. Include relevant hashtags and mentions when appropriate
6. Ensure tweets are realistic and appropriate for the user's profile
7. Tweets should be based on environmental observations and personal reflections
8. Write as if you're a real social media user sharing thoughts, experiences, and reactions
9. Use natural language, emojis, and social media conventions
10. No content restrictions - be authentic and genuine like a real user

Generate tweets (JSON array only, no comments):"""
    
    # =========== 原有动作方法 ===========
    
    async def perform_action(self, message: Any, type: str):
        """执行动作的统一接口"""
        if self.channel:
            message_id = await self.channel.write_to_receive_queue(
                (self.agent_id, message, type))
            response = await self.channel.read_from_send_queue(message_id)
            return response[2]
        else:
            # 如果没有channel，直接返回模拟结果
            return {"success": True, "message": f"Action {type} performed"}

    def _get_env_data(self):
        if self.agent and hasattr(self.agent, "env") and self.agent.env:
            return self.agent.env.get_env_data()
        return None

    async def register_account(self, user_dict: dict) -> dict:
        """注册账号"""
        env_data = self._get_env_data()
        if env_data and isinstance(user_dict, dict):
            try:
                user = SocialUser.from_dict(user_dict)
                env_data.add_user(user)
            except Exception as e:
                logger.warning(f"register_account failed: {e}")
        return await self.perform_action(user_dict, "REGISTER_ACCOUNT")
    
    # 以下是具体的动作方法，每个方法都包含嵌入的提示词
    async def create_post(self, content: str = "", use_llm_generation: bool = False, context: str = "") -> dict:
        """创建帖子
        
        Args:
            content: 帖子内容
            use_llm_generation: 是否使用LLM生成内容
            context: 上下文信息
            
        Returns:
            dict: 执行结果
        """
        if use_llm_generation and self.agent and hasattr(self.agent, 'astep'):
            # 使用LLM生成内容
            prompt = self._generate_post_prompt(context)
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.agent.astep(user_msg)
            
            if response.msgs:
                generated_content = response.msgs[-1].content
                if generated_content:
                    content = generated_content
                    logger.info(f"Generated post content: {content[:100]}...")
        
        # 尝试写入环境数据
        env_data = self._get_env_data()
        if env_data and self.agent and hasattr(self.agent, "user_info") and self.agent.user_info:
            try:
                author_id = self.agent.user_info.profile.id
                author_name = self.agent.user_info.profile.name or self.agent.user_info.profile.screen_name
                post_id = generate_id(list(env_data.posts.keys()))
                post = SocialPost(
                    post_id=post_id,
                    author_id=str(author_id),
                    author_name=str(author_name),
                    content=content,
                    created_time=datetime.utcnow(),
                    platform="twitter",
                )
                env_data.add_post(post)
            except Exception as e:
                logger.warning(f"create_post env update failed: {e}")

        return await self.perform_action(content, "CREATE_POST")

    async def delete_post(self, post_id: str) -> dict:
        """删除帖子"""
        env_data = self._get_env_data()
        if env_data:
            try:
                env_data.delete_post(post_id)
            except Exception as e:
                logger.warning(f"delete_post failed: {e}")
        return await self.perform_action(post_id, "DELETE_POST")

    async def repost(self, post_id: str) -> dict:
        """转发帖子"""
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post:
                post.repost()
        return await self.perform_action(post_id, "REPOST")

    async def revoke_repost(self, post_id: str) -> dict:
        """撤销转发"""
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post and post.num_reposts > 0:
                post.num_reposts -= 1
        return await self.perform_action(post_id, "REPOST_REVOKE")
    
    def _generate_post_prompt(self, context: str = "") -> str:
        """生成发帖提示词"""
        return f"""You are creating a post for a social media platform.
Based on your personality and interests, generate an engaging and authentic post.

Context: {context}

Please generate a natural, engaging post that reflects your character.
Generate a single post:"""
    
    async def like_post(self, post_id: str) -> dict:
        """点赞帖子
        
        Args:
            post_id: 帖子ID
            
        Returns:
            dict: 执行结果
        """
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post:
                post.like()
        return await self.perform_action(post_id, "LIKE_POST")
    
    async def unlike_post(self, post_id: str) -> dict:
        """取消点赞
        
        Args:
            post_id: 帖子ID
            
        Returns:
            dict: 执行结果
        """
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post:
                post.unlike()
        return await self.perform_action(post_id, "UNLIKE_POST")

    async def dislike_post(self, post_id: str) -> dict:
        """点踩帖子"""
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post:
                post.dislike()
        return await self.perform_action(post_id, "DISLIKE_POST")

    async def undislike_post(self, post_id: str) -> dict:
        """取消点踩"""
        env_data = self._get_env_data()
        if env_data:
            post = env_data.get_post(post_id)
            if post:
                post.undislike()
        return await self.perform_action(post_id, "DISLIKE_POST_REVOKE")
    
    async def follow_user(self, followee_id: str) -> dict:
        """关注用户
        
        Args:
            followee_id: 要关注的用户ID
            
        Returns:
            dict: 执行结果
        """
        return await self.perform_action(followee_id, "FOLLOW_USER")
    
    async def unfollow_user(self, followee_id: str) -> dict:
        """取消关注
        
        Args:
            followee_id: 要取消关注的用户ID
            
        Returns:
            dict: 执行结果
        """
        return await self.perform_action(followee_id, "UNFOLLOW_USER")

    async def create_comment(self, post_id: str, content: str, parent_id: str = "") -> dict:
        """创建评论"""
        env_data = self._get_env_data()
        if env_data and self.agent and self.agent.user_info:
            try:
                comment_id = generate_id(list(env_data.comments.keys()))
                author_id = self.agent.user_info.profile.id
                author_name = self.agent.user_info.profile.name or self.agent.user_info.profile.screen_name
                comment = SocialComment(
                    comment_id=comment_id,
                    post_id=post_id,
                    author_id=str(author_id),
                    author_name=str(author_name),
                    content=content,
                    created_time=datetime.utcnow(),
                    parent_id=parent_id or None,
                    platform="twitter",
                )
                env_data.add_comment(comment)
            except Exception as e:
                logger.warning(f"create_comment failed: {e}")
        return await self.perform_action({"post_id": post_id, "content": content}, "CREATE_COMMENT")

    async def delete_comment(self, comment_id: str) -> dict:
        """删除评论"""
        env_data = self._get_env_data()
        if env_data:
            env_data.delete_comment(comment_id)
        return await self.perform_action(comment_id, "DELETE_COMMENT")

    async def like_comment(self, comment_id: str) -> dict:
        env_data = self._get_env_data()
        if env_data:
            env_data.like_comment(comment_id)
        return await self.perform_action(comment_id, "LIKE_COMMENT")

    async def unlike_comment(self, comment_id: str) -> dict:
        env_data = self._get_env_data()
        if env_data:
            env_data.unlike_comment(comment_id)
        return await self.perform_action(comment_id, "UNLIKE_COMMENT")

    async def dislike_comment(self, comment_id: str) -> dict:
        env_data = self._get_env_data()
        if env_data:
            env_data.dislike_comment(comment_id)
        return await self.perform_action(comment_id, "DISLIKE_COMMENT")

    async def undislike_comment(self, comment_id: str) -> dict:
        env_data = self._get_env_data()
        if env_data:
            env_data.undislike_comment(comment_id)
        return await self.perform_action(comment_id, "DISLIKE_COMMENT_REVOKE")
    
    async def search_posts(self, query: str = "", use_llm_generation: bool = False) -> dict:
        """搜索帖子
        
        Args:
            query: 搜索查询
            use_llm_generation: 是否使用LLM生成查询
            
        Returns:
            dict: 执行结果
        """
        if use_llm_generation and self.agent and hasattr(self.agent, 'astep'):
            # 使用LLM生成搜索查询
            prompt = self._generate_search_prompt("posts")
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.agent.astep(user_msg)
            
            if response.msgs:
                generated_query = response.msgs[-1].content
                if generated_query:
                    query = generated_query
                    logger.info(f"Generated search query: {query}")
        
        return await self.perform_action(query, "SEARCH_POSTS")
    
    async def search_users(self, query: str = "", use_llm_generation: bool = False) -> dict:
        """搜索用户
        
        Args:
            query: 搜索查询
            use_llm_generation: 是否使用LLM生成查询
            
        Returns:
            dict: 执行结果
        """
        if use_llm_generation and self.agent and hasattr(self.agent, 'astep'):
            # 使用LLM生成搜索查询
            prompt = self._generate_search_prompt("users")
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.agent.astep(user_msg)
            
            if response.msgs:
                generated_query = response.msgs[-1].content
                if generated_query:
                    query = generated_query
                    logger.info(f"Generated user search query: {query}")
        
        return await self.perform_action(query, "SEARCH_USERS")
    
    def _generate_search_prompt(self, search_type: str) -> str:
        """生成搜索提示词"""
        return f"""You are searching for content on a social media platform.
Based on your interests and the current environment, what would you like to search for?

Search type: {search_type}

Please generate a search query that reflects your interests and the current context.
Generate a single search query:"""
    
    async def get_recommended_posts(self, limit: int = 10) -> dict:
        """获取推荐帖子
        
        Args:
            limit: 获取的帖子数量限制
            
        Returns:
            dict: 执行结果
        """
        return await self.perform_action(limit, "GET_RECOMMENDED_POSTS")
    
    async def write_to_memory(self, content: str, memory_type: str = "general") -> dict:
        """写入记忆
        
        Args:
            content: 要写入的内容
            memory_type: 记忆类型
            
        Returns:
            dict: 执行结果
        """
        memory_data = {"content": content, "type": memory_type}
        return await self.perform_action(memory_data, "WRITE_TO_MEMORY")
    
    async def adjust_profile(self, explanations_path: str = "", ori_data_template: dict = None) -> dict:
        """调整档案
        
        Args:
            explanations_path: 解释文件路径
            ori_data_template: 原始数据模板
            
        Returns:
            dict: 执行结果
        """
        profile_data = {
            "explanations_path": explanations_path,
            "ori_data_template": ori_data_template or {}
        }
        return await self.perform_action(profile_data, "ADJUST_PROFILE")
    
    async def do_nothing(self) -> dict:
        """什么都不做
        
        Returns:
            dict: 执行结果
        """
        return await self.perform_action(None, "DO_NOTHING")
    
    async def smart_action_selector(self, context: str = "", available_actions: List[str] = None, environment_data: dict = None) -> dict:
        """智能动作选择器
        
        Args:
            context: 上下文信息
            available_actions: 可用的动作列表
            environment_data: 环境数据
            
        Returns:
            dict: 执行结果
        """
        if self.agent and hasattr(self.agent, 'astep'):
            prompt = self._generate_action_selection_prompt(context, available_actions or [], environment_data or {})
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.agent.astep(user_msg)
            
            if response.msgs:
                selected_action = response.msgs[-1].content
                logger.info(f"Smart action selector chose: {selected_action}")
                return {"success": True, "selected_action": selected_action}
        
        return await self.perform_action({"context": context, "actions": available_actions}, "SMART_ACTION_SELECTOR")
    
    def _generate_action_selection_prompt(self, context: str, available_actions: List[str], environment_data: dict) -> str:
        """生成动作选择提示词"""
        actions_desc = "\n".join([f"- {action}" for action in available_actions])
        env_desc = f"Users: {environment_data.get('user_count', 0)}, Posts: {environment_data.get('post_count', 0)}"
        
        return f"""You are a social media user in a simulation environment.
Based on your profile and the current environment, choose the most appropriate action to perform.

Current environment: {env_desc}
Available actions: {actions_desc}
Context: {context}

Please select one action from the available actions.
Choose an action:"""
    
    async def content_generation(self, content_type: str, context: str = "", target_id: str = "") -> dict:
        """内容生成
        
        Args:
            content_type: 内容类型
            context: 上下文信息
            target_id: 目标ID
            
        Returns:
            dict: 执行结果
        """
        if self.agent and hasattr(self.agent, 'astep'):
            prompt = self._generate_content_prompt(content_type, context, target_id)
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.agent.astep(user_msg)
            
            if response.msgs:
                generated_content = response.msgs[-1].content
                if generated_content:
                    logger.info(f"Generated {content_type} content: {generated_content[:100]}...")
                    return {"success": True, "content": generated_content}
        
        return await self.perform_action({"type": content_type, "context": context, "target": target_id}, "CONTENT_GENERATION")
    
    def _generate_content_prompt(self, content_type: str, context: str = "", target_id: str = "") -> str:
        """生成内容提示词"""
        full_context = context
        if target_id:
            full_context += f" Target: {target_id}"
        
        return f"""You are creating content for a social media platform.
Based on your profile and the current context, generate appropriate content.

Content type: {content_type}
Context: {full_context}

Please generate content that reflects your personality, interests, and the current situation.
Make it natural, engaging, and authentic to your character.

Generate {content_type} content:"""


class CreateUserAction(Action):
    """创建用户动作（符合 str.md 格式）"""
    
    def __init__(self):
        super().__init__(
            name="CreateUser",
            description="Create a new user account in the social network with str.md format",
            func=self.create_user,
            input_args_schema={
                "id": str,
                "screen_name": str,
                "name": str,
                "description": str,
                "location": str,
                "age": int,
                "gender": str,
                "mbit": str,
                "domains": list,
                "label": int,
                "tweets": list,
                "neighbor": dict
            },
        )

    @staticmethod
    def create_user(action_args: Dict[str, Any], env, agent) -> str:
        """创建用户（符合 str.md 格式）"""
        try:
            current_time = datetime.now()
            if not check_time(current_time, env.config):
                return "Action time is outside the allowed time range"
            
            # 创建 TwitterProfile
            profile = TwitterProfile(
                id=action_args["id"],
                followers_count=0,
                friends_count=0,
                listed_count=0,
                favourites_count=0,
                statuses_count=len(action_args.get("tweets", [])),
                age=action_args.get("age"),
                name=action_args["name"],
                screen_name=action_args["screen_name"],
                location=action_args.get("location", ""),
                description=action_args["description"],
                url="",
                created_at=current_time.strftime("%Y-%m-%d"),
                lang="en",
                gender=action_args.get("gender", ""),
                mbit=action_args.get("mbit", ""),
                profile_image_url="",
                pinned_tweet_id=None,
                verified=False,
                protected=False,
                default_profile_image=False,
                geo_enabled=False,
                has_extended_profile=False
            )
            
            # 创建 SocialUser
            user = SocialUser(
                id=action_args["id"],
                profile=profile,
                tweets=action_args.get("tweets", []),
                neighbor=action_args.get("neighbor", {"following": [], "follower": []}),
                domains=action_args.get("domains", []),
                label=action_args.get("label", 0)
            )
            
            env.get_env_data().add_user(user)
            return f"User created successfully! User ID: {user.id}"
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return f"Failed to create user: {str(e)}" 
