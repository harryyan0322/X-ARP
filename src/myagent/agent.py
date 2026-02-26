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

import inspect
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union, Tuple

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

from .actions import SocialAction
from .environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.typing import ActionType

from .entity import SocialUser  # 使用本地的实体类

import random
import json
from camel.memories.records import MemoryRecord
# generate_profile_and_dict_by_llm 功能已集成到 SocialAgent 类中
# prompt_manager 功能已迁移到 SocialAction 类中
# ActionGenerator 功能已集成到 SocialAction 类中

if TYPE_CHECKING:
    from oasis.graphrag import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            f"./log/social.agent-{str(now)}.log")
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)

# 定义所有可用的社交行为类型
ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]

# 全局工具缓存，避免重复创建 FunctionTool
_GLOBAL_FUNCTION_TOOLS_CACHE = None

def get_cached_function_tools():
    """获取全局缓存的函数工具列表"""
    global _GLOBAL_FUNCTION_TOOLS_CACHE
    if _GLOBAL_FUNCTION_TOOLS_CACHE is None:
        # 延迟导入，避免循环导入问题
        from .actions import SocialAction
        # 创建一个临时的 SocialAction 实例来获取工具列表
        temp_action = SocialAction(0)
        _GLOBAL_FUNCTION_TOOLS_CACHE = temp_action.get_openai_function_list()
    return _GLOBAL_FUNCTION_TOOLS_CACHE


class SocialAgent(ChatAgent):
    """
    社交网络智能体类，继承自ChatAgent
    
    该类实现了社交网络中的智能体功能，包括：
    1. 基于大语言模型的社交行为决策
    2. 社交网络环境感知和交互
    3. 用户画像生成和管理
    4. 多种行为执行模式（LLM驱动、人机交互、数据驱动）
    5. 社交网络图结构维护
    6. 访谈和测试功能
    """
    
    # 类级别的工具缓存，所有智能体实例共享
    _cached_action_tools = None

    def __init__(self,
                 agent_id: int,                    # 智能体唯一标识符
                 user_info: SocialUser,              # 用户信息对象
                 user_info_template: TextPrompt | None = None,  # 自定义用户信息模板
                 channel: Channel | None = None,   # 通信通道
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,  # 大语言模型
                 agent_graph: "AgentGraph" = None, # 智能体图结构
                 available_actions: list[ActionType] = None,    # 可用的行为类型
                 tools: Optional[List[Union[FunctionTool, Callable]]] = None,  # 工具函数列表
                 single_iteration: bool = True,    # 是否单次迭代
                 interview_record: bool = False):  # 是否记录访谈
        """
        初始化社交智能体
        
        Args:
            agent_id: 智能体ID
            user_info: 用户信息对象，包含用户画像和属性
            user_info_template: 自定义用户信息模板，用于生成系统消息
            channel: 通信通道，用于智能体间通信
            model: 大语言模型，支持单个模型、模型列表或模型管理器
            agent_graph: 智能体图结构，维护智能体间的关系网络
            available_actions: 可用的行为类型列表，为空时使用所有行为
            tools: 额外的工具函数列表
            single_iteration: 是否只执行单次迭代
            interview_record: 是否记录访谈过程到内存
        """
        # 设置智能体基本属性
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        
        # 创建社交环境
        self.env = SocialEnvironment({})
        
        # 创建社交行为执行器
        self.env.action = SocialAction(agent_id, self.channel, self)
        
        # 生成系统消息内容
        if user_info_template is None:
            system_message_content = self._create_system_message()
        else:
            system_message_content = self._create_custom_system_message(user_info_template)
        
        # 创建系统消息
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_message_content,  # 系统提示词
        )

        # 配置可用的行为工具（使用全局缓存优化）
        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            # 使用全局缓存，避免重复创建 FunctionTool
            self.action_tools = get_cached_function_tools()
        else:
            # 使用全局缓存
            all_tools = get_cached_function_tools()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            # 验证指定的行为是否支持
            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            
            # 过滤出指定的行为工具
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        
        # 合并所有工具
        all_tools = (tools or []) + (self.action_tools or [])
        
        # 调用父类初始化
        super().__init__(system_message=system_message,
                         model=model,
                         scheduling_strategy='random_model',
                         tools=all_tools,
                         single_iteration=single_iteration)
        
        # 确保model属性被正确设置
        self.model = model
        
        # 设置其他属性
        self.interview_record = interview_record
        self.agent_graph = agent_graph
        
    def _create_system_message(self) -> str:
        """创建系统消息内容"""
        if self.user_info and hasattr(self.user_info, 'profile'):
            profile = self.user_info.profile
            name = getattr(profile, 'name', 'Unknown User')
            description = getattr(profile, 'description', 'No description available')
            return f"You are {name}, a social media user. {description}"
        else:
            return "You are a social media user. You can interact with other users through various social actions."
    
    def _create_custom_system_message(self, user_info_template: TextPrompt) -> str:
        """使用自定义模板创建系统消息内容"""
        try:
            return user_info_template.format(user_info=self.user_info)
        except Exception as e:
            agent_log.warning(f"Failed to create custom system message: {e}")
            return self._create_system_message()
        
        # 动作生成功能已集成到 SocialAction 类中
        


    async def perform_action_by_llm(self, context: str = "", environment_data: dict = None, available_actions: list = None):
        """
        基于大语言模型执行社交行为
        
        该方法让智能体观察社交环境，然后使用LLM决定执行什么社交行为。
        智能体会分析环境中的帖子、用户等信息，然后选择合适的行为（如点赞、评论、关注等）。
        
        Returns:
            执行结果，包含工具调用信息
        """
        try:
            # 获取环境信息
            env_prompt = await self.env.to_text_prompt()
            
            # 构建环境数据
            if environment_data is None:
                environment_data = {
                    'environment_description': env_prompt,
                    'user_count': len(self.env.get_env_data().users) if hasattr(self.env, 'get_env_data') else 0,
                    'post_count': len(self.env.get_env_data().posts) if hasattr(self.env, 'get_env_data') else 0,
                    'recent_posts': [],
                    'recent_users': [],
                    'trending_topics': []
                }
            else:
                environment_data = dict(environment_data)
                environment_data.setdefault('environment_description', env_prompt)
                environment_data.setdefault('user_count', len(self.env.get_env_data().users) if hasattr(self.env, 'get_env_data') else 0)
                environment_data.setdefault('post_count', len(self.env.get_env_data().posts) if hasattr(self.env, 'get_env_data') else 0)
            
            # 使用 SocialAction 生成动作提示词
            actions = available_actions or ALL_SOCIAL_ACTIONS
            prompt = self.env.action._generate_action_selection_prompt(context, actions, environment_data)
            
            # 创建用户消息，要求智能体执行社交行为
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt)
            
            agent_log.info(
                f"Agent {self.social_agent_id} observing environment: "
                f"{env_prompt}")
            
            # 执行一步推理
            response = await self.astep(user_msg)
            
            # 处理工具调用结果
            if response.info and 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    action_name = tool_call.tool_name
                    args = tool_call.args
                    agent_log.info(f"Agent {self.social_agent_id} performed "
                                   f"action: {action_name} with args: {args}")
                    
                    if action_name not in ALL_SOCIAL_ACTIONS:
                        agent_log.info(
                            f"Agent {self.social_agent_id} get the result: "
                            f"{tool_call.result}")
                    
                    # 注释掉图结构更新，避免大规模智能体时的性能问题
                    # await self.perform_agent_graph_action(action_name, args)

            return response
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e


    async def generate_new_user_profile_by_llm(self):
        """
        Generate only the new user's profile field, strictly following str.md, without friends field.
        The prompt is in English and uses as many fields from user_info as possible, referring to str.md for details.
        The personal description is a natural, fluent paragraph, not a field list. No backslash (\\) escapes in the output.
        Only profile.id is required to be unique among all agents.
        """
        profile = self.user_info.profile.to_dict() if self.user_info and self.user_info.profile else {}
        other_info = profile.get('other_info', {})
        
        # 使用 SocialAction 创建用户描述
        user_description = self.env.action.create_user_description(profile)
        
        # 获取现有ID列表（完整版本）
        existing_ids = set()
        if hasattr(self, 'agent_graph') and self.agent_graph and hasattr(self.agent_graph, 'get_agents'):
            try:
                all_agents = list(self.agent_graph.get_agents())
                for aid, agent in all_agents:
                    if hasattr(agent, 'user_info') and agent.user_info and hasattr(agent.user_info, 'profile'):
                        pid = agent.user_info.profile.id if hasattr(agent.user_info.profile, 'id') else None
                        if pid:
                            existing_ids.add(str(pid))
            except Exception as e:
                agent_log.warning(f"获取现有ID失败: {e}")
        
        # 使用 SocialAction 格式化提示词
        prompt = self.env.action.format_prompt(
            "generate_profile",
            user_description=user_description,
            existing_ids=self.env.action.format_existing_ids(list(existing_ids))
        )
        
        user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
        try:
            agent_log.debug(f"调用LLM生成档案，提示词长度: {len(prompt)}")
            
            # 检查模型是否可用
            if not self.model:
                agent_log.error("模型未初始化")
                return ""
            
            response = await self.astep(user_msg, response_format=None)
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                agent_log.debug(f"LLM返回内容长度: {len(content)}")
            else:
                content = ""
                agent_log.warning("LLM返回空消息")
        except Exception as e:
            agent_log.error(f"LLM call failed: {e}")
            content = ""
        
        # 检查内容是否为空
        if not content or content.strip() == "":
            agent_log.error("LLM返回内容为空，无法生成档案")
            return ""
        
        return content


    async def generate_user_interests_by_llm(self, topic_choices=None, max_topics=3):
        """
        让LLM为当前用户生成感兴趣的话题（topics/interests），从给定的topic_choices中选择，输出为JSON数组。
        """
        if topic_choices is None:
            topic_choices = [
                'Politics', 'Urban Legends', 'Business', 'Terrorism & War',
                'Science & Technology', 'Entertainment', 'Natural Disasters', 'Health',
                'Education'
            ]
        profile = self.user_info.profile.to_dict() if self.user_info and self.user_info.profile else {}
        
        # 使用 SocialAction 创建用户描述
        user_description = self.env.action.create_user_description(profile)
        
        # 使用 SocialAction 格式化提示词
        prompt = self.env.action.format_prompt(
            "generate_interests",
            topic_choices=topic_choices,
            user_description=user_description,
            max_topics=max_topics
        )
        
        user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
        try:
            response = await self.astep(user_msg, response_format=None)
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
            else:
                content = ""
        except Exception as e:
            agent_log.error(f"LLM call failed: {e}")
            content = ""
        # 检查内容是否为空
        if not content or content.strip() == "":
            agent_log.warning("LLM返回内容为空，使用默认空兴趣列表")
            return []
        
        # 清理转义字符
        cleaned_content = content.replace('\\', '')
        agent_log.debug(f"清理后的兴趣内容: {cleaned_content[:200]}...")
        
        try:
            # 提取JSON内容（处理markdown代码块）
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                agent_log.debug(f"提取的兴趣JSON内容: {json_content}")
            else:
                # 尝试直接解析
                json_content = cleaned_content
            
            # 如果内容看起来像自然语言描述而不是JSON，使用默认空列表
            if not json_content.strip().startswith('['):
                agent_log.warning(f"LLM返回的不是JSON格式，使用默认空兴趣列表: {json_content[:100]}...")
                return []
            
            topics = json.loads(json_content)
            if not isinstance(topics, list):
                raise ValueError("LLM output is not a list")
            # 校验每个元素都在topic_choices中且为str
            topics = [t for t in topics if isinstance(t, str) and t in topic_choices]
            return topics
        except Exception as e:
            agent_log.error(f"Failed to parse topics list: {e}, 原始内容: {content[:100]}...")
            return []


    async def decide_user_following_by_llm(self, ori_data: dict, candidate_users: list, min_friends: int = 5, graph_prompt: str = "") -> dict:
        """
        生成关注关系：基于用户信息和候选用户列表决定关注哪些用户
        """
        # 使用 SocialAction 创建用户描述
        user_description = self.env.action.create_user_description(ori_data)
        
        # 使用 SocialAction 创建候选用户描述
        candidate_users_text = self.env.action.create_candidate_users_description(candidate_users)
        
        # 获取可用ID列表
        show_ids = [user.get('profile.id', user.get('id', '')) for user in candidate_users]
        available_ids = [str(sid) for sid in show_ids]
        
        # 使用 SocialAction 格式化提示词
        prompt = self.env.action.format_prompt(
            "decide_following",
            user_description=user_description,
            candidate_users=candidate_users_text,
            min_friends=min_friends,
            available_ids=available_ids,
            graph_prompt=graph_prompt
        )
        
        # 让智能体通过动作执行来生成关注关系
        user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
        try:
            # 使用智能体的动作执行能力，禁用工具调用以避免错误
            response = await self.astep(user_msg, response_format=None)
            
            # 解析LLM返回的内容
            following_list = []
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                # 解析LLM返回的JSON内容
                following_list = self._parse_following_list_from_content(content, available_ids)
                if following_list:
                    agent_log.debug(f"智能体通过LLM返回内容解析关注关系")
            
            # 更新用户数据
            if 'neighbor' not in ori_data or not isinstance(ori_data['neighbor'], dict):
                ori_data['neighbor'] = {"following": [], "follower": []}
            ori_data['neighbor']['following'] = following_list
            
            # 真正执行关注动作到环境中
            for followee_id in following_list:
                try:
                    # 更新当前用户的following列表
                    if 'neighbor' not in ori_data:
                        ori_data['neighbor'] = {"following": [], "follower": []}
                    if followee_id not in ori_data['neighbor']['following']:
                        ori_data['neighbor']['following'].append(followee_id)
                    
                    # 更新被关注用户的follower列表
                    current_user_id = ori_data.get('id', f'u{self.social_agent_id}')
                    
                    # 查找被关注的用户并更新其follower列表
                    if hasattr(self, 'agent_graph') and self.agent_graph:
                        try:
                            all_agents = list(self.agent_graph.get_agents())
                            for aid, other_agent in all_agents:
                                if hasattr(other_agent, 'user_info') and other_agent.user_info:
                                    other_user_dict = other_agent.user_info.to_dict()
                                    if other_user_dict.get('id') == followee_id:
                                        if 'neighbor' not in other_user_dict:
                                            other_user_dict['neighbor'] = {"following": [], "follower": []}
                                        if current_user_id not in other_user_dict['neighbor']['follower']:
                                            other_user_dict['neighbor']['follower'].append(current_user_id)
                                        break
                        except Exception as e:
                            agent_log.warning(f"更新被关注用户follower列表失败: {e}")
                    
                    agent_log.debug(f"智能体 {self.social_agent_id} 成功关注用户 {followee_id}")
                except Exception as e:
                    agent_log.warning(f"智能体 {self.social_agent_id} 关注用户 {followee_id} 失败: {e}")
            
            agent_log.info(f"生成关注关系: {len(following_list)}个")
            return ori_data
            
        except Exception as e:
            agent_log.error(f"生成关注关系失败: {e}")
            # 回退到原始方法
            return await self._fallback_decide_following(ori_data, candidate_users, min_friends, graph_prompt=graph_prompt)
    
    def _parse_following_list_from_content(self, content: str, available_ids: list) -> list:
        """从LLM返回的内容中解析关注列表"""
        try:
            if not content or content.strip() == "":
                return []
            
            # 清理转义字符
            cleaned_content = content.replace('\\', '')
            
            # 提取JSON内容（处理markdown代码块）
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = cleaned_content
            
            # 如果内容看起来像自然语言描述而不是JSON，返回空列表
            if not json_content.strip().startswith('['):
                return []
            
            following_list = json.loads(json_content)
            if not isinstance(following_list, list):
                return []
            
            # 过滤出有效的用户ID
            valid_following = [uid for uid in following_list if uid in available_ids]
            return valid_following
            
        except Exception as e:
            agent_log.error(f"解析关注列表失败: {e}")
            return []
    
    async def _fallback_decide_following(self, ori_data: dict, candidate_users: list, min_friends: int = 5, graph_prompt: str = "") -> dict:
        """回退方法：使用原始的数据生成方式"""
        try:
            agent_log.debug(f"使用回退方法生成关注关系")
            
            # 使用 SocialAction 创建用户描述
            user_description = self.env.action.create_user_description(ori_data)
            
            # 使用 SocialAction 创建候选用户描述
            candidate_users_text = self.env.action.create_candidate_users_description(candidate_users)
            
            # 获取可用ID列表
            show_ids = [user.get('profile.id', user.get('id', '')) for user in candidate_users]
            available_ids = [str(sid) for sid in show_ids]
            
            # 使用 SocialAction 格式化提示词（使用原始JSON格式）
            prompt = f"""You are deciding which users to follow on social media.

Your profile: {user_description}
Candidate users:
{candidate_users_text}

Available user IDs: {available_ids}
Minimum friends to follow: {min_friends}

IMPORTANT: Return ONLY valid JSON array without any comments, explanations, or markdown formatting.

Please select user IDs to follow based on your interests and the candidate users' profiles.
Return the result as a JSON array of user IDs (strings).

CRITICAL REQUIREMENTS:
1. Select users whose interests and personality align with your profile
2. Choose users you would genuinely want to follow based on shared interests
3. Do not randomly select users - make thoughtful choices
4. Ensure the selection reflects realistic social media behavior
5. Follow at least the minimum number of friends specified

Choose users to follow (JSON array only, no comments):"""
            
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.astep(user_msg, response_format=None)
            
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                following_list = self._parse_following_list_from_content(content, available_ids)
            else:
                following_list = []
            
            # 更新用户数据
            if 'neighbor' not in ori_data or not isinstance(ori_data['neighbor'], dict):
                ori_data['neighbor'] = {"following": [], "follower": []}
            ori_data['neighbor']['following'] = following_list
            
            # 真正执行关注动作到环境中（与主要方案保持一致）
            for followee_id in following_list:
                try:
                    # 更新被关注用户的follower列表
                    current_user_id = ori_data.get('id', f'u{self.social_agent_id}')
                    
                    # 查找被关注的用户并更新其follower列表
                    if hasattr(self, 'agent_graph') and self.agent_graph:
                        try:
                            all_agents = list(self.agent_graph.get_agents())
                            for aid, other_agent in all_agents:
                                if hasattr(other_agent, 'user_info') and other_agent.user_info:
                                    other_user_dict = other_agent.user_info.to_dict()
                                    if other_user_dict.get('id') == followee_id:
                                        if 'neighbor' not in other_user_dict:
                                            other_user_dict['neighbor'] = {"following": [], "follower": []}
                                        if current_user_id not in other_user_dict['neighbor']['follower']:
                                            other_user_dict['neighbor']['follower'].append(current_user_id)
                                        break
                        except Exception as e:
                            agent_log.warning(f"回退方法更新被关注用户follower列表失败: {e}")
                    
                    agent_log.debug(f"回退方法智能体 {self.social_agent_id} 成功关注用户 {followee_id}")
                except Exception as e:
                    agent_log.warning(f"回退方法智能体 {self.social_agent_id} 关注用户 {followee_id} 失败: {e}")
            
            agent_log.info(f"回退方法生成关注关系: {len(following_list)}个")
            return ori_data
            
        except Exception as e:
            agent_log.error(f"回退方法生成关注关系失败: {e}")
            # 返回空的关注列表
            if 'neighbor' not in ori_data or not isinstance(ori_data['neighbor'], dict):
                ori_data['neighbor'] = {"following": [], "follower": []}
            ori_data['neighbor']['following'] = []
            return ori_data


    async def generate_user_tweets_by_llm(self, ori_data: dict, max_tweets: int = 100, graph_prompt: str = "") -> str:
        """
        生成推文：基于用户信息生成符合用户特征的推文
        """
        # 使用 SocialAction 创建用户描述
        user_description = self.env.action.create_user_description(ori_data)
        
        # 获取当前agent的推文示例
        tweet_examples = []
        if hasattr(self, 'user_info') and self.user_info and hasattr(self.user_info, 'tweets'):
            tweet_examples = self.user_info.tweets[:3]  # 取前3条推文作为示例
        
        # 使用 SocialAction 创建示例推文文本
        example_tweets = self.env.action.create_example_tweets_text(tweet_examples)
        
        # 使用 SocialAction 格式化提示词
        prompt = self.env.action.format_prompt(
            "generate_tweets",
            user_description=user_description,
            max_tweets=max_tweets,
            example_tweets=example_tweets,
            graph_prompt=graph_prompt
        )
        
        # 让智能体通过动作执行来生成推文
        user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
        try:
            # 使用智能体的动作执行能力，禁用工具调用以避免错误
            response = await self.astep(user_msg, response_format=None)
            
            # 解析LLM返回的内容
            published_tweets = []
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                # 解析LLM返回的JSON内容
                published_tweets = self._parse_tweets_from_content(content)
                if published_tweets:
                    agent_log.debug(f"智能体通过LLM返回内容解析推文: {len(published_tweets)}条")
            
            # 确保推文数量在合理范围内
            if len(published_tweets) == 0:
                agent_log.warning(f"生成的推文数量为0，期望至少1条")
            elif len(published_tweets) > max_tweets:
                published_tweets = published_tweets[:max_tweets]
                agent_log.info(f"推文数量超过上限，截取前{max_tweets}条")
            
            # 真正执行发布推文动作到环境中
            for tweet_content in published_tweets:
                try:
                    # 更新用户的推文列表
                    if 'tweets' not in ori_data:
                        ori_data['tweets'] = []
                    if tweet_content not in ori_data['tweets']:
                        ori_data['tweets'].append(tweet_content)
                    
                    # 更新用户的statuses_count
                    if 'profile' in ori_data and isinstance(ori_data['profile'], dict):
                        ori_data['profile']['statuses_count'] = len(ori_data['tweets'])
                    
                    agent_log.debug(f"智能体 {self.social_agent_id} 成功发布推文: {tweet_content[:50]}...")
                except Exception as e:
                    agent_log.warning(f"智能体 {self.social_agent_id} 发布推文失败: {e}")
            
            agent_log.info(f"生成推文: {len(published_tweets)}条")
            return json.dumps(published_tweets, ensure_ascii=False)
            
        except Exception as e:
            agent_log.error(f"生成推文失败: {e}")
            # 回退到原始方法
            return await self._fallback_generate_tweets(ori_data, max_tweets, graph_prompt=graph_prompt)
    
    def _parse_tweets_from_content(self, content: str) -> list:
        """从LLM返回的内容中解析推文列表"""
        try:
            if not content or content.strip() == "":
                return []
            
            # 清理转义字符
            cleaned_content = content.replace('\\', '')
            
            # 提取JSON内容（处理markdown代码块）
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = cleaned_content
            
            # 如果内容看起来像自然语言描述而不是JSON，返回空列表
            if not json_content.strip().startswith('['):
                return []
            
            tweets = json.loads(json_content)
            if not isinstance(tweets, list):
                return []
            
            # 确保每条推文为字符串
            valid_tweets = [str(t) for t in tweets if isinstance(t, str)]
            return valid_tweets
            
        except Exception as e:
            agent_log.error(f"解析推文列表失败: {e}")
            return []
    
    async def _fallback_generate_tweets(self, ori_data: dict, max_tweets: int = 100, graph_prompt: str = "") -> str:
        """回退方法：使用原始的数据生成方式"""
        try:
            agent_log.debug(f"使用回退方法生成推文")
            
            # 使用 SocialAction 创建用户描述
            user_description = self.env.action.create_user_description(ori_data)
            
            # 获取当前agent的推文示例
            tweet_examples = []
            if hasattr(self, 'user_info') and self.user_info and hasattr(self.user_info, 'tweets'):
                tweet_examples = self.user_info.tweets[:3]  # 取前3条推文作为示例
            
            # 使用 SocialAction 创建示例推文文本
            example_tweets = self.env.action.create_example_tweets_text(tweet_examples)
            
            # 使用原始JSON格式的提示词
            prompt = f"""You are generating tweets for a social media user.

User profile: {user_description}
Example tweets:
{example_tweets}
Maximum tweets: {max_tweets}

Graph-structured context (neighbors + posts):
{graph_prompt}

IMPORTANT: Return ONLY valid JSON array without any comments, explanations, or markdown formatting.

Please generate {max_tweets} tweets that reflect the user's personality, interests, and writing style.
Return the result as a JSON array of strings.

CRITICAL REQUIREMENTS:
1. Create unique, original tweets that match the user's personality and interests
2. Do not copy or closely mimic the example tweets - create fresh content
3. Ensure tweets reflect the user's writing style, tone, and interests
4. Make tweets engaging and authentic to the user's character
5. Include relevant hashtags and mentions when appropriate
6. Ensure tweets are realistic and appropriate for the user's profile

Generate tweets (JSON array only, no comments):"""
            
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.astep(user_msg, response_format=None)
            
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                published_tweets = self._parse_tweets_from_content(content)
            else:
                published_tweets = []
            
            # 真正执行发布推文动作到环境中（与主要方案保持一致）
            for tweet_content in published_tweets:
                try:
                    # 更新用户的推文列表
                    if 'tweets' not in ori_data:
                        ori_data['tweets'] = []
                    if tweet_content not in ori_data['tweets']:
                        ori_data['tweets'].append(tweet_content)
                    
                    # 更新用户的statuses_count
                    if 'profile' in ori_data and isinstance(ori_data['profile'], dict):
                        ori_data['profile']['statuses_count'] = len(ori_data['tweets'])
                    
                    agent_log.debug(f"回退方法智能体 {self.social_agent_id} 成功发布推文: {tweet_content[:50]}...")
                except Exception as e:
                    agent_log.warning(f"回退方法智能体 {self.social_agent_id} 发布推文失败: {e}")
            
            agent_log.info(f"回退方法生成推文: {len(published_tweets)}条")
            return json.dumps(published_tweets, ensure_ascii=False)
            
        except Exception as e:
            agent_log.error(f"回退方法生成推文失败: {e}")
            return "[]"
    



    async def generate_content_by_llm(self, content_type: str, context: str = "") -> str:
        """
        使用LLM生成内容（帖子、评论等）
        
        Args:
            content_type: 内容类型（post, comment, quote等）
            context: 上下文信息
            
        Returns:
            str: 生成的内容
        """
        try:
            # 使用 SocialAction 生成内容提示词
            prompt = self.env.action._generate_content_prompt(content_type, context)
            
            # 创建用户消息
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt)
            
            # 调用LLM
            response = await self.astep(user_msg, response_format=None)
            
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                return content
            else:
                return ""
                
        except Exception as e:
            agent_log.error(f"Failed to generate content: {e}")
            return ""
    
    async def generate_search_query_by_llm(self, search_type: str, environment_data: Dict[str, Any] = None) -> str:
        """
        使用LLM生成搜索查询
        
        Args:
            search_type: 搜索类型（posts, users等）
            environment_data: 环境数据
            
        Returns:
            str: 生成的搜索查询
        """
        try:
            if environment_data is None:
                environment_data = {}
            
            # 使用 SocialAction 生成搜索提示词
            prompt = self.env.action._generate_search_prompt(search_type)
            
            # 创建用户消息
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt)
            
            # 调用LLM
            response = await self.astep(user_msg, response_format=None)
            
            if response.msgs:
                content = response.msgs[-1].content if response.msgs else ""
                return content
            else:
                return ""
                
        except Exception as e:
            agent_log.error(f"Failed to generate search query: {e}")
            return ""
    
    async def make_relationship_decision_by_llm(self, available_options: List[str], context: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        使用LLM做出社交关系决策
        
        Args:
            available_options: 可用的关系操作选项
            context: 上下文信息
            
        Returns:
            Tuple[str, Dict[str, Any]]: (选择的动作, 动作参数)
        """
        try:
            # 使用 SocialAction 生成关系决策提示词
            prompt = self.env.action._generate_action_selection_prompt(context, available_options, {})
            
            # 执行动作生成（简化版本）
            user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)
            response = await self.astep(user_msg)
            
            if response.msgs:
                action_name = response.msgs[-1].content
                action_args = {}
            else:
                action_name = "do_nothing"
                action_args = {}
            
            return action_name, action_args
            
        except Exception as e:
            agent_log.error(f"Failed to make relationship decision: {e}")
            return "do_nothing", {}

    async def adjust_and_regenerate_user_profile(self, explanations_path: str = "social_network/explanations/natural_language_explanations.json", ori_data_template: dict = None, agent_user_id_mapping: dict = None) -> dict:
        """
        1. 读取指定的explanations文件，将内容写入 memory。
        2. 整合 memory 中所有内容。
        3. 参考 generate_profile_and_dict_by_llm 重新生成用户 profile。
        4. 返回 user_dict。
        
        Args:
            explanations_path: explanations文件的路径，默认为"social_network/explanations/natural_language_explanations.json"
            ori_data_template: 可选，原始 ori_data 字典模板（用于补全非 profile 字段）
            agent_user_id_mapping: 可选，agent_id 到原始 ID 的映射
        Returns:
            dict: 满足 str.md 结构的用户字典
        """
        # 1. 读取指定的 explanations 文件
        with open(explanations_path, "r", encoding="utf-8") as f:
            explanations = json.load(f)
        # 2. 过滤与当前智能体相关的解释（若可用）
        filtered = []
        agent_id_str = str(self.social_agent_id)
        user_id_str = ""
        if agent_user_id_mapping:
            user_id_str = str(agent_user_id_mapping.get(self.social_agent_id, agent_user_id_mapping.get(agent_id_str, "")))
        for explanation in explanations:
            if not isinstance(explanation, str):
                continue
            if f"Agent {agent_id_str}" in explanation or (user_id_str and f"User {user_id_str}" in explanation):
                filtered.append(explanation)
        if filtered:
            explanations = filtered

        # 3. 写入 memory（每条作为一条assistant消息）
        for explanation in explanations:
            msg = BaseMessage.make_assistant_message(role_name="assistant", content=explanation)
            record = MemoryRecord(message=msg, role_at_backend=OpenAIBackendRole.ASSISTANT)
            self.memory.write_record(record)
        # 4. 整合 memory 内容为英文不足总结
        all_records = self.memory.retrieve()
        merged_content = "\n".join([r.memory_record.message.content for r in all_records])
        summary_content = (
            "Summary of deficiencies in agent profile generation and graph structure:\n"
            + merged_content
        )
        # 5. 以系统消息写入 memory
        sys_msg = BaseMessage.make_assistant_message(role_name="system", content=summary_content)
        sys_record = MemoryRecord(message=sys_msg, role_at_backend=OpenAIBackendRole.ASSISTANT)
        self.memory.write_record(sys_record)
        # 6. 调用生成函数（使用内置方法）并解析
        profile_json_str = await self.generate_new_user_profile_by_llm()
        profile = {}
        if profile_json_str:
            try:
                cleaned = profile_json_str.strip()
                import re
                match = re.search(r'\\{.*\\}', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(0)
                profile = json.loads(cleaned)
            except Exception:
                profile = {}

        # 7. 组装用户字典（保留原字段，更新 profile）
        if ori_data_template and isinstance(ori_data_template, dict):
            user_dict = dict(ori_data_template)
        elif self.user_info:
            user_dict = self.user_info.to_dict()
        else:
            user_dict = {"id": f"u{self.social_agent_id}", "tweets": [], "domains": [], "neighbor": {"following": [], "follower": []}, "label": 1}

        if profile:
            user_dict["profile"] = profile
            # 同步顶层 id
            if profile.get("id"):
                user_dict["id"] = profile.get("id")

        # 8. 更新智能体内部 profile
        try:
            from .entity import TwitterProfile
            if profile:
                self.user_info.profile = TwitterProfile.from_dict(profile)
        except Exception:
            pass

        return user_dict

    def __str__(self) -> str:
        """
        返回智能体的字符串表示
        
        Returns:
            包含智能体类名、ID和模型类型的字符串
        """
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
