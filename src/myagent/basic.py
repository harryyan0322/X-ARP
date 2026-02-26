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

from typing import List, Optional, Callable, Dict, Any, Type
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class Action(BaseModel):
    """
    动作基类，定义所有社交动作的基本结构
    
    参考 BotSim 框架的设计模式，每个动作都是可调用的对象，
    包含名称、描述、执行函数和参数模式
    """
    name: str
    description: str
    func: Optional[Callable[..., str]]
    input_args_schema: Optional[Dict[str, Any]] = None
    enable: bool = True
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_function(
            cls,
            func: Optional[Callable],
            name: str,
            description: str,
            input_args_schema: dict,
            **kwargs: Any,
    ):
        """从函数创建动作实例"""
        return cls(
            name=name,
            func=func,
            description=description,
            input_args_schema=input_args_schema,
            **kwargs,
        )

    def args_check(self, args, args_schema):
        """检查输入参数是否符合模式定义"""
        if args_schema is None:
            return None
            
        for arg_name, arg_type in args_schema.items():
            if arg_name not in args:
                return f"Missing argument: {arg_name}"
            if not isinstance(args[arg_name], arg_type):
                return f"Invalid argument type for {arg_name}: {type(args[arg_name])}, expected {arg_type}"
        return None 

    def __call__(
            self,
            input_action_args: Dict[str, Any],
            env: Any,
            agent: Any,
    ) -> str:
        """执行动作的调用接口"""
        error_message = self.args_check(input_action_args, self.input_args_schema)
        if error_message:
            logger.error(f"Action {self.name} argument check failed: {error_message}")
            return error_message
        else:
            try:
                action_output = self.func(input_action_args, env, agent)
                return action_output
            except Exception as e:
                logger.error(f"Action {self.name} execution failed: {e}")
                return f"Action execution failed: {str(e)}"


class ActionSpace:
    """动作空间类，管理所有可用的动作"""
    
    def __init__(self):
        self.actions: List[Action] = []
        self.action_names: List[str] = []

    def add_action(self, action: Action):
        """添加动作到动作空间"""
        self.actions.append(action)
        self.action_names.append(action.name)

    def get_action(self, action_name: str) -> Optional[Action]:
        """根据名称获取动作"""
        for action in self.actions:
            if action.name == action_name:
                return action
        return None

    def get_all_actions(self) -> List[Action]:
        """获取所有动作"""
        return self.actions

    def get_enabled_actions(self) -> List[Action]:
        """获取所有启用的动作"""
        return [action for action in self.actions if action.enable]

    def __repr__(self):
        return [{"name": action.name, "description": action.description, 
                 "input_args": action.input_args_schema} for action in self.actions] 