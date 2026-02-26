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

import yaml
import logging
import os
import importlib
import json
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional
import math

logger = logging.getLogger(__name__)


def load_config(filename: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config from {filename}: {e}")
        return {}


def str_to_datetime(time_str: str) -> datetime:
    """字符串转换为datetime对象"""
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            logger.warning(f"Invalid time format: {time_str}, using current time")
            return datetime.now()


def check_time(cur_time: datetime, config: Dict[str, Any]) -> bool:
    """检查时间是否在配置的时间范围内"""
    start_time = str_to_datetime(config.get("time", {}).get("start_time", "2020-01-01 00:00:00"))
    end_time = str_to_datetime(config.get("time", {}).get("end_time", "2030-12-31 23:59:59"))
    return start_time <= cur_time <= end_time


def generate_id(existing_ids: List[str] = None, length: int = 7) -> str:
    """生成唯一ID"""
    if existing_ids is None:
        existing_ids = []
    
    characters = string.ascii_lowercase + string.digits
    max_retries = 1000
    
    for _ in range(max_retries):
        new_id = "".join(random.choice(characters) for _ in range(length))
        if new_id not in existing_ids:
            return new_id
    
    # 如果重试次数过多，增加长度
    return generate_id(existing_ids, length + 1)


def load_action_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """加载动作空间"""
    action_space = {}
    actions_config = config.get("actions", {})
    
    for action_name, action_path in actions_config.items():
        try:
            module_name, class_name = action_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            action_class = getattr(module, class_name)
            input_args_schema = getattr(action_class(), "input_args_schema", None)
            description = getattr(action_class(), "description", None)
            action_space[action_name] = (action_class, input_args_schema, description)
        except Exception as e:
            logger.error(f"Failed to load action {action_name}: {e}")
    
    return action_space


def get_action_info(action_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """获取动作信息"""
    return [
        {
            "ActionName": action_name,
            "ActionArgs": list(input_args_schema.keys()) if input_args_schema else [],
            "Description": description or "",
        }
        for action_name, (
            action_class,
            input_args_schema,
            description,
        ) in action_space.items()
    ]


def set_logger(config: Dict[str, Any], name: str = "social.agent") -> logging.Logger:
    """设置日志记录器"""
    log_config = config.get("log", {})
    log_file = log_config.get("log_file", "social_agent.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 创建输出目录
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_folder = os.path.join(output_folder, "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    
    # 避免重复添加handler
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def load_start_time(config: Dict[str, Any]) -> str:
    """加载开始时间"""
    return config.get("time", {}).get("start_time", "2020-01-01 00:00:00")


def load_end_time(config: Dict[str, Any]) -> str:
    """加载结束时间"""
    return config.get("time", {}).get("end_time", "2030-12-31 23:59:59")


def load_init_observation(env_action_info: List[Dict[str, Any]], 
                         platform: str = "twitter",
                         start_time: str = None,
                         end_time: str = None) -> str:
    """加载初始观察"""
    action_descriptions = []
    for action_info in env_action_info:
        action_desc = f"- {action_info['ActionName']}: {action_info['Description']}"
        if action_info['ActionArgs']:
            action_desc += f" (Args: {', '.join(action_info['ActionArgs'])})"
        action_descriptions.append(action_desc)
    
    actions_text = "\n".join(action_descriptions)
    
    time_info = ""
    if start_time and end_time:
        time_info = f"System Start Time: {start_time}, System End Time: {end_time}"
    
    return f"""
System Overview:
Welcome to the {platform.capitalize()} Simulation System! This system aims to replicate the functionalities and interactions observed on the {platform.capitalize()} platform. Users can engage in various actions.

{time_info}

Action Space:
The system provides the following actions for intelligent agents to execute:
{actions_text}

The users can select actions and specify parameters. The system will simulate user interactions based on these actions, aiming to emulate real {platform.capitalize()} behavior.

The users must conduct their actions within the specified time frame, which ranges from the system's start time to its end time. It is imperative for agents to ensure that their actions fall within this time period.
"""


def epoch_seconds(date: datetime) -> float:
    """获取时间戳"""
    return date.timestamp()


def hot(ups: int, downs: int, date: datetime) -> float:
    """计算热度分数（Reddit算法）"""
    s = ups - downs
    order = math.log10(max(abs(s), 1))
    if s > 0:
        sign = 1
    elif s < 0:
        sign = -1
    else:
        sign = 0
    seconds = epoch_seconds(date) - 1134028003
    return round(sign * order + seconds / 45000, 7)


def hot_by_score_time(score: int, date: datetime) -> float:
    """根据分数和时间计算热度"""
    return hot(score, 0, date)


def list2str(input_list: List[Any]) -> str:
    """列表转字符串"""
    return json.dumps(input_list, ensure_ascii=False)


def format_time(time: datetime) -> str:
    """格式化时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def generate_reddit_id(existing_ids: List[str] = None, length: int = 7, max_retries: int = 1000) -> str:
    """生成Reddit风格的ID"""
    if existing_ids is None:
        existing_ids = []
    
    characters = string.ascii_lowercase + string.digits
    existing_ids_set = set(existing_ids)

    for _ in range(max_retries):
        reddit_id = "".join(random.choice(characters) for _ in range(length))
        if reddit_id not in existing_ids_set:
            return reddit_id
    
    # 如果重试次数过多，增加长度
    return generate_reddit_id(existing_ids, length + 1, max_retries)


def load_user_data(file_path: str) -> List[Dict[str, Any]]:
    """加载用户数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load user data from {file_path}: {e}")
        return []


def save_user_data(data: List[Dict[str, Any]], file_path: str):
    """保存用户数据"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save user data to {file_path}: {e}")


def create_directory_if_not_exists(directory: str):
    """创建目录（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名"""
    return os.path.splitext(file_path)[1].lower()


def is_valid_json(data: str) -> bool:
    """检查字符串是否为有效的JSON"""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def safe_json_loads(data: str, default: Any = None) -> Any:
    """安全的JSON解析"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {data}")
        return default


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """合并字典，dict2的值会覆盖dict1的值"""
    result = dict1.copy()
    result.update(dict2)
    return result


def filter_dict_by_keys(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """根据键列表过滤字典"""
    return {k: v for k, v in data.items() if k in keys}


def get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """获取嵌套字典的值"""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_nested_value(data: Dict[str, Any], keys: List[str], value: Any):
    """设置嵌套字典的值"""
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value 