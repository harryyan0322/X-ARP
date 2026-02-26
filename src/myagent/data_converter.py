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

"""
数据格式转换工具

用于在旧格式和新格式（str.md 格式）之间转换用户数据
"""

import json
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .entity import SocialUser, TwitterProfile

logger = logging.getLogger(__name__)


def convert_csv_to_str_format(csv_file: str, output_file: str) -> bool:
    """
    将 CSV 格式的用户数据转换为 str.md 格式
    
    Args:
        csv_file: 输入的 CSV 文件路径
        output_file: 输出的 JSON 文件路径
    
    Returns:
        bool: 转换是否成功
    """
    try:
        users = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                
                # 创建 SocialUser
                user = SocialUser(
                    id=str(row.get("user_id", "")),
                    profile=profile,
                    tweets=json.loads(row.get("tweets", "[]")),
                    neighbor=json.loads(row.get("neighbor", '{"following": [], "follower": []}')),
                    domains=json.loads(row.get("domains", "[]")),
                    label=int(row.get("label", 0))
                )
                
                users.append(user.to_dict())
        
        # 保存为 JSON 格式
        with open(output_file, 'w', encoding='utf-8') as f:
            for user in users:
                json.dump(user, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"Successfully converted {len(users)} users from {csv_file} to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert CSV to str format: {e}")
        return False


def convert_str_format_to_csv(json_file: str, output_file: str) -> bool:
    """
    将 str.md 格式的用户数据转换为 CSV 格式
    
    Args:
        json_file: 输入的 JSON 文件路径
        output_file: 输出的 CSV 文件路径
    
    Returns:
        bool: 转换是否成功
    """
    try:
        users = []
        
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    user_data = json.loads(line.strip())
                    user = SocialUser.from_dict(user_data)
                    users.append(user)
        
        # 写入 CSV 文件
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'user_id', 'followers_count', 'friends_count', 'listed_count',
                'favourites_count', 'statuses_count', 'age', 'name', 'screen_name',
                'location', 'description', 'url', 'created_at', 'lang', 'gender',
                'mbit', 'profile_image_url', 'pinned_tweet_id', 'verified',
                'protected', 'default_profile_image', 'geo_enabled',
                'has_extended_profile', 'tweets', 'neighbor', 'domains', 'label'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for user in users:
                row = {
                    'user_id': user.id,
                    'followers_count': user.profile.followers_count,
                    'friends_count': user.profile.friends_count,
                    'listed_count': user.profile.listed_count,
                    'favourites_count': user.profile.favourites_count,
                    'statuses_count': user.profile.statuses_count,
                    'age': user.profile.age,
                    'name': user.profile.name,
                    'screen_name': user.profile.screen_name,
                    'location': user.profile.location,
                    'description': user.profile.description,
                    'url': user.profile.url,
                    'created_at': user.profile.created_at,
                    'lang': user.profile.lang,
                    'gender': user.profile.gender,
                    'mbit': user.profile.mbit,
                    'profile_image_url': user.profile.profile_image_url,
                    'pinned_tweet_id': user.profile.pinned_tweet_id,
                    'verified': user.profile.verified,
                    'protected': user.profile.protected,
                    'default_profile_image': user.profile.default_profile_image,
                    'geo_enabled': user.profile.geo_enabled,
                    'has_extended_profile': user.profile.has_extended_profile,
                    'tweets': json.dumps(user.tweets, ensure_ascii=False),
                    'neighbor': json.dumps(user.neighbor, ensure_ascii=False),
                    'domains': json.dumps(user.domains, ensure_ascii=False),
                    'label': user.label
                }
                writer.writerow(row)
        
        logger.info(f"Successfully converted {len(users)} users from {json_file} to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert str format to CSV: {e}")
        return False


def validate_str_format(data_file: str) -> bool:
    """
    验证数据文件是否符合 str.md 格式
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        bool: 是否符合格式
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    user_data = json.loads(line.strip())
                    
                    # 检查必需字段
                    required_fields = ['id', 'profile', 'tweets', 'neighbor', 'domains', 'label']
                    for field in required_fields:
                        if field not in user_data:
                            logger.error(f"Missing required field '{field}' in line {i}")
                            return False
                    
                    # 检查 profile 字段
                    profile = user_data['profile']
                    profile_fields = [
                        'id', 'followers_count', 'friends_count', 'listed_count',
                        'favourites_count', 'statuses_count', 'name', 'screen_name',
                        'description', 'created_at', 'lang', 'verified'
                    ]
                    for field in profile_fields:
                        if field not in profile:
                            logger.error(f"Missing required profile field '{field}' in line {i}")
                            return False
                    
                    # 检查数据类型
                    if not isinstance(user_data['tweets'], list):
                        logger.error(f"Field 'tweets' must be a list in line {i}")
                        return False
                    
                    if not isinstance(user_data['domains'], list):
                        logger.error(f"Field 'domains' must be a list in line {i}")
                        return False
                    
                    if not isinstance(user_data['label'], int):
                        logger.error(f"Field 'label' must be an integer in line {i}")
                        return False
        
        logger.info(f"Data file {data_file} is valid str.md format")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate str format: {e}")
        return False


def create_sample_str_format_data(output_file: str, num_users: int = 5) -> bool:
    """
    创建示例的 str.md 格式数据
    
    Args:
        output_file: 输出文件路径
        num_users: 用户数量
    
    Returns:
        bool: 创建是否成功
    """
    try:
        sample_users = []
        
        for i in range(num_users):
            # 创建示例 profile
            profile = TwitterProfile(
                id=f"u{10000000 + i}",
                followers_count=100 + i * 50,
                friends_count=50 + i * 25,
                listed_count=5 + i,
                favourites_count=200 + i * 100,
                statuses_count=150 + i * 75,
                age=25 + i * 5,
                name=f"Sample User {i + 1}",
                screen_name=f"sample_user_{i + 1}",
                location=f"City {i + 1}",
                description=f"This is sample user {i + 1} for testing",
                url="",
                created_at="2020-01-01",
                lang="en",
                gender="male" if i % 2 == 0 else "female",
                mbit="INTJ" if i % 2 == 0 else "ENFP",
                profile_image_url="",
                pinned_tweet_id=None,
                verified=False,
                protected=False,
                default_profile_image=False,
                geo_enabled=False,
                has_extended_profile=False
            )
            
            # 创建示例用户
            user = SocialUser(
                id=f"u{10000000 + i}",
                profile=profile,
                tweets=[
                    f"This is sample tweet {j + 1} from user {i + 1}"
                    for j in range(3)
                ],
                neighbor={
                    "following": [f"u{10000000 + (i + 1) % num_users}"],
                    "follower": [f"u{10000000 + (i - 1) % num_users}"]
                },
                domains=["technology", "science", "art"][i % 3:],
                label=i % 2  # 交替设置为真人和机器人
            )
            
            sample_users.append(user.to_dict())
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for user in sample_users:
                json.dump(user, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"Successfully created sample data with {num_users} users in {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    create_sample_str_format_data("./data/sample_users.json", 5)
    
    # 验证数据格式
    validate_str_format("./data/sample_users.json") 