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

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import csv


@dataclass
class SocialPost:
    """社交帖子实体类"""
    post_id: str
    author_id: str
    author_name: str
    content: str
    created_time: datetime
    platform: str = "twitter"  # twitter, reddit, etc.
    score: int = 0
    num_comments: int = 0
    num_likes: int = 0
    num_dislikes: int = 0
    num_reposts: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # comments: List["SocialComment"] = field(default_factory=list)  # 暂时注释掉

    # def add_comment(self, comment: "SocialComment"):
    #     """添加评论"""
    #     self.comments.append(comment)
    #     self.num_comments += 1

    def like(self):
        """点赞"""
        self.num_likes += 1

    def unlike(self):
        """取消点赞"""
        if self.num_likes > 0:
            self.num_likes -= 1

    def dislike(self):
        """点踩"""
        self.num_dislikes += 1

    def undislike(self):
        """取消点踩"""
        if self.num_dislikes > 0:
            self.num_dislikes -= 1

    def repost(self):
        """转发"""
        self.num_reposts += 1

    def save_to_file(self, filename: str):
        """保存到文件"""
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.post_id,
                self.author_id,
                self.author_name,
                self.content,
                self.created_time,
                self.platform,
                self.score,
                self.num_comments,
                self.num_likes,
                self.num_dislikes,
                self.num_reposts,
                json.dumps(self.tags),
                json.dumps(self.metadata)
            ])


@dataclass
class SocialComment:
    """社交评论实体类"""
    comment_id: str
    post_id: str
    author_id: str
    author_name: str
    content: str
    created_time: datetime
    parent_id: Optional[str] = None  # 用于回复其他评论
    score: int = 0
    num_likes: int = 0
    num_dislikes: int = 0
    platform: str = "twitter"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def like(self):
        """点赞"""
        self.num_likes += 1

    def unlike(self):
        """取消点赞"""
        if self.num_likes > 0:
            self.num_likes -= 1

    def dislike(self):
        """点踩"""
        self.num_dislikes += 1

    def undislike(self):
        """取消点踩"""
        if self.num_dislikes > 0:
            self.num_dislikes -= 1

    def save_to_file(self, filename: str):
        """保存到文件"""
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.comment_id,
                self.post_id,
                self.author_id,
                self.author_name,
                self.content,
                self.created_time,
                self.parent_id,
                self.score,
                self.num_likes,
                self.num_dislikes,
                self.platform,
                json.dumps(self.metadata)
            ])


@dataclass
class TwitterProfile:
    """Twitter 用户资料，对应 str.md 中的 profile 字段"""
    id: str  # 用户 ID（唯一标识，带前缀 'u'）
    followers_count: int = 0
    friends_count: int = 0
    listed_count: int = 0
    favourites_count: int = 0
    statuses_count: int = 0
    age: Optional[int] = None
    name: str = ""
    screen_name: str = ""
    location: str = ""
    description: str = ""
    url: str = ""
    created_at: str = ""
    lang: str = "en"
    gender: str = ""
    mbit: str = ""
    profile_image_url: str = ""
    pinned_tweet_id: Optional[str] = None
    verified: bool = False
    protected: bool = False
    default_profile_image: bool = False
    geo_enabled: bool = False
    has_extended_profile: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "followers_count": self.followers_count,
            "friends_count": self.friends_count,
            "listed_count": self.listed_count,
            "favourites_count": self.favourites_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "name": self.name,
            "screen_name": self.screen_name,
            "location": self.location,
            "description": self.description,
            "url": self.url,
            "created_at": self.created_at,
            "lang": self.lang,
            "gender": self.gender,
            "mbit": self.mbit,
            "profile_image_url": self.profile_image_url,
            "pinned_tweet_id": self.pinned_tweet_id,
            "verified": self.verified,
            "protected": self.protected,
            "default_profile_image": self.default_profile_image,
            "geo_enabled": self.geo_enabled,
            "has_extended_profile": self.has_extended_profile
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwitterProfile":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class SocialUser:
    """社交用户实体类，对应 str.md 中的用户数据格式"""
    id: str  # 用户唯一 ID
    profile: TwitterProfile
    tweets: List[str] = field(default_factory=list)  # 推文文本序列
    neighbor: Optional[Dict[str, List[str]]] = None  # 图结构邻居关系
    domains: List[str] = field(default_factory=list)  # 用户主题/兴趣领域标签
    label: int = 0  # 是否为机器人（0真人，1机器人）
    
    # 内部关系数据（不保存到文件）
    posts: Dict[str, SocialPost] = field(default_factory=dict)
    # comments: List[SocialComment] = field(default_factory=list)  # 暂时注释掉

    def __post_init__(self):
        """初始化后处理"""
        if self.neighbor is None:
            self.neighbor = {"following": [], "follower": []}

    def add_post(self, post: SocialPost):
        """添加帖子"""
        self.posts[post.post_id] = post

    # def add_comment(self, comment: SocialComment):
    #     """添加评论"""
    #     self.comments.append(comment)

    def add_tweet(self, tweet: str):
        """添加推文"""
        self.tweets.append(tweet)

    def add_following(self, user_id: str):
        """添加关注"""
        if user_id not in self.neighbor["following"]:
            self.neighbor["following"].append(user_id)

    def add_follower(self, user_id: str):
        """添加粉丝"""
        if user_id not in self.neighbor["follower"]:
            self.neighbor["follower"].append(user_id)

    def remove_following(self, user_id: str):
        """移除关注"""
        if user_id in self.neighbor["following"]:
            self.neighbor["following"].remove(user_id)

    def remove_follower(self, user_id: str):
        """移除粉丝"""
        if user_id in self.neighbor["follower"]:
            self.neighbor["follower"].remove(user_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 str.md 格式的字典"""
        return {
            "id": self.id,
            "profile": self.profile.to_dict(),
            "tweets": self.tweets,
            "neighbor": self.neighbor,
            "domains": self.domains,
            "label": self.label
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialUser":
        """从 str.md 格式的字典创建实例"""
        profile_data = data.get("profile", {})
        profile = TwitterProfile.from_dict(profile_data)
        
        return cls(
            id=data.get("id", ""),
            profile=profile,
            tweets=data.get("tweets", []),
            neighbor=data.get("neighbor"),
            domains=data.get("domains", []),
            label=data.get("label", 0)
        )

    def save_to_file(self, filename: str):
        """保存为 JSON 格式（符合 str.md 格式）"""
        import json
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
            f.write("\n")


@dataclass
class Domain:
    """兴趣领域/主题分类，对应 str.md 中的 domains 字段"""
    domain_id: str
    domain_name: str
    description: str
    category: str = "general"  # 分类：politics, business, entertainment, sports, technology, etc.
    popularity: int = 0  # 该领域的受欢迎程度
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 关系数据
    users: List[str] = field(default_factory=list)  # 对该领域感兴趣的用户ID列表
    posts: List[str] = field(default_factory=list)  # 该领域的帖子ID列表

    def add_user(self, user_id: str):
        """添加对该领域感兴趣的用户"""
        if user_id not in self.users:
            self.users.append(user_id)
            self.popularity += 1

    def remove_user(self, user_id: str):
        """移除用户兴趣"""
        if user_id in self.users:
            self.users.remove(user_id)
            self.popularity -= 1

    def add_post(self, post_id: str):
        """添加该领域的帖子"""
        if post_id not in self.posts:
            self.posts.append(post_id)

    def remove_post(self, post_id: str):
        """移除帖子"""
        if post_id in self.posts:
            self.posts.remove(post_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "description": self.description,
            "category": self.category,
            "popularity": self.popularity,
            "metadata": self.metadata,
            "user_count": len(self.users),
            "post_count": len(self.posts)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Domain":
        """从字典创建实例"""
        return cls(**data)

    def save_to_file(self, filename: str):
        """保存为 JSON 格式"""
        import json
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
            f.write("\n")


# @dataclass
# class SocialGroup:
#     """社交群组实体类（兼容性保留）"""
#     group_id: str
#     group_name: str
#     description: str
#     created_time: datetime
#     creator_id: str
#     platform: str = "twitter"
#     member_count: int = 0
#     is_private: bool = False
#     metadata: Dict[str, Any] = field(default_factory=dict)
    
#     # 成员关系
#     members: List[str] = field(default_factory=list)  # 成员用户ID列表
#     messages: List[Dict[str, Any]] = field(default_factory=list)  # 群组消息

#     def add_member(self, user_id: str):
#         """添加成员"""
#         if user_id not in self.members:
#             self.members.append(user_id)
#             self.member_count += 1

#     def remove_member(self, user_id: str):
#         """移除成员"""
#         if user_id in self.members:
#             self.members.remove(user_id)
#             self.member_count -= 1

#     def add_message(self, message: Dict[str, Any]):
#         """添加消息"""
#         self.messages.append(message)

#     def save_to_file(self, filename: str):
#         """保存到文件"""
#         with open(filename, "a", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 self.group_id,
#                 self.group_name,
#                 self.description,
#                 self.created_time,
#                 self.creator_id,
#                 self.platform,
#                 self.member_count,
#                 self.is_private,
#                 json.dumps(self.members),
#                 json.dumps(self.messages),
#                 json.dumps(self.metadata)
#             ]) 
