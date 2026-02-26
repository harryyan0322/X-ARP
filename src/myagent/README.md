# 社交网络智能体框架 (Social Network Agent Framework)

## 概述

这是一个基于 CAMEL 库重构的社交网络智能体框架，参考了 BotSim 的设计模式，提供了完整的社交网络模拟环境。框架支持多种社交行为，包括发帖、评论、点赞、关注等，并且集成了 LLM 驱动的智能体决策系统。**新增了提示词管理器集成功能，让智能体能够生成更智能、更个性化的社交动作。**

## 架构设计

### 核心组件

```
myagent/
├── basic.py              # 基础架构：Action 基类和 ActionSpace
├── entity.py             # 实体定义：用户、帖子、评论、群组
├── data_operations.py    # 数据管理：加载、保存、查询
├── environment.py        # 环境管理：状态、动作执行
├── utils.py              # 工具函数：配置、时间、ID生成
├── actions.py            # 动作实现：各种社交行为
# 提示词已直接嵌入到 actions.py 中
├── config.yaml           # 配置文件：参数和路径
└── README.md             # 说明文档
```

### 设计模式

框架采用了以下设计模式：

1. **模块化架构**：各组件独立，易于扩展和维护
2. **配置驱动**：通过 YAML 配置文件统一管理参数
3. **动作系统**：每个动作都是可调用的对象，支持参数验证
4. **环境抽象**：统一的环境接口，支持不同平台
5. **数据管理**：完整的数据加载、保存、查询功能

## 主要特性

### 1. 完整的社交行为支持
- 用户管理：创建、查询、更新用户信息
- 内容管理：发帖、评论、点赞、转发
- 社交关系：关注、取消关注、群组管理
- 搜索功能：帖子搜索、用户搜索
- 推荐系统：基于用户兴趣和社交关系的推荐

### 2. LLM 集成
- 基于 CAMEL 库的智能体系统
- 支持多种 LLM 模型
- 智能决策和行为生成
- 记忆系统：支持写入和读取记忆
- **嵌入提示词：所有提示词直接嵌入到动作类中，简化架构**
- **智能动作生成：基于用户档案和环境信息生成个性化动作**
- **内容生成：使用LLM生成帖子、评论等内容**
- **搜索优化：智能生成搜索查询**

### 3. 数据管理
- 支持多种数据格式（CSV、JSON）
- 自动数据加载和索引
- 关系数据管理
- 统计信息生成

### 4. 可扩展性
- 插件化动作系统
- 可配置的环境参数
- 支持多种社交平台
- 模块化设计
- **嵌入提示词：提示词直接嵌入到动作中，易于维护**
- **统一动作接口：参考 agent_action.py 的设计模式**

## 安装和使用

### 1. 环境要求
```bash
pip install camel-ai pandas pyyaml
```

### 2. 配置设置
编辑 `config.yaml` 文件：
```yaml
# LLM 配置
llm:
  model: "gpt-4o-mini"
  api_key: "your-api-key"
  api_base: "your-api-base"

# 数据路径
paths:
  users: "./data/users.csv"
  posts: "./data/posts.csv"
```

### 3. 基本使用
```python
from myagent.environment import SocialEnvironment
from myagent.utils import load_config

# 加载配置
config = load_config("config.yaml")

# 创建环境
env = SocialEnvironment(config)
init_observation = env.reset()

# 执行动作
from myagent.actions import SocialAction, CreateUserAction

# 创建用户
action = CreateUserAction()
result = env.step(action, {
    "id": "u12345678",
    "screen_name": "test_user",
    "name": "Test User",
    "description": "A test user",
    "location": "Test City",
    "age": 25,
    "gender": "male",
    "mbit": "INTJ",
    "domains": ["technology", "science"],
    "label": 0,
    "tweets": ["Hello world!", "This is a test"],
    "neighbor": {"following": [], "follower": []}
})

# 创建帖子（使用LLM生成内容）
social_action = SocialAction(agent_id=1, agent=agent)
result = await social_action.create_post(
    content="placeholder",  # 将被LLM生成的内容替换
    use_llm_generation=True,  # 启用LLM生成
    context="分享一些想法"
)

# 点赞帖子
result = await social_action.like_post("post_001")

# 关注用户
result = await social_action.follow_user("u87654321")
```

## 动作系统

### 可用动作

通过 `SocialAction` 类提供以下动作：

1. **create_post** - 创建帖子（支持LLM生成内容）
2. **like_post** - 点赞帖子
3. **unlike_post** - 取消点赞
4. **follow_user** - 关注用户
5. **unfollow_user** - 取消关注
6. **search_posts** - 搜索帖子（支持LLM生成搜索查询）
7. **search_users** - 搜索用户（支持LLM生成搜索查询）
8. **get_recommended_posts** - 获取推荐帖子
9. **write_to_memory** - 写入记忆
10. **adjust_profile** - 调整用户画像
11. **do_nothing** - 什么都不做
12. **smart_action_selector** - 智能动作选择器
13. **content_generation** - 内容生成动作

通过 `CreateUserAction` 类提供：
14. **CreateUser** - 创建用户（符合 str.md 格式）

### 动作执行
```python
# 创建 SocialAction 实例
social_action = SocialAction(agent_id=1, agent=agent)

# 执行动作（传统方式）
result = await social_action.create_post(content="Hello, world!")

# 执行动作（使用LLM生成）
result = await social_action.create_post(
    content="placeholder",
    use_llm_generation=True,
    context="分享一些想法"
)
```

### 智能动作生成
```python
# 使用智能动作选择器
result = await social_action.smart_action_selector(
    context="User wants to engage with content",
    available_actions=["like_post", "follow", "create_post"],
    environment_data={"user_count": 100, "post_count": 500}
)

# 使用内容生成动作
result = await social_action.content_generation(
    content_type="post",
    context="Sharing thoughts about AI"
)
```

## 数据格式

框架现在完全支持 `str.md` 中定义的数据格式，确保输入和输出的一致性。

### 用户数据格式（符合 str.md）

每个用户数据为一个 JSON 对象，包含以下字段：

```json
{
  "id": "u17461978",
  "profile": {
    "id": "u17461978",
    "followers_count": 1200,
    "friends_count": 350,
    "listed_count": 15,
    "favourites_count": 500,
    "statuses_count": 3200,
    "age": 29,
    "name": "Alice Smith",
    "screen_name": "alice_smith",
    "location": "New York, USA",
    "description": "Data scientist. Love AI & coffee.",
    "url": "https://twitter.com/alice_smith",
    "created_at": "2012-05-01",
    "lang": "en",
    "gender": "female",
    "mbit": "INTJ",
    "profile_image_url": "https://twitter.com/alice_smith/profile_image",
    "pinned_tweet_id": null,
    "verified": false,
    "protected": false,
    "default_profile_image": false,
    "geo_enabled": true,
    "has_extended_profile": true
  },
  "tweets": [
    "RT @CarnivalCruise: 🎉 Are you ready to see what our newest ship's name will be? 🎉",
    "Who has time for receipts? Not me. @epson receipt scanners make it easy."
  ],
  "neighbor": {
    "following": ["u12345", "u67890", "u24680"],
    "follower": ["u54321", "u98765"]
  },
  "domains": ["AI", "Data Science", "Coffee"],
  "label": 0
}
```

### 数据文件格式

- **users.json**: 每行一个用户 JSON 对象（符合 str.md 格式）
- **posts.csv**: 帖子数据
- **domains.json**: 领域数据（可选，会自动从用户数据中提取）

### 数据转换工具

框架提供了数据转换工具 `data_converter.py`，支持：

```python
from myagent.data_converter import (
    convert_csv_to_str_format,
    convert_str_format_to_csv,
    validate_str_format,
    create_sample_str_format_data
)

# 转换旧格式到新格式
convert_csv_to_str_format("old_users.csv", "new_users.json")

# 验证数据格式
validate_str_format("users.json")

# 创建示例数据
create_sample_str_format_data("sample_users.json", 5)
```

## 框架验证

框架提供了完整的功能验证，包括：
- 环境初始化
- 数据操作
- 动作执行
- 用户信息流
- 搜索功能
- 记忆动作
- 提示词管理器集成
- 智能动作生成

## 扩展开发

### 添加新动作
```python
from myagent.basic import Action

class CustomAction(Action):
    def __init__(self):
        super().__init__(
            name="CustomAction",
            description="A custom action",
            func=self.custom_function,
            input_args_schema={"param": str}
        )
    
    @staticmethod
    def custom_function(action_args, env, agent):
        # 实现动作逻辑
        return "Action completed"
```

### 添加新实体
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CustomEntity:
    entity_id: str
    name: str
    created_time: datetime
    
    def save_to_file(self, filename: str):
        # 实现保存逻辑
        pass
```

## 与原有代码的兼容性

重构后的框架保持了与原有代码的兼容性：

1. **保留原有提示方法**：所有原有的提示模板和生成方法都被保留
2. **保持数据结构**：用户、帖子等数据结构与原有格式一致
3. **兼容原有函数**：`adjust_and_regenerate_user_profile` 等函数仍然可用
4. **扩展功能**：新增了记忆写入动作和环境感知功能

## 主要改进

1. **数据格式标准化**：完全支持 `str.md` 中定义的数据格式
2. **领域管理系统**：支持用户兴趣领域（domains）的管理和查询
3. **点赞系统**：完整的点赞和取消点赞功能，支持用户统计更新
4. **社交关系管理**：关注/取消关注功能，自动更新粉丝和关注数
5. **环境感知**：智能体现在可以感知社交环境状态
6. **动作系统**：标准化的动作执行接口
7. **数据管理**：统一的数据加载和查询接口
8. **配置管理**：集中化的配置管理
9. **测试支持**：完整的测试框架
10. **数据转换工具**：提供格式转换和验证功能
11. **文档完善**：详细的文档和示例

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

Apache License 2.0

## 联系方式

如有问题或建议，请提交 Issue 或联系开发团队。 