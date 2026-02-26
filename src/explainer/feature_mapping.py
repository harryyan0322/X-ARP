# 特征-字段映射表
# 根据特征映射说明文档重写，用于将模型输入的特征向量各维映射回原始profile字段

# =====================
# 1. 数值特征映射 (num_properties_tensor.pt)
# =====================
# shape: [num_nodes, 6]
NUMERICAL_FIELDS = [
    'followers_count',      # 粉丝数（对数变换）
    'friends_count',        # 关注人数（对数变换）
    'listed_count',         # 被加入公开列表数（对数变换）
    'favourites_count',     # 点赞总数（对数变换）
    'tweet_count',          # 发推总数（对数变换）
    'account_age_days'      # 账户创建天数（对数变换）
]

# =====================
# 2. 类别特征映射 (cat_properties_tensor.pt)
# =====================
# shape: [num_nodes, 14+N1+1+1] (N1=5, 总共21维)
CATEGORICAL_FIELDS = [
    # 布尔特征 (7维)
    'verified',                    # 是否为认证用户
    'protected',                   # 是否为私密账号
    'default_profile_image',       # 是否使用默认头像
    'geo_enabled',                 # 是否开启地理位置功能
    'has_extended_profile',        # 是否使用扩展资料页
    'url_exists',                  # profile.url是否存在
    'pinned_tweet_exists',         # pinned_tweet_id是否为null
    
    # 编码特征 (4维)
    'gender',                      # 性别编码 (male=0.0, female=1.0, other=2.0)
    'lang',                        # 语言编码 (Top5+其他)
    'mbit',                        # MBTI编码 (4位二进制转十进制)
    'domain_0',                    # domain多标签第0维
    'domain_1',                    # domain多标签第1维
    'domain_2',                    # domain多标签第2维
    'domain_3',                    # domain多标签第3维
    'domain_4',                    # domain多标签第4维
    'location',                    # 地理位置编码 (Top20+其他)
    'age_group'                    # 年龄段编码 (0-6)
]

# =====================
# 3. 其他特征映射 (num_for_h.pt)
# =====================
# shape: [num_nodes, 9]
OTHER_FIELDS = [
    'avg_tweet_length',            # 平均推文长度（对数变换）
    'avg_mention_count',           # 平均@数（对数变换）
    'avg_hashtag_count',           # 平均#数（对数变换）
    'url_ratio',                   # 含URL比例（0-1）
    'avg_emoji_count',             # 平均emoji数量（对数变换）
    'neighbor_total',              # neighbor关注+粉丝数总和（对数变换）
    'follower_friend_ratio',       # 粉丝关注比（对数变换）
    'mbti_decimal',                # MBTI 4位二进制转十进制
    'has_pinned_tweet'             # 是否有置顶推文（1/0）
]

# =====================
# 4. 特征维度映射表
# =====================
# 格式: (字段名, 起始索引, 结束索引, 特征类型)
FIELD_INDEX_MAPPING = [
    # 数值特征 (0-5)
    ('followers_count', 0, 0, 'numerical'),
    ('friends_count', 1, 1, 'numerical'),
    ('listed_count', 2, 2, 'numerical'),
    ('favourites_count', 3, 3, 'numerical'),
    ('tweet_count', 4, 4, 'numerical'),
    ('account_age_days', 5, 5, 'numerical'),
    
    # 类别特征 (6-26)
    ('verified', 6, 6, 'categorical'),
    ('protected', 7, 7, 'categorical'),
    ('default_profile_image', 8, 8, 'categorical'),
    ('geo_enabled', 9, 9, 'categorical'),
    ('has_extended_profile', 10, 10, 'categorical'),
    ('url_exists', 11, 11, 'categorical'),
    ('pinned_tweet_exists', 12, 12, 'categorical'),
    ('gender', 13, 13, 'categorical'),
    ('lang', 14, 14, 'categorical'),
    ('mbit', 15, 15, 'categorical'),
    ('domain_0', 16, 16, 'categorical'),
    ('domain_1', 17, 17, 'categorical'),
    ('domain_2', 18, 18, 'categorical'),
    ('domain_3', 19, 19, 'categorical'),
    ('domain_4', 20, 20, 'categorical'),
    ('location', 21, 21, 'categorical'),
    ('age_group', 22, 22, 'categorical'),
    
    # 其他特征 (23-31)
    ('avg_tweet_length', 23, 23, 'other'),
    ('avg_mention_count', 24, 24, 'other'),
    ('avg_hashtag_count', 25, 25, 'other'),
    ('url_ratio', 26, 26, 'other'),
    ('avg_emoji_count', 27, 27, 'other'),
    ('neighbor_total', 28, 28, 'other'),
    ('follower_friend_ratio', 29, 29, 'other'),
    ('mbti_decimal', 30, 30, 'other'),
    ('has_pinned_tweet', 31, 31, 'other')
]

# =====================
# 5. 字段分组
# =====================
FIELD_GROUPS = {
    'numerical': NUMERICAL_FIELDS,
    'categorical': CATEGORICAL_FIELDS,
    'other': OTHER_FIELDS
}

# =====================
# 6. 特征描述
# =====================
FIELD_DESCRIPTIONS = {
    # 数值特征描述
    'followers_count': '用户粉丝数量（对数变换）',
    'friends_count': '用户关注人数（对数变换）',
    'listed_count': '被加入公开列表数（对数变换）',
    'favourites_count': '点赞总数（对数变换）',
    'tweet_count': '发推总数（对数变换）',
    'account_age_days': '账户创建天数（对数变换）',
    
    # 类别特征描述
    'verified': '是否为Twitter认证用户',
    'protected': '是否为私密账号',
    'default_profile_image': '是否使用默认头像',
    'geo_enabled': '是否开启地理位置功能',
    'has_extended_profile': '是否使用扩展资料页',
    'url_exists': '个人资料页是否包含URL',
    'pinned_tweet_exists': '是否有置顶推文',
    'gender': '用户性别（0=男性，1=女性，2=其他）',
    'lang': '用户语言偏好编码',
    'mbit': 'MBTI性格类型编码',
    'domain_0': '用户兴趣领域标签1',
    'domain_1': '用户兴趣领域标签2',
    'domain_2': '用户兴趣领域标签3',
    'domain_3': '用户兴趣领域标签4',
    'domain_4': '用户兴趣领域标签5',
    'location': '用户地理位置编码',
    'age_group': '用户年龄段（0=未成年，1=青年，2=中青年，3=中年，4=中老年，5=老年，6=未知）',
    
    # 其他特征描述
    'avg_tweet_length': '平均推文长度（对数变换）',
    'avg_mention_count': '平均每条推文@用户数（对数变换）',
    'avg_hashtag_count': '平均每条推文#标签数（对数变换）',
    'url_ratio': '推文中包含URL的比例（0-1）',
    'avg_emoji_count': '平均每条推文表情符号数（对数变换）',
    'neighbor_total': '邻居用户总数（对数变换）',
    'follower_friend_ratio': '粉丝关注比例（对数变换）',
    'mbti_decimal': 'MBTI性格类型十进制编码',
    'has_pinned_tweet': '是否有置顶推文（1=是，0=否）'
}

# =====================
# 7. 工具函数
# =====================
def get_field_slices():
    """获取字段切片信息"""
    field_slices = {}
    for field_name, start_idx, end_idx, field_type in FIELD_INDEX_MAPPING:
        field_slices[field_name] = {
            'start': start_idx,
            'end': end_idx,
            'type': field_type,
            'description': FIELD_DESCRIPTIONS.get(field_name, '')
        }
    return field_slices

def get_feature_dimensions():
    """获取各特征组的维度信息"""
    return {
        'numerical': len(NUMERICAL_FIELDS),
        'categorical': len(CATEGORICAL_FIELDS),
        'other': len(OTHER_FIELDS),
        'total': len(NUMERICAL_FIELDS) + len(CATEGORICAL_FIELDS) + len(OTHER_FIELDS)
    }

def get_field_by_index(index):
    """根据索引获取字段信息"""
    for field_name, start_idx, end_idx, field_type in FIELD_INDEX_MAPPING:
        if start_idx <= index <= end_idx:
            return {
                'name': field_name,
                'type': field_type,
                'description': FIELD_DESCRIPTIONS.get(field_name, '')
            }
    return None

# =====================
# 8. 验证函数
# =====================
def validate_feature_mapping():
    """验证特征映射的完整性"""
    total_features = len(NUMERICAL_FIELDS) + len(CATEGORICAL_FIELDS) + len(OTHER_FIELDS)
    mapped_features = len(FIELD_INDEX_MAPPING)
    
    print(f"特征映射验证:")
    print(f"  数值特征: {len(NUMERICAL_FIELDS)} 维")
    print(f"  类别特征: {len(CATEGORICAL_FIELDS)} 维")
    print(f"  其他特征: {len(OTHER_FIELDS)} 维")
    print(f"  总特征数: {total_features}")
    print(f"  映射特征数: {mapped_features}")
    
    if total_features == mapped_features:
        print("✅ 特征映射完整")
        return True
    else:
        print("❌ 特征映射不完整")
        return False

if __name__ == '__main__':
    validate_feature_mapping() 