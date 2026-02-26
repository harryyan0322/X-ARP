import json
import re
from typing import List, Tuple


def replace_user_ids_with_agent_ids(
    mapping_file_path: str, 
    explanations_file_path: str
) -> Tuple[List[str], List[str]]:
    """
    将自然语言解释文件中的用户ID替换为对应的Agent ID，并返回需要调整的Agent ID列表
    
    Args:
        mapping_file_path: 用户ID到Agent ID的映射文件路径
        explanations_file_path: 自然语言解释文件路径
        
    Returns:
        Tuple[List[str], List[str]]: 
            - 第一个元素：替换后的自然语言解释列表
            - 第二个元素：需要调整的Agent ID列表
    """
    # 1. 读取用户ID到Agent ID的映射
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    
    # 2. 读取自然语言解释文件
    with open(explanations_file_path, 'r', encoding='utf-8') as f:
        explanations = json.load(f)
    
    # 3. 创建反向映射（从用户ID到Agent ID）
    user_to_agent_mapping = {}
    for agent_id, user_id in id_mapping.items():
        user_to_agent_mapping[str(user_id)] = str(agent_id)
    
    # 4. 替换解释中的用户ID为Agent ID
    updated_explanations = []
    agent_ids_to_adjust = set()
    
    for explanation in explanations:
        updated_explanation = explanation
        
        # 使用正则表达式查找所有 "User X" 格式的用户ID
        user_id_pattern = r'User (\d+)'
        
        def replace_user_id(match):
            user_id = match.group(1)
            if user_id in user_to_agent_mapping:
                agent_id = user_to_agent_mapping[user_id]
                agent_ids_to_adjust.add(agent_id)
                return f'Agent {agent_id}'
            else:
                # 如果找不到映射，保持原样
                return match.group(0)
        
        # 替换所有匹配的用户ID
        updated_explanation = re.sub(user_id_pattern, replace_user_id, updated_explanation)
        updated_explanations.append(updated_explanation)
    
    # 5. 返回结果
    return updated_explanations, list(agent_ids_to_adjust)


def save_updated_explanations(
    updated_explanations: List[str], 
    output_file_path: str
) -> None:
    """
    保存更新后的自然语言解释到文件
    
    Args:
        updated_explanations: 更新后的自然语言解释列表
        output_file_path: 输出文件路径
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_explanations, f, ensure_ascii=False, indent=2)
