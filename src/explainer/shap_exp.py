"""
LIME/SHAP 可解释性分析主脚本
===========================
功能：
- 对判别为“机器人”的用户，分析 profile 文件中除 bio/推文描述外的所有字段对模型判别的影响
- 输出每个字段的重要性分数及排名前几的字段名称
- 支持 SHAP 可解释方法

主要流程：
1. 加载模型与数据
   - 加载训练好的判别模型
   - 加载 processed_data/generated_profiles/ 下的特征张量
   - 加载 feature_mapping.py 字段映射表

2. 筛选目标用户
   - 用模型对所有用户进行预测
   - 仅保留被判为“机器人”的用户

3. 特征还原与准备
   - 将特征张量还原为 profile 字段级别（利用映射表）
   - 仅保留 bio/推文描述以外的特征

4. SHAP 解释
   - 对每个目标用户用 SHAP 解释
   - 获取每个字段的重要性分数
   - 对多维特征（如 OneHot/TF-IDF）合并贡献

5. 结果输出
   - 输出每个字段的重要性分数
   - 输出排名前 N 的关键字段名称
   - 可选：生成可视化图表

依赖：
- shap
- torch/torch_geometric
- numpy
- feature_mapping.py

"""

import os
import torch
import numpy as np
from feature_mapping import FIELD_INDEX_MAPPING, FIELD_GROUPS
import shap

# =====================
# 1. 数据与模型加载
# =====================
def load_features_labels(data_dir):
    features_dict = {
        'num': torch.load(os.path.join(data_dir, 'num_properties_tensor.pt')),
        'cat': torch.load(os.path.join(data_dir, 'cat_properties_tensor.pt')),
        'rem': torch.load(os.path.join(data_dir, 'num_for_h.pt')),
    }
    labels = torch.load(os.path.join(data_dir, 'label.pt'))
    return features_dict, labels

def lime_shap_load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

def lime_shap_concat_features(features_dict):
    arrs = [features_dict['num'], features_dict['cat'], features_dict['rem']]
    arrs = [a.numpy() if isinstance(a, torch.Tensor) else a for a in arrs]
    features = np.concatenate(arrs, axis=1)
    return features

# =====================
# 2. 字段映射工具
# =====================
def lime_shap_get_field_slices():
    field_slices = {}
    offset_dict = {'num': 0, 'cat': 0, 'rem': 0}
    for group, fields in FIELD_GROUPS.items():
        for i, field in enumerate(fields):
            for name, start, end in FIELD_INDEX_MAPPING:
                if name == field:
                    field_slices[field] = (offset_dict[group] + start, offset_dict[group] + end, group)
                    offset_dict[group] += (end - start + 1)
                    break
    return field_slices

# =====================
# 3. 筛选目标用户
# =====================
def lime_shap_select_bot_users(model, features, labels, bot_label=1):
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float())
        if hasattr(logits, 'detach'):
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        else:
            preds = np.argmax(logits, axis=1)
    bot_indices = np.where(preds == bot_label)[0]
    bot_features = features[bot_indices]
    return bot_indices, bot_features

# =====================
# 4. 特征还原与字段级别准备
# =====================
def lime_shap_extract_field_features(features, field_slices):
    user_field_features = []
    for user_feat in features:
        field_dict = {}
        for field, (start, end, group) in field_slices.items():
            field_dict[field] = user_feat[start:end+1] if end > start else np.array([user_feat[start]])
        user_field_features.append(field_dict)
    return user_field_features

# =====================
# 5. SHAP解释
# =====================
def shap_explain_user(model, features, user_idx, field_slices):
    explainer = shap.KernelExplainer(lambda x: model(torch.from_numpy(x).float()).detach().cpu().numpy(), features)
    shap_values = explainer.shap_values(features[user_idx:user_idx+1], nsamples=100)
    shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    field_importance = {k: 0.0 for k in field_slices}
    for feat_idx, score in enumerate(shap_vals):
        for field, (start, end, _) in field_slices.items():
            if start <= feat_idx <= end:
                field_importance[field] += score
    return field_importance

# =====================
# 6. 结果保存
# =====================
def save_top_fields(shap_importance, top_n, out_dir='.'):  # 只保留SHAP
    shap_top = [k for k, v in sorted(shap_importance.items(), key=lambda x: -abs(x[1]))[:top_n]]
    shap_file = os.path.join(out_dir, f'shap_top{top_n}_fields.txt')
    with open(shap_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(shap_top))
    print(f'SHAP字段排名已保存: {shap_file}')

# =====================
# 7. 主流程入口
# =====================
def main(top_n=5, out_dir='.'):  # top_n为参数
    data_dir = '../detector/processed_data/generated_profiles/'
    model_path = '../detector/models_quick/best_model.pt'  # 路径需根据实际情况调整
    features_dict, labels = load_features_labels(data_dir)
    features = lime_shap_concat_features(features_dict)
    field_slices = lime_shap_get_field_slices()
    model = lime_shap_load_model(model_path)
    bot_indices, bot_features = lime_shap_select_bot_users(model, features, labels)
    print(f'判别为机器人用户数: {len(bot_indices)}')
    print('示例机器人用户特征 shape:', bot_features.shape)
    bot_field_features = lime_shap_extract_field_features(bot_features, field_slices)
    print('示例机器人用户字段级特征:', {k: v for k, v in bot_field_features[0].items()})
    # 只保留SHAP解释
    if len(bot_features) > 0:
        shap_importance = shap_explain_user(model, bot_features, 0, field_slices)
        print(f'SHAP字段排名（前{top_n}）:')
        print([k for k, v in sorted(shap_importance.items(), key=lambda x: -abs(x[1]))[:top_n]])
        save_top_fields(shap_importance, top_n, out_dir)

if __name__ == '__main__':
    main() 