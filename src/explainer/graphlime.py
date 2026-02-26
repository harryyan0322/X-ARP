import torch
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Lasso
import networkx as nx
import os

# =====================
# 1. N-hop子图采样
# =====================
def get_n_hop_subgraph(X, edge_index, node_idx, hop=2):
    """
    采样N-hop邻居子图特征
    X: [N, F] 节点特征
    edge_index: [2, E] 边索引
    node_idx: 目标节点索引
    hop: 邻居阶数
    返回: 子图特征, 子图节点索引
    """
    G = nx.Graph()
    G.add_edges_from(edge_index.T.cpu().numpy())
    neighbors = set([node_idx])
    for _ in range(hop):
        neighbors |= set(j for i in neighbors for j in G.neighbors(i))
    neighbors = list(neighbors)
    return X[neighbors], neighbors

# =====================
# 2. HSIC核Lasso特征归因
# =====================
def hsic_kernel_lasso(X_sub, y_sub, alpha=0.1):
    """
    简化版 HSIC Lasso：核化特征+L1回归
    X_sub: [n, f] 子图特征
    y_sub: [n] 子图目标（如概率）
    返回: 每个特征的归因系数
    """
    K = rbf_kernel(X_sub)
    features = [rbf_kernel(X_sub[:, i:i+1]) for i in range(X_sub.shape[1])]
    X_kernel = np.stack([f.flatten() for f in features], axis=1)
    y_kernel = K.flatten()
    model = Lasso(alpha=alpha)
    model.fit(X_kernel, y_kernel)
    return model.coef_

# =====================
# 3. GraphLIME解释主流程
# =====================
def graphlime_explain_node(model, X, edge_index, node_idx, hop=2, alpha=0.1, top_n=5, out_dir=None):
    """
    对单节点进行GraphLIME解释，输出top_n重要特征索引
    model: GNN模型，输入(X, edge_index)输出[N, C]概率
    X: [N, F] 节点特征 (torch.Tensor or np.ndarray)
    edge_index: [2, E] torch.LongTensor
    node_idx: 目标节点索引
    hop: 邻居阶数
    alpha: Lasso正则
    top_n: 输出前N重要特征
    out_dir: 可选，保存结果目录
    """
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X
    X_sub, idx_sub = get_n_hop_subgraph(X_np, edge_index, node_idx, hop)
    # 获取子图节点的GNN输出（如机器人概率）
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_np).float(), edge_index).detach().cpu().numpy()
    # 假设二分类，取机器人类别概率
    y_sub = y_pred[idx_sub][:, 1] if y_pred.shape[1] > 1 else y_pred[idx_sub][:, 0]
    coeffs = hsic_kernel_lasso(X_sub, y_sub, alpha=alpha)
    top_features = np.argsort(np.abs(coeffs))[::-1][:top_n]
    print(f"GraphLIME选出的Top{top_n}特征索引：", top_features)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'graphlime_node{node_idx}_top{top_n}_features.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(str, top_features)))
        print(f"已保存至: {out_path}")
    return top_features, coeffs

# =====================
# 4. 主流程入口
# =====================
def main(node_idx=0, top_n=5, hop=2, alpha=0.1, out_dir=None):
    """
    示例主流程，需根据实际数据和模型路径调整
    """
    # 假设数据加载部分
    # X = ... # [N, F] torch.Tensor
    # edge_index = ... # [2, E] torch.LongTensor
    # model = ... # 已加载的GNN模型
    # node_idx = ... # 目标节点索引
    # 下面为伪代码/接口示例
    print("请在main函数中补充实际的数据和模型加载代码！")
    # top_features, coeffs = graphlime_explain_node(model, X, edge_index, node_idx, hop, alpha, top_n, out_dir)
    # print("特征归因系数:", coeffs)

if __name__ == '__main__':
    main()