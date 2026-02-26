import torch
import torch.nn.functional as F
import shap
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, to_networkx
from feature_mapping import NUMERICAL_FIELDS, CATEGORICAL_FIELDS, REMAINING_FIELDS
from PGExplainer import train_pgexplainer, explain_node_neighbors


def explain_node(model, data, node_idx, num_classes=2, hop=2, top_k=5, alpha=0.1, device='cpu'):
    model.eval()
    model = model.to(device)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    node_feat = x[node_idx].unsqueeze(0)

    # ---------- Part 1: SHAP ----------
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, node_idx, edge_index):
            super().__init__()
            self.model = model
            self.node_idx = node_idx
            self.edge_index = edge_index

        def forward(self, x_input):
            out = self.model(x_input, self.edge_index)
            return out[self.node_idx]

    shap_model = WrappedModel(model, node_idx, edge_index)
    explainer = shap.Explainer(shap_model, x, algorithm="permutation")
    shap_values = explainer(node_feat)
    shap_scores = shap_values.values[0]
    shap_top_idx = np.argsort(np.abs(shap_scores))[::-1][:top_k]

    # ---------- Part 2: GraphLIME ----------
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, hop, edge_index, relabel_nodes=True)
    x_sub = x[subset]
    y_sub = model(x, edge_index).detach()[subset][:, 1]  # bot class prob

    def hsic_lasso(X_sub, y_sub, alpha):
        K = rbf_kernel(X_sub.cpu().numpy())
        features = [rbf_kernel(X_sub[:, i:i+1].cpu().numpy()) for i in range(X_sub.shape[1])]
        X_kernel = np.stack([f.flatten() for f in features], axis=1)
        y_kernel = K.flatten()
        model = Lasso(alpha=alpha)
        model.fit(X_kernel, y_kernel)
        return model.coef_

    hsic_weights = hsic_lasso(x_sub, y_sub, alpha)
    graphlime_top_idx = np.argsort(np.abs(hsic_weights))[::-1][:top_k]

    # ---------- Part 3: Structure Explanation (Pseudo PGExplainer) ----------
    edge_mask = torch.rand(sub_edge_index.size(1)).to(device)
    threshold = edge_mask.quantile(0.8)
    important_edges = sub_edge_index[:, edge_mask > threshold]

    # ---------- Output ----------
    important_edge_list = [(int(u), int(v)) for u, v in important_edges.t().tolist()]
    return {
        "shap_top_features": shap_top_idx.tolist(),
        "shap_scores": shap_scores[shap_top_idx].tolist(),
        "graphlime_top_features": graphlime_top_idx.tolist(),
        "graphlime_scores": hsic_weights[graphlime_top_idx].tolist(),
        "important_edges": important_edge_list
    }


def get_feature_names():
    return NUMERICAL_FIELDS + CATEGORICAL_FIELDS + REMAINING_FIELDS


def explain_robot_node(model, data, node_idx, node_name, labels, pgexplainer, hop=2, top_k=5, alpha=0.1, device='cpu'):
    feature_names = get_feature_names()
    # 1. SHAP + GraphLIME
    shap_result = explain_node(model, data, node_idx, num_classes=2, hop=hop, top_k=top_k, alpha=alpha, device=device)
    shap_top_features = shap_result['shap_top_features']
    shap_scores = shap_result['shap_scores']
    graphlime_top_features = shap_result['graphlime_top_features']
    graphlime_scores = shap_result['graphlime_scores']

    # 2. PGExplainer structure explanation
    neighbors, edge_mask = explain_node_neighbors(pgexplainer, node_idx, data.x, data.edge_index, top_n=top_k)
    label_dict = {int(i): int(l) for i, l in enumerate(labels)}
    robot_neighbors = [n for n in neighbors if label_dict.get(n, 0) == 1]

    # 3. Generate English explanation
    desc = []
    desc.append(f"User @{node_name} is classified as a bot mainly because:")

    # Structure explanation
    if len(robot_neighbors) > 0:
        desc.append(f"Among the {len(neighbors)} accounts followed, {len(robot_neighbors)} are also classified as bots;")
    else:
        desc.append(f"Structurally, this node is connected to {len(neighbors)} high-weight neighbors, which is considered an important factor by the model.")

    # Node feature importance
    shap_feature_desc = []
    for idx, score in zip(shap_top_features, shap_scores):
        if idx < len(feature_names):
            shap_feature_desc.append(f"Feature '{feature_names[idx]}' (importance {score:.2f})")
    if shap_feature_desc:
        desc.append("The model identifies the following node features as most influential:")
        desc.append(", ".join(shap_feature_desc) + ".")

    # Neighborhood feature importance
    graphlime_feature_desc = []
    for idx, score in zip(graphlime_top_features, graphlime_scores):
        if idx < len(feature_names):
            graphlime_feature_desc.append(f"'{feature_names[idx]}' (contribution {score:.2f})")
    if graphlime_feature_desc:
        desc.append("In the local neighborhood, the following features also contribute significantly to the prediction:")
        desc.append(", ".join(graphlime_feature_desc) + ".")

    # Structure summary
    desc.append(f"Structurally, this node is connected to {len(neighbors)} high-weight neighbors, which is considered an important factor by the model.")

    # Merge and output
    final_text = "\n".join(desc)
    print(final_text)
    return final_text

