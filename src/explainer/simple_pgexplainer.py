import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class PGExplainerNet(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, edge_feats):
        x = F.relu(self.fc1(edge_feats))
        return torch.sigmoid(self.fc2(x))

class PGExplainerWrapper:
    def __init__(self, gnn_model, emb_dim, device='cpu'):
        self.gnn_model = gnn_model.to(device)
        self.explainer_net = PGExplainerNet(emb_dim).to(device)
        self.device = device

    def get_edge_features(self, node_idx, x, edge_index, k=2):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx, k, edge_index, relabel_nodes=True)
        x_sub = x[subset]
        edge_index_sub = sub_edge_index

        with torch.no_grad():
            embeddings = self.gnn_model(x, edge_index)

        edge_feats = []
        for i in range(edge_index_sub.size(1)):
            src, dst = edge_index_sub[:, i]
            emb_src = embeddings[subset[src]]
            emb_dst = embeddings[subset[dst]]
            edge_feat = torch.cat([emb_src, emb_dst], dim=0)
            edge_feats.append(edge_feat)

        return torch.stack(edge_feats), edge_index_sub, subset, mapping

    def train_explainer(self, x, edge_index, labels, node_indices, epochs=20, lr=0.01):
        optimizer = torch.optim.Adam(self.explainer_net.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.gnn_model.eval()
        for epoch in range(epochs):
            total_loss = 0
            for node_idx in node_indices:
                edge_feats, edge_idx_sub, _, _ = self.get_edge_features(
                    node_idx, x, edge_index)
                pred_logits = self.explainer_net(edge_feats).squeeze()
                pred_labels = torch.ones_like(pred_logits)  # all 1 (positive importance)
                loss = loss_fn(pred_logits, pred_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"[PGExplainer] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def explain_node(self, node_idx, x, edge_index, topk=10):
        self.gnn_model.eval()
        self.explainer_net.eval()
        edge_feats, edge_idx_sub, subset, mapping = self.get_edge_features(node_idx, x, edge_index)
        edge_mask = self.explainer_net(edge_feats).detach().squeeze()
        topk_idx = edge_mask.topk(min(topk, edge_mask.size(0))).indices
        important_edges = edge_idx_sub[:, topk_idx]
        return important_edges, edge_mask

# 训练PGExplainer

def train_pgexplainer(model, x, edge_index, labels, device='cpu'):
    node_indices = torch.where(labels == 1)[0][:100]  # 选100个机器人节点
    pgexplainer = PGExplainerWrapper(gnn_model=model, emb_dim=model.hidden_channels, device=device)
    pgexplainer.train_explainer(x, edge_index, labels, node_indices)
    return pgexplainer

# 解释节点邻居

def explain_node_neighbors(pgexplainer, node_idx, x, edge_index, top_n=5):
    important_edges, edge_mask = pgexplainer.explain_node(node_idx, x, edge_index, topk=top_n)
    neighbors = set()
    for i in range(important_edges.size(1)):
        src, dst = important_edges[:, i]
        if src == node_idx:
            neighbors.add(int(dst))
        elif dst == node_idx:
            neighbors.add(int(src))
    return list(neighbors), edge_mask 