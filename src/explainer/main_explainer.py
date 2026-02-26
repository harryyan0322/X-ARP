"""
Social network bot detection model explainability analysis main script
==========================================
Function:
- For users predicted as "bot", use GraphLIME, SHAP, and PGExplainer to perform explainability analysis
- Output the importance score for each field and the names of the top fields
- Generate an explainability report

Main process:
1. Load the trained model and data
2. Filter users predicted as bots
3. Use GraphLIME, SHAP, and PGExplainer for explainability analysis
4. Output feature importance rankings and visualization results

Dependencies:
- torch, torch_geometric
- shap, sklearn, networkx
- numpy, pandas, matplotlib
- Trained model file
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx
import shap
from torch_geometric.utils import k_hop_subgraph

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # social_network/
sys.path.insert(0, project_root)

# Add detector path to import the model
detector_path = os.path.join(project_root, 'src', 'detector')
sys.path.insert(0, detector_path)

# Add explainer path to import feature_mapping
explainer_path = os.path.join(project_root, 'src', 'explainer')
sys.path.insert(0, explainer_path)


from ThreeInONE import AllInOne1_rgcn_rgt_gcn
from feature_mapping import get_field_slices, get_feature_dimensions, FIELD_DESCRIPTIONS





class BotExplainer:
    def __init__(self, model_path, data_dir, device='cpu'):
        """
        Initialize the explainability analyzer for the social network bot detection model.
        
        Args:
            model_path: Path to the trained model
            data_dir: Path to the data directory
            device: Computation device
        """
        self.device = device
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model
        self.model = self._load_model()
        
        # Load data
        self.data = self._load_data()
        
        # Get feature mapping
        self.field_slices = get_field_slices()
        self.feature_dims = get_feature_dimensions()
        
        # Filter bot users
        self.bot_users = self._identify_bot_users()

        # Prepare feature matrix and group statistics for V_c
        self.feature_matrix = torch.cat([
            self.data.num_property_embedding,
            self.data.cat_property_embedding,
            self.data.num_for_h
        ], dim=1).detach().cpu()
        self.feature_stats = self._compute_group_stats(self.feature_matrix, self.data.y.detach().cpu())

        # Build graph for V_s (subgraph-level evidence)
        try:
            edge_index = self.data.edge_index.detach().cpu().numpy()
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(self.data.num_nodes))
            self.graph.add_edges_from(zip(edge_index[0], edge_index[1]))
        except Exception:
            self.graph = None
        
        # print(f"✅ 初始化完成:")
        # print(f"   - 模型: {model_path}")
        # print(f"   - 数据: {data_dir}")
        # print(f"   - 总用户数: {self.data.y.shape[0]}")
        # print(f"   - 机器人用户数: {len(self.bot_users)}")
        # print(f"   - 特征维度: {self.feature_dims['total']}")
        print(f"✅ Initialization complete - Model: {os.path.basename(model_path)}, Data: {os.path.basename(data_dir)}")
        print(f"   Total users: {self.data.y.shape[0]}, Bots: {len(self.bot_users)}, Feature dim: {self.feature_dims['total']}")

    def _compute_group_stats(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Compute per-feature mean/std for human vs bot groups.
        """
        eps = 1e-6
        human_mask = labels == 0
        bot_mask = labels == 1
        human = features[human_mask]
        bot = features[bot_mask]
        # Fallback if a group is empty
        if human.numel() == 0:
            human = features
        if bot.numel() == 0:
            bot = features
        stats = {
            "human_mean": human.mean(dim=0),
            "human_std": human.std(dim=0) + eps,
            "bot_mean": bot.mean(dim=0),
            "bot_std": bot.std(dim=0) + eps,
        }
        return stats

    def _compute_contrastive_scores(self, user_idx: int):
        """
        Compute V_c contrastive scores per feature:
        delta_h = |x - mu_h| / (sigma_h + eps)
        delta_b = |x - mu_b| / (sigma_b + eps)
        Delta = delta_h - delta_b
        """
        x_u = self.feature_matrix[user_idx]
        mu_h = self.feature_stats["human_mean"]
        mu_b = self.feature_stats["bot_mean"]
        std_h = self.feature_stats["human_std"]
        std_b = self.feature_stats["bot_std"]
        delta_h = torch.abs(x_u - mu_h) / std_h
        delta_b = torch.abs(x_u - mu_b) / std_b
        delta = (delta_h - delta_b).tolist()
        return delta

    def _compute_subgraph_metrics(self, user_idx: int, k_hops: int = 2):
        """
        Approximate V_s using a k-hop induced subgraph.
        Returns clustering coefficient, hub score (degree centrality), and density.
        """
        if self.graph is None or user_idx not in self.graph:
            return {
                "clustering": 0.0,
                "hub_score": 0.0,
                "density": 0.0,
                "subgraph_size": 0,
                "bot_ratio": 0.0,
            }
        nodes = nx.single_source_shortest_path_length(self.graph, user_idx, cutoff=k_hops).keys()
        subgraph = self.graph.subgraph(nodes).copy()
        if subgraph.number_of_nodes() == 0:
            return {
                "clustering": 0.0,
                "hub_score": 0.0,
                "density": 0.0,
                "subgraph_size": 0,
                "bot_ratio": 0.0,
            }

        clustering = float(nx.clustering(subgraph, user_idx))
        degree_centrality = nx.degree_centrality(subgraph)
        hub_score = float(degree_centrality.get(user_idx, 0.0))
        density = float(nx.density(subgraph)) if subgraph.number_of_nodes() > 1 else 0.0

        # Bot ratio inside subgraph (excluding the target node)
        labels = self.data.y.detach().cpu().numpy()
        neighbors = [n for n in subgraph.nodes if n != user_idx]
        if neighbors:
            bot_ratio = float((labels[neighbors] == 1).mean())
        else:
            bot_ratio = 0.0

        return {
            "clustering": clustering,
            "hub_score": hub_score,
            "density": density,
            "subgraph_size": int(subgraph.number_of_nodes()),
            "bot_ratio": bot_ratio,
        }
    
    def _load_model(self):
        """Load the trained model"""
        # print("🔄 加载模型...")
        
        # Create model instance (with parameters same as during training)
        model = AllInOne1_rgcn_rgt_gcn(
            output_size=2,  # Changed to output_size
            num_gnn=3,
            num_text=2,
            gnn_k=1,
            align_size=64
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # print(f"✅ 模型加载成功: {self.model_path}")
        return model
    
    def _load_data(self):
        """Load feature data and labels"""
        # print("🔄 加载数据...")
        
        from torch_geometric.data import Data
        
        # Load all data files
        data = Data(
            des_embedding=torch.load(os.path.join(self.data_dir, 'des_tensor.pt')),
            tweet_embedding=torch.load(os.path.join(self.data_dir, 'tweets_tensor.pt')),
            num_property_embedding=torch.load(os.path.join(self.data_dir, 'num_properties_tensor.pt')),
            cat_property_embedding=torch.load(os.path.join(self.data_dir, 'cat_properties_tensor.pt')),
            num_for_h=torch.load(os.path.join(self.data_dir, 'num_for_h.pt')),
            edge_index=torch.load(os.path.join(self.data_dir, 'edge_index.pt')),
            edge_type=torch.load(os.path.join(self.data_dir, 'edge_type.pt')),
            y=torch.load(os.path.join(self.data_dir, 'label.pt')),
            num_nodes=torch.load(os.path.join(self.data_dir, 'des_tensor.pt')).shape[0]
        )
        
        # print(f"✅ 数据加载成功:")
        # print(f"   - 描述嵌入: {data.des_embedding.shape}")
        # print(f"   - 推文嵌入: {data.tweet_embedding.shape}")
        # print(f"   - 数值特征: {data.num_property_embedding.shape}")
        # print(f"   - 类别特征: {data.cat_property_embedding.shape}")
        # print(f"   - 其他特征: {data.num_for_h.shape}")
        # print(f"   - 标签: {data.y.shape}")
        # print(f"   - 边索引: {data.edge_index.shape}")
        # print(f"   - 边类型: {data.edge_type.shape}")
        
        return data
    
    def _identify_bot_users(self):
        """Identify users who are both truly bots and predicted as bots by the model"""
        # print("🔄 识别机器人用户...")
        
        # Perform prediction
        with torch.no_grad():
            self.data = self.data.to(self.device)
            logits, _ = self.model(
                self.data.des_embedding,
                self.data.tweet_embedding,
                self.data.num_property_embedding,
                self.data.cat_property_embedding,
                self.data.num_for_h,
                self.data.edge_index,
                self.data.edge_type
            )
            pred_labels = torch.argmax(logits, dim=-1)
            # Only keep users with true label 1 and predicted label 1
            mask = (pred_labels == 1) & (self.data.y == 1)
            bot_indices = torch.where(mask)[0].cpu().numpy()
        return bot_indices
    
    def _create_single_user_data(self, user_idx):
        """Create a data object for a single user"""
        from torch_geometric.data import Data
        
        return Data(
            des_embedding=self.data.des_embedding[user_idx:user_idx+1],
            tweet_embedding=self.data.tweet_embedding[user_idx:user_idx+1],
            num_property_embedding=self.data.num_property_embedding[user_idx:user_idx+1],
            cat_property_embedding=self.data.cat_property_embedding[user_idx:user_idx+1],
            num_for_h=self.data.num_for_h[user_idx:user_idx+1],
            edge_index=self.data.edge_index,
            edge_type=self.data.edge_type,
            y=self.data.y[user_idx:user_idx+1],
            num_nodes=self.data.num_nodes
        )
    
    def explain_with_shap(self, user_idx, top_n=10):
        """
        Perform global feature importance analysis using SHAP.
        
        Principle of SHAP (SHapley Additive exPlanations):
        1. Calculate the contribution of each feature to the prediction based on Shapley values from game theory
        2. Estimate feature importance by permuting feature values
        3. Provide both global and local interpretability
        
        Global analysis process:
        1. Prepare feature data for all users
        2. Create a wrapper model that can handle arbitrary user features
        3. Use KernelExplainer to compute global SHAP values
        4. Analyze the global contribution of all features to bot classification
        """
        # print(f"🔄 使用SHAP进行全局特征重要性分析...")
        
        # Step 1: Prepare feature data for all users
        x = torch.cat([
            self.data.num_property_embedding,
            self.data.cat_property_embedding,
            self.data.num_for_h
        ], dim=1)
        
        # Step 2: Create a global wrapper model
        class GlobalWrappedModel:
            def __init__(self, model, data, device):
                self.model = model
                self.data = data
                self.device = device
            
            def __call__(self, x_input):
                # Handle batch input - SHAP will pass multiple samples
                batch_size = x_input.shape[0]
                results = []
                
                for i in range(batch_size):
                    # Convert SHAP input to user features
                    num_dim = self.data.num_property_embedding.shape[1]
                    cat_dim = self.data.cat_property_embedding.shape[1]
                    
                    user_features = x_input[i]  # Take the i-th sample
                    num_props = user_features[:num_dim]
                    cat_props = user_features[num_dim:num_dim+cat_dim]
                    other_props = user_features[num_dim+cat_dim:]
                    
                    # Create data object - use the full graph structure
                    from torch_geometric.data import Data
                    user_data = Data(
                        des_embedding=self.data.des_embedding,
                        tweet_embedding=self.data.tweet_embedding,
                        num_property_embedding=self.data.num_property_embedding.clone(),
                        cat_property_embedding=self.data.cat_property_embedding.clone(),
                        num_for_h=self.data.num_for_h.clone(),
                        edge_index=self.data.edge_index,
                        edge_type=self.data.edge_type,
                        y=self.data.y,
                        num_nodes=self.data.num_nodes
                    )
                    
                    # Modify features of all users to the features of the current SHAP sample
                    # This is the key for global analysis: we want to know the impact of this feature combination on all users
                    user_data.num_property_embedding[:] = torch.tensor(num_props, dtype=torch.float32)
                    user_data.cat_property_embedding[:] = torch.tensor(cat_props, dtype=torch.float32)
                    user_data.num_for_h[:] = torch.tensor(other_props, dtype=torch.float32)
                    
                    with torch.no_grad():
                        user_data = user_data.to(self.device)
                        logits, _ = self.model(
                            user_data.des_embedding,
                            user_data.tweet_embedding,
                            user_data.num_property_embedding,
                            user_data.cat_property_embedding,
                            user_data.num_for_h,
                            user_data.edge_index,
                            user_data.edge_type
                        )
                        probs = torch.softmax(logits, dim=-1)
                    
                    # Calculate global bot probability (average bot probability across all users)
                    global_bot_prob = torch.mean(probs[:, 1]).cpu().numpy()
                    results.append([1 - global_bot_prob, global_bot_prob])  # [Human probability, Bot probability]
                
                # Return batch results
                return np.array(results)
        
        # Step 3: Prepare global background data - use representative samples with stratified sampling
        # Get indices of human and bot users
        human_indices = torch.where(self.data.y == 0)[0]  # Human users
        bot_indices = torch.where(self.data.y == 1)[0]    # Bot users
        
        # Stratified sampling: half human, half bot
        n_human = min(100, len(human_indices))  # 100 human samples
        n_bot = min(100, len(bot_indices))      # 100 bot samples
        
        # Randomly select samples
        human_sample_indices = human_indices[torch.randperm(len(human_indices))[:n_human]]
        bot_sample_indices = bot_indices[torch.randperm(len(bot_indices))[:n_bot]]
        
        # Combine sample indices
        background_indices = torch.cat([human_sample_indices, bot_sample_indices])
        background_data = x[background_indices].cpu().numpy()
        
        # Use kmeans clustering to reduce the number of background samples and improve calculation efficiency
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(50, len(background_data))  # Max 50 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            background_data = kmeans.fit(background_data).cluster_centers_
            # print(f"📊 使用K-means聚类将背景样本从 {len(background_indices)} 减少到 {n_clusters} 个")
        except ImportError:
            # print(f"📊 使用 {len(background_indices)} 个背景样本进行全局分析")
            # print(f"   - 人类样本: {n_human} 个")
            # print(f"   - 机器人样本: {n_bot} 个")
            pass
        
        # Create global wrapper model
        wrapped_model = GlobalWrappedModel(self.model, self.data, self.device)
        
        # Step 4: Use SHAP for global explanation
        # print("🔄 初始化SHAP解释器...")
        explainer = shap.KernelExplainer(
            wrapped_model,
            background_data
        )
        
        # Step 5: Get representative user features and perform global SHAP explanation
        # Use the features of the target user as the analysis sample
        user_features = x[user_idx:user_idx+1].cpu().numpy()
        # print("�� 开始SHAP值计算（这可能需要几分钟，请耐心等待）...")
        # print("📊 正在分析特征对全局机器人判别的贡献...")
        shap_values = explainer.shap_values(user_features, nsamples=50)  # Reduce sampling number
        # print("✅ SHAP值计算完成！")
        
        # Step 6: Extract global feature importance (SHAP values for bot class)
        bot_shap_values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # Step 7: Create a dictionary of global feature importance
        feature_names = list(self.field_slices.keys())
        importance = {}
        
        for i, importance_val in enumerate(bot_shap_values):
            if i < len(feature_names):
                # Ensure importance_val is a scalar
                if hasattr(importance_val, '__len__') and len(importance_val) > 0:
                    importance[feature_names[i]] = float(importance_val[0])
                else:
                    importance[feature_names[i]] = float(importance_val)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        importance = dict(sorted_features[:top_n])
        
        # print(f"✅ 全局SHAP分析完成，找到 {len(importance)} 个重要特征")
        # print("🎯 这些特征对机器人判别具有全局重要性")
        return importance
    
    def explain_with_graphlime(self, user_idx, top_n=10):
        """
        Explain a single user using GraphLIME.
        
        Principle of GraphLIME (Graph Local Interpretable Model-agnostic Explanations):
        1. Sample local subgraphs based on k-hop neighborhoods
        2. Use HSIC (Hilbert-Schmidt Independence Criterion) Lasso for feature attribution
        3. Provide local interpretability considering graph structure
        
        Implementation process:
        1. Obtain the N-hop subgraph (default 2-hop neighbors)
        2. Compute prediction probabilities for subgraph nodes
        3. Use HSIC Lasso to compute feature importance
        4. Return the most important features in the local neighborhood
        """
        # print(f"🔄 使用GraphLIME解释用户 {user_idx}...")
        
        # Step 1: Prepare feature data
        x = torch.cat([
            self.data.num_property_embedding,
            self.data.cat_property_embedding,
            self.data.num_for_h
        ], dim=1)
        
        # Step 2: Obtain the N-hop subgraph - sample 2-hop neighbors of the target node
        try:
            # Ensure user_idx is a tensor
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                user_idx_tensor, 2, self.data.edge_index, relabel_nodes=True)
            
            # Check if subgraph is empty
            if len(subset) == 0:
                # print(f"⚠️ 用户 {user_idx} 没有邻居，跳过GraphLIME分析")
                return {}
            
            x_sub = x[subset]
            
            # Step 3: Get prediction probabilities for subgraph nodes - use the full graph instead of the subgraph
            with torch.no_grad():
                # Predict using the full graph, then extract results for subgraph nodes
                y_pred_full = self.model(
                    self.data.des_embedding,
                    self.data.tweet_embedding,
                    self.data.num_property_embedding,
                    self.data.cat_property_embedding,
                    self.data.num_for_h,
                    self.data.edge_index,
                    self.data.edge_type
                )[0].detach().cpu().numpy()
                
                # Extract prediction results for subgraph nodes
                y_sub = y_pred_full[subset.cpu().numpy()]
            
            # Assuming binary classification, take bot class probability
            y_sub = y_sub[:, 1] if y_sub.shape[1] > 1 else y_sub[:, 0]
            
            # Step 4: HSIC Lasso feature attribution - core algorithm
            def hsic_lasso(X_sub, y_sub, alpha=0.1):
                """
                HSIC Lasso algorithm implementation:
                1. Calculate kernel matrix K (based on subgraph features)
                2. Calculate separate kernel matrices for each feature
                3. Learn feature importance weights using Lasso regression
                """
                K = rbf_kernel(X_sub.cpu().numpy())  # Calculate kernel matrix
                features = [rbf_kernel(X_sub[:, i:i+1].cpu().numpy()) for i in range(X_sub.shape[1])]  # Kernel matrix for each feature
                X_kernel = np.stack([f.flatten() for f in features], axis=1)  # Stack feature kernel matrices
                y_kernel = K.flatten()  # Target kernel matrix
                
                # Fix convergence issues: increase iterations, adjust regularization parameter
                model = Lasso(
                    alpha=alpha,
                    max_iter=5000,  # Further increase max iterations
                    tol=1e-3,       # Relax tolerance
                    random_state=42, # Set random seed
                    warm_start=True  # Enable warm start
                )
                
                # Feature standardization
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_kernel_scaled = scaler.fit_transform(X_kernel)
                
                model.fit(X_kernel_scaled, y_kernel)  # Fit model
                return model.coef_  # Return feature weights
            
            hsic_weights = hsic_lasso(x_sub, y_sub, alpha=0.01)  # Further reduce regularization strength
            
            # Step 5: Create feature importance dictionary
            feature_names = list(self.field_slices.keys())
            importance = {}
            for i, weight in enumerate(hsic_weights):
                if i < len(feature_names):
                    importance[feature_names[i]] = float(weight)
            
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            importance = dict(sorted_features[:top_n])
            
            # print(f"✅ GraphLIME解释完成，找到 {len(importance)} 个重要特征")
            return importance
            
        except Exception as e:
            # print(f"❌ GraphLIME解释失败: {e}")
            return {}
    
    def explain_with_pgexplainer(self, user_idx, top_n=10):
        """
        Explain a single user using PGExplainer.
        
        Principle of PGExplainer (Parameterized Graph Explainer):
        1. Identify the most influential 1-hop neighbors for the target node's bot prediction
        2. Quantify neighbor importance by edge removal experiments
        3. Provide structural interpretability
        
        Implementation process:
        1. Obtain the 1-hop neighbors of the target node
        2. Evaluate the impact of each neighbor by removing edges
        3. Calculate the impact of edge removal on bot prediction
        4. Return the most important neighbors and edge information
        """
        # print(f"🔄 使用PGExplainer解释用户 {user_idx}...")
        
        # Step 1: Obtain 1-hop neighbors
        try:
            # Ensure user_idx is a tensor
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                user_idx_tensor, 1, self.data.edge_index, relabel_nodes=True)  # Only get 1-hop neighbors
            
            # Check if subgraph is empty
            if len(subset) == 0:
                # print(f"⚠️ 用户 {user_idx} 没有邻居，跳过PGExplainer分析")
                return {
                    'total_important_neighbors': 0,
                    'bot_neighbors': [],
                    'human_neighbors': [],
                    'bot_ratio': 0.0,
                    'neighbor_importance': {},
                    'edge_importance_scores': [],
                    'most_influential_neighbors': [],
                    'explanation': "This node has no neighbors."
                }
            
            # Step 2: Calculate original prediction probability
            with torch.no_grad():
                original_logits, _ = self.model(
                    self.data.des_embedding,
                    self.data.tweet_embedding,
                    self.data.num_property_embedding,
                    self.data.cat_property_embedding,
                    self.data.num_for_h,
                    self.data.edge_index,
                    self.data.edge_type
                )
                original_probs = torch.softmax(original_logits, dim=-1)
                original_bot_prob = original_probs[user_idx, 1].item()  # Original bot probability
            
            # Step 3: Calculate the impact of each neighbor (by removing edges)
            neighbor_importance = {}
            neighbor_relations = {}  # Store relationship between neighbor and target node
            bot_neighbors = []
            human_neighbors = []
            
            subset_numpy = subset.cpu().numpy()
            neighbor_labels = self.data.y[subset].cpu().numpy()
            
            # Find the edge connecting the target node and the current neighbor
            # Fix mapping access issue
            if isinstance(mapping, torch.Tensor):
                # If mapping is a tensor, find the position of user_idx in subset
                user_idx_in_subset = torch.where(subset == user_idx)[0]
                if len(user_idx_in_subset) > 0:
                    target_in_subset = user_idx_in_subset[0].item()
                else:
                    print(f"⚠️ User {user_idx} not in subgraph")
                    target_in_subset = 0
            else:
                # If mapping is a dictionary, access directly
                target_in_subset = mapping[user_idx]
            
            for i, neighbor_idx in enumerate(subset_numpy):
                if neighbor_idx != user_idx:  # Exclude self
                    # Find the edge connecting the target node and the current neighbor
                    edge_mask = ((sub_edge_index[0] == target_in_subset) & (sub_edge_index[1] == i)) | \
                               ((sub_edge_index[0] == i) & (sub_edge_index[1] == target_in_subset))
                    
                    if edge_mask.any():
                        # Create a graph with the edge removed
                        edge_to_remove = sub_edge_index[:, edge_mask]
                        
                        # Map subgraph edge indices back to original graph
                        original_edge_indices = []
                        for j in range(edge_to_remove.size(1)):
                            src_original = subset[edge_to_remove[0, j]]
                            dst_original = subset[edge_to_remove[1, j]]
                            
                            # Find the corresponding edge in the original graph
                            edge_mask_original = ((self.data.edge_index[0] == src_original) & 
                                                (self.data.edge_index[1] == dst_original)) | \
                                               ((self.data.edge_index[0] == dst_original) & 
                                                (self.data.edge_index[1] == src_original))
                            original_edge_indices.extend(edge_mask_original.nonzero(as_tuple=True)[0].tolist())
                        
                        # Predict after removing the edge
                        if original_edge_indices:
                            try:
                                # Create edge indices after removing the edge
                                edge_mask_keep = torch.ones(self.data.edge_index.size(1), dtype=torch.bool, device=self.device)
                                edge_mask_keep[original_edge_indices] = False
                                
                                # Ensure at least some edges are kept to avoid empty graph
                                if edge_mask_keep.sum() < 10:  # If too few edges, use feature similarity
                                    raise ValueError("Graph too small, edges removed")
                                
                                edge_index_removed = self.data.edge_index[:, edge_mask_keep]
                                edge_type_removed = self.data.edge_type[edge_mask_keep]
                                
                                # Validate edge indices
                                if edge_index_removed.size(1) == 0:
                                    raise ValueError("No edges remaining after removing")
                                
                                if edge_index_removed.max() >= self.data.num_nodes:
                                    raise ValueError("Edge index out of bounds")
                                
                                # Predict using the graph after removing the edge
                                with torch.no_grad():
                                    removed_logits, _ = self.model(
                                        self.data.des_embedding,
                                        self.data.tweet_embedding,
                                        self.data.num_property_embedding,
                                        self.data.cat_property_embedding,
                                        self.data.num_for_h,
                                        edge_index_removed,
                                        edge_type_removed
                                    )
                                    removed_probs = torch.softmax(removed_logits, dim=-1)
                                    removed_bot_prob = removed_probs[user_idx, 1].item()
                                
                                # Calculate influence: change in bot probability after removing the edge
                                influence = original_bot_prob - removed_bot_prob
                                neighbor_importance[int(neighbor_idx)] = influence
                                
                                # Determine the relationship between the neighbor and the target node (directed graph)
                                # Check if there is an outgoing edge from the target node to the neighbor (outgoing edge)
                                out_edge = ((self.data.edge_index[0] == user_idx) & (self.data.edge_index[1] == neighbor_idx)).any()
                                # Check if there is an incoming edge from the neighbor to the target node (incoming edge)
                                in_edge = ((self.data.edge_index[0] == neighbor_idx) & (self.data.edge_index[1] == user_idx)).any()
                                
                                if out_edge and in_edge:
                                    relation = "bidirectional connection"
                                elif out_edge:
                                    relation = "target node follows neighbor"
                                elif in_edge:
                                    relation = "neighbor follows target node"
                                else:
                                    relation = "unknown relation"
                                
                                neighbor_relations[int(neighbor_idx)] = relation
                                
                                # Classify neighbor type
                                label = neighbor_labels[i]
                                if label == 1:  # Bot
                                    bot_neighbors.append(int(neighbor_idx))
                                else:  # Human
                                    human_neighbors.append(int(neighbor_idx))
                                    
                            except Exception as e:
                                # If edge removal fails, skip this neighbor
                                # print(f"⚠️ Edge removal failed, skipping neighbor {neighbor_idx}: {e}")
                                continue
                        else:
                            # If no corresponding edge is found, skip this neighbor
                            # print(f"⚠️ No edge found connecting to neighbor {neighbor_idx}, skipping")
                            continue
            
            # Step 4: Generate results
            total_neighbors = len(bot_neighbors) + len(human_neighbors)
            bot_ratio = len(bot_neighbors) / total_neighbors if total_neighbors > 0 else 0.0
            
            # Get the most important neighbors (sorted by influence)
            sorted_neighbors = sorted(neighbor_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            most_influential_neighbors = sorted_neighbors[:top_n]
            
            # Generate explanation - create neighbor label dictionary
            neighbor_labels_dict = {}
            for i, neighbor_idx in enumerate(subset_numpy):
                if neighbor_idx != user_idx:  # Exclude self
                    # Ensure i is within the range of neighbor_labels
                    if i < len(neighbor_labels):
                        neighbor_labels_dict[int(neighbor_idx)] = int(neighbor_labels[i])
                    else:
                        # print(f"⚠️ Index {i} out of bounds for neighbor_labels {len(neighbor_labels)}")
                        neighbor_labels_dict[int(neighbor_idx)] = 0  # Default value
            
            explanation = self._generate_edge_explanation(
                list(neighbor_importance.keys()),
                neighbor_importance,
                neighbor_labels_dict,
                bot_neighbors,
                neighbor_relations
            )
            
            result = {
                'total_important_neighbors': total_neighbors,
                'bot_neighbors': bot_neighbors,
                'human_neighbors': human_neighbors,
                'bot_ratio': bot_ratio,
                'neighbor_importance': neighbor_importance,
                'neighbor_relations': neighbor_relations,
                'most_influential_neighbors': most_influential_neighbors,
                'explanation': explanation
            }
            
            # print(f"✅ PGExplainer解释完成，找到 {total_neighbors} 个重要邻居")
            # print(f"   原始机器人概率: {original_bot_prob:.3f}")
            return result
            
        except Exception as e:
            # print(f"❌ PGExplainer解释失败: {e}")
            return {
                'total_important_neighbors': 0,
                'bot_neighbors': [],
                'human_neighbors': [],
                'bot_ratio': 0.0,
                'neighbor_importance': {},
                'neighbor_relations': {},
                'most_influential_neighbors': [],
                'explanation': f"Analysis failed: {str(e)}"
            }
    
    def _generate_edge_explanation(self, important_neighbors, neighbor_importance, neighbor_labels, bot_neighbors, neighbor_relations=None):
        """Generate edge importance explanation"""
        explanation = []
        
        if not important_neighbors:
            return "This node has no important neighbors."
        
        # Sort neighbors by importance
        sorted_neighbors = sorted(neighbor_importance.items(), key=lambda x: x[1], reverse=True)
        
        explanation.append(f"This node is primarily influenced by the following neighbors:")
        
        for i, (neighbor_id, importance) in enumerate(sorted_neighbors[:5], 1):
            neighbor_type = "bot" if neighbor_labels.get(neighbor_id, 0) == 1 else "human"
            relation = neighbor_relations.get(neighbor_id, "unknown relation") if neighbor_relations else "unknown relation"
            explanation.append(f"{i}. Neighbor {neighbor_id} (importance: {importance:.3f}, type: {neighbor_type}, relation: {relation})")
        
        if bot_neighbors:
            bot_importance = sum(neighbor_importance.get(n, 0) for n in bot_neighbors)
            total_importance = sum(neighbor_importance.values())
            bot_ratio = bot_importance / total_importance if total_importance > 0 else 0
            explanation.append(f"The total importance of bot neighbors is {bot_ratio:.3f}.")
            explanation.append(f"This indicates that this node is primarily influenced by bot neighbors.")
        else:
            explanation.append(f"This node is primarily connected to human users, but with a high connection strength.")
        
        result = "\n".join(explanation)
        # Remove trailing newline
        if result.endswith('\n'):
            result = result.rstrip('\n')
        return result
    
    def generate_explanation_report(self, user_idx, top_n=10):
        """
        Generate a complete explanation report.
        
        Integrate the results of three methods:
        1. SHAP: Global feature importance
        2. GraphLIME: Local neighborhood feature importance
        3. PGExplainer: Graph structure importance
        
        Args:
            user_idx: User index
            top_n: Number of top important features to display
        
        Returns:
            report: Explanation report dictionary
        """
        # print(f"🔄 为用户 {user_idx} 生成解释报告...")
        
        # Step 1: Get explanation results from three methods
        shap_importance = self.explain_with_shap(user_idx, top_n)  # Enable SHAP
        graphlime_importance = self.explain_with_graphlime(user_idx, top_n)
        pgexplainer_importance = self.explain_with_pgexplainer(user_idx, top_n)
        
        # Temporarily create an empty SHAP result to avoid NameError
        # shap_importance = {}
        
        # Step 2: Get user prediction probability - use full graph to avoid edge index issues
        with torch.no_grad():
            # Predict directly using the full graph to avoid edge index out of bounds
            logits, _ = self.model(
                self.data.des_embedding,
                self.data.tweet_embedding,
                self.data.num_property_embedding,
                self.data.cat_property_embedding,
                self.data.num_for_h,
                self.data.edge_index,
                self.data.edge_type
            )
            bot_prob = float(torch.softmax(logits, dim=-1)[user_idx, 1])
            # Debug info
            # print(f"[DEBUG] user_idx: {user_idx}")
            # print(f"[DEBUG] num_property_embedding (first 5): {self.data.num_property_embedding[user_idx][:5].cpu().numpy()}")
            # print(f"[DEBUG] logits: {logits[user_idx].cpu().numpy()}")
            # print(f"[DEBUG] softmax: {torch.softmax(logits, dim=-1)[user_idx].cpu().numpy()}")
            # print(f"[DEBUG] bot_prob: {bot_prob}")
        
        # Step 3: Generate comprehensive report
        report = {
            'user_id': int(user_idx),
            'prediction': 'Bot',
            'confidence': bot_prob,
            'shap_explanation': {
                'top_features': shap_importance,
                'summary': self._format_shap_summary(shap_importance)
            },
            'graphlime_explanation': {
                'top_features': graphlime_importance,
                'summary': self._format_graphlime_summary(graphlime_importance)
            },
            'pgexplainer_explanation': {
                'structure_info': pgexplainer_importance,
                'summary': self._format_pgexplainer_summary(pgexplainer_importance)
            },
            'evidence_bundle': self._build_evidence_bundle(
                user_idx,
                shap_importance,
                graphlime_importance,
                pgexplainer_importance,
                top_n=top_n
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return report

    def _build_evidence_bundle(self, user_idx, shap_importance, graphlime_importance, pgexplainer_importance, top_n=10):
        """
        Build evidence tuple aligned with paper notation:
        V_f (feature), V_c (contrastive/local), V_e (edge/neighbor), V_s (subgraph).
        """
        # V_f: feature-level evidence (SHAP)
        sorted_shap = sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        V_f = [{"feature": f, "weight": float(w)} for f, w in sorted_shap]

        # V_c: contrastive evidence (paper Eq. 4)
        contrastive = self._compute_contrastive_scores(user_idx)
        feature_names = list(self.field_slices.keys())
        V_c_pairs = []
        for i, delta in enumerate(contrastive):
            if i < len(feature_names):
                V_c_pairs.append((feature_names[i], float(delta)))
        V_c_pairs = sorted(V_c_pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        V_c = [{"feature": f, "delta": float(d)} for f, d in V_c_pairs]

        # V_e: edge/neighbor influence (PGExplainer)
        V_e = []
        for neighbor_id, importance in pgexplainer_importance.get("most_influential_neighbors", [])[:top_n]:
            relation = pgexplainer_importance.get("neighbor_relations", {}).get(neighbor_id, "unknown relation")
            neighbor_type = "bot" if neighbor_id in pgexplainer_importance.get("bot_neighbors", []) else "human"
            V_e.append(
                {
                    "neighbor_id": int(neighbor_id),
                    "importance": float(importance),
                    "relation": relation,
                    "type": neighbor_type,
                }
            )

        # V_s: subgraph/community signals (k-hop induced subgraph)
        V_s = self._compute_subgraph_metrics(user_idx)

        return {"V_f": V_f, "V_c": V_c, "V_e": V_e, "V_s": V_s}
    
    def _format_shap_summary(self, shap_importance):
        """Format SHAP explanation summary"""
        summary = "SHAP analysis shows that the following features contribute most to bot classification:\n"
        
        # Only show features with non-zero importance
        meaningful_features = [(feature, importance) for feature, importance in shap_importance.items() if abs(importance) > 1e-6]
        
        if meaningful_features:
            for i, (feature, importance) in enumerate(meaningful_features, 1):
                description = FIELD_DESCRIPTIONS.get(feature, feature)
                direction = "positive" if importance > 0 else "negative"
                summary += f"{i}. {feature}: {importance:.4f} ({direction} contribution) - {description}\n"
        else:
            summary += "No significant feature contributions found.\n"
        
        # Remove trailing newline
        if summary.endswith('\n'):
            summary = summary.rstrip('\n')
        
        return summary
    
    def _format_graphlime_summary(self, graphlime_importance):
        """Format GraphLIME explanation summary"""
        summary = "GraphLIME analysis shows that the following features contribute most in the local neighborhood:\n"
        
        # Only show features with non-zero importance
        meaningful_features = [(feature, importance) for feature, importance in graphlime_importance.items() if abs(importance) > 1e-6]
        
        if meaningful_features:
            for i, (feature, importance) in enumerate(meaningful_features, 1):
                description = FIELD_DESCRIPTIONS.get(feature, feature)
                direction = "positive" if importance > 0 else "negative"
                summary += f"{i}. {feature}: {importance:.4f} ({direction} contribution) - {description}\n"
        else:
            summary += "No significant feature contributions found.\n"
        
        # Remove trailing newline
        if summary.endswith('\n'):
            summary = summary.rstrip('\n')
        
        return summary
    
    def _format_pgexplainer_summary(self, pgexplainer_importance):
        """Format PGExplainer explanation summary"""
        summary = "PGExplainer structural analysis results:\n"
        summary += f"• Total important neighbors: {pgexplainer_importance['total_important_neighbors']}\n"
        summary += f"• Bot neighbors: {len(pgexplainer_importance['bot_neighbors'])}\n"
        summary += f"• Human neighbors: {len(pgexplainer_importance['human_neighbors'])}\n"
        summary += f"• Bot neighbor ratio: {pgexplainer_importance['bot_ratio']:.3f}\n"
        
        # Display the most important neighbors
        if pgexplainer_importance['most_influential_neighbors']:
            summary += "\nMost influential neighbors (sorted by importance):\n"
            for i, (neighbor_id, importance) in enumerate(pgexplainer_importance['most_influential_neighbors'][:5], 1):
                neighbor_type = "bot" if neighbor_id in pgexplainer_importance['bot_neighbors'] else "human"
                relation = pgexplainer_importance.get('neighbor_relations', {}).get(neighbor_id, "unknown relation")
                summary += f"  {i}. Neighbor {neighbor_id} (importance: {importance:.3f}, type: {neighbor_type}, relation: {relation})\n"
            # Remove trailing newline
            if summary.endswith('\n'):
                summary = summary.rstrip('\n')
        
        # Add detailed explanation
        if 'explanation' in pgexplainer_importance:
            summary += f"\nDetailed explanation:\n{pgexplainer_importance['explanation']}"
        
        return summary
    
    def save_explanation_report(self, report, output_dir='./explanations'):
        """Save explanation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON format
        json_file = os.path.join(output_dir, f'user_{report["user_id"]}_explanation.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save text format
        txt_file = os.path.join(output_dir, f'user_{report["user_id"]}_explanation.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"User {report['user_id']} bot classification explanation report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prediction: {report['prediction']}\n")
            f.write(f"Confidence: {report['confidence']:.3f}\n\n")
            f.write("SHAP explanation:\n")
            f.write(report['shap_explanation']['summary'] + "\n")
            f.write("GraphLIME explanation:\n")
            f.write(report['graphlime_explanation']['summary'] + "\n")
            f.write("PGExplainer structural explanation:\n")
            f.write(report['pgexplainer_explanation']['summary'] + "\n")
            f.write(f"Generated time: {report['timestamp']}\n")
        
        # print(f"✅ Explanation report saved:")
        # print(f"   - JSON: {json_file}")
        # print(f"   - TXT: {txt_file}")
    
    def analyze_multiple_users(self, num_users=5, top_n=10, save_individual_files=True):
        """
        Analyze multiple bot users.
        
        Args:
            num_users: Number of users to analyze, -1 means all bot users
            top_n: Number of features returned by each explanation method
            save_individual_files: Whether to save individual files, False means only save summary file
        """
        if num_users == -1:
            selected_users = self.bot_users
            # print(f"🔄 Analyzing all {len(selected_users)} bot users...")
        else:
            selected_users = self.bot_users[:num_users]
            # print(f"🔄 Analyzing {len(selected_users)} bot users...")
        
        all_reports = []
        
        for i, user_idx in enumerate(selected_users):
            # print(f"\n--- Analyzing user {i+1}/{len(selected_users)} (ID: {user_idx}) ---")
            report = self.generate_explanation_report(user_idx, top_n)
            all_reports.append(report)
            
            # Decide whether to save individual files based on parameter
            if save_individual_files:
                self.save_explanation_report(report)
        
        # Generate summary report
        self._generate_summary_report(all_reports)
        
        return all_reports
    
    def _generate_summary_report(self, reports):
        """Generate summary report"""
        # print("🔄 Generating summary report...")
        
        # Count feature frequency
        shap_feature_counts = {}
        graphlime_feature_counts = {}
        
        for report in reports:
            # SHAP feature statistics
            for feature in report['shap_explanation']['top_features'].keys():
                shap_feature_counts[feature] = shap_feature_counts.get(feature, 0) + 1
            
            # GraphLIME feature statistics
            for feature in report['graphlime_explanation']['top_features'].keys():
                graphlime_feature_counts[feature] = graphlime_feature_counts.get(feature, 0) + 1
        
        # Generate summary report
        summary = {
            'total_users': len(reports),
            'average_confidence': np.mean([r['confidence'] for r in reports]),
            'shap_top_features': sorted(shap_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'graphlime_top_features': sorted(graphlime_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary report
        with open('./explanations/summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # print(f"\n📊 Summary report:")
        # print(f"   - Analyzed users: {summary['total_users']}")
        # print(f"   - Average confidence: {summary['average_confidence']:.3f}")
        # print(f"   - SHAP most frequent features: {[f[0] for f in summary['shap_top_features'][:5]]}")
        # print(f"   - GraphLIME most frequent features: {[f[0] for f in summary['graphlime_top_features'][:5]]}")
        # print(f"✅ Summary report saved: ./explanations/summary_report.json")

    def generate_natural_language_explanation(self, report, shap_top_n=3, graphlime_top_n=3, user_id_mapping=None):
        """
        Convert the explanation report to a natural language description (English version).
        
        Args:
            report: Explanation report dictionary
            shap_top_n: Number of features selected for SHAP part
            graphlime_top_n: Number of features selected for GraphLIME part
            user_id_mapping: User ID mapping dictionary, mapping index to real ID
        
        Returns:
            str: Natural language explanation (English)
        """
        # Get real user ID
        if user_id_mapping and str(report['user_id']) in user_id_mapping:
            real_user_id = user_id_mapping[str(report['user_id'])]
        else:
            real_user_id = report['user_id']
        
        confidence = report['confidence']
        explanation_parts = []
        
        # Opening
        explanation_parts.append(f"User {real_user_id} is classified as a bot (confidence: {confidence:.3f}). The main reasons are as follows:")
        
        # PGExplainer part - Neighbor influence (V_e / V_s)
        pgexplainer_info = report['pgexplainer_explanation']['structure_info']
        if pgexplainer_info['total_important_neighbors'] > 0:
            bot_count = len(pgexplainer_info['bot_neighbors'])
            human_count = len(pgexplainer_info['human_neighbors'])
            total_neighbors = pgexplainer_info['total_important_neighbors']
            
            if bot_count > 0:
                explanation_parts.append(f"Among the {total_neighbors} important neighbors, {bot_count} are identified as bots.")
            
            # Most influential neighbor
            if pgexplainer_info['most_influential_neighbors']:
                top_neighbor = pgexplainer_info['most_influential_neighbors'][0]
                neighbor_id, importance = top_neighbor
                if user_id_mapping and str(neighbor_id) in user_id_mapping:
                    real_neighbor_id = user_id_mapping[str(neighbor_id)]
                else:
                    real_neighbor_id = neighbor_id
                neighbor_type = "bot" if neighbor_id in pgexplainer_info['bot_neighbors'] else "human"
                relation = pgexplainer_info.get('neighbor_relations', {}).get(neighbor_id, "unknown relation")
                explanation_parts.append(f"Neighbor {real_neighbor_id} ({neighbor_type}, {relation}) has the greatest impact on the classification (importance: {abs(importance):.3f}).")
        
        # SHAP part - Global feature influence (V_f)
        shap_info = report['shap_explanation']['top_features']
        if shap_info:
            sorted_shap_features = sorted(shap_info.items(), key=lambda x: abs(x[1]), reverse=True)[:shap_top_n]
            shap_descriptions = []
            for feature, importance in sorted_shap_features:
                # Use the field English name directly
                direction = "positive" if importance > 0 else "negative"
                shap_descriptions.append(f"'{feature}' ({direction} contribution {abs(importance):.3f})")
            if len(shap_descriptions) == 1:
                explanation_parts.append(f"The following feature most strongly leads to the node being classified as a bot: {shap_descriptions[0]}.")
            else:
                explanation_parts.append(f"The following features most strongly lead to the node being classified as a bot: {', '.join(shap_descriptions)}.")
        
        # GraphLIME part - Local feature influence (V_c)
        graphlime_info = report['graphlime_explanation']['top_features']
        if graphlime_info:
            sorted_features = sorted(graphlime_info.items(), key=lambda x: abs(x[1]), reverse=True)[:graphlime_top_n]
            feature_descriptions = []
            for feature, importance in sorted_features:
                # Use the field English name directly
                direction = "positive" if importance > 0 else "negative"
                feature_descriptions.append(f"'{feature}' ({direction} contribution {abs(importance):.3f})")
            if len(feature_descriptions) == 1:
                explanation_parts.append(f"In the local neighborhood, the following neighbor's feature contributes significantly to the prediction: {feature_descriptions[0]}.")
            else:
                explanation_parts.append(f"In the local neighborhood, the following neighbors' features contribute significantly to the prediction: {', '.join(feature_descriptions)}.")
        
        # Structural summary (V_s)
        if pgexplainer_info['total_important_neighbors'] > 0:
            if pgexplainer_info['bot_ratio'] > 0.5:
                explanation_parts.append("Structurally, this user is mainly connected to bot users, which is an important factor for being classified as a bot.")
            else:
                explanation_parts.append("Structurally, this user is mainly connected to human users, but the connection strength is high, so it is still classified as a bot.")

        if report.get("evidence_bundle"):
            explanation_parts.append("Evidence bundle summary: V_f (feature), V_c (contrastive), V_e (edge), V_s (subgraph) are provided in the JSON report.")
        
        natural_explanation = " ".join(explanation_parts)
        return natural_explanation
    
    def save_natural_language_explanations(self, reports, output_file='./explanations/natural_language_explanations.txt', shap_top_n=3, graphlime_top_n=3, user_id_mapping=None):
        """
        Save natural language explanations for all users.
        
        Args:
            reports: List of explanation reports
            output_file: Output file path
            shap_top_n: Number of features selected for SHAP part
            graphlime_top_n: Number of features selected for GraphLIME part
            user_id_mapping: User ID mapping dictionary, mapping index to real ID
        """
        import json
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate natural language explanations for all users
        natural_explanations = []
        for report in reports:
            natural_explanation = self.generate_natural_language_explanation(report, shap_top_n, graphlime_top_n, user_id_mapping)
            natural_explanations.append(natural_explanation)
        
        # Save to txt file
        with open(output_file, 'w', encoding='utf-8') as f:
            for explanation in natural_explanations:
                f.write(f"{explanation}\n")
        
        # Also save to json file
        json_output_file = os.path.splitext(output_file)[0] + '.json'
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(natural_explanations, f, ensure_ascii=False, indent=2)

    def generate_reprompt_template(self, report, top_k=5, user_id_mapping=None) -> str:
        """
        Generate re-prompt template aligned with Appendix C.4 (Figure 11).
        """
        # Resolve real user id
        if user_id_mapping and str(report['user_id']) in user_id_mapping:
            real_user_id = user_id_mapping[str(report['user_id'])]
        else:
            real_user_id = report['user_id']

        bundle = report.get("evidence_bundle", {})
        V_f = bundle.get("V_f", [])
        V_c = bundle.get("V_c", [])
        V_e = bundle.get("V_e", [])
        V_s = bundle.get("V_s", {})

        def _fmt_k(items, key_field, val_field, k):
            lines = []
            for item in items[:k]:
                lines.append(f"- {item.get(key_field)}: {item.get(val_field)}")
            return "\n".join(lines) if lines else "- None"

        vf_text = _fmt_k(V_f, "feature", "weight", top_k)
        vc_text = _fmt_k(V_c, "feature", "delta", top_k)
        ve_text = _fmt_k(V_e, "neighbor_id", "importance", top_k)
        vs_text = (
            f"- clustering: {V_s.get('clustering', 0.0)}\n"
            f"- hub_score: {V_s.get('hub_score', 0.0)}\n"
            f"- density: {V_s.get('density', 0.0)}\n"
            f"- bot_ratio: {V_s.get('bot_ratio', 0.0)}\n"
            f"- subgraph_size: {V_s.get('subgraph_size', 0)}"
        )

        template = f"""Evidence to Re-Prompt Template
[Structured Evidence Input]
Receive a structured explanation tuple ψ(ũ) = {{ p̂_ũ , V_f , V_c , V_e , V_s }}, summarizing attribute-level, content-level, and structural factors that contribute to the detector’s classification of the agent.
Target user: {real_user_id}
V_f (feature-level):
{vf_text}
V_c (contrastive-level):
{vc_text}
V_e (edge-level):
{ve_text}
V_s (subgraph-level):
{vs_text}

[Evidence Saliency and Localization]
Identify Top-K influential factors from the explanatory components based on saliency and contrastive signals. Trace the corresponding attribute fields, interaction content, or relational patterns within the agent state that exhibit detectable deviations.

[Semantic Feedback Construction]
Transform the localized explanatory evidence into natural-language adjustment directives T(ψ(ũ)), which specify targeted refinements to attributes, content, or relations. These directives function as semantic gradients rather than numerical optimization signals.

[Memory Update via Re-Prompt]
Augment the agent’s short-term memory by M^S_(u,t+1) ← M^S_(u,t) ∪ T(ψ(ũ)), accumulating interpretable adversarial feedback for subsequent decision making.

[In-Place Behavioral Refinement]
Guided by the updated memory state, perform localized, in-place modifications to persona attributes, content artifacts, or relational links corresponding to the identified vulnerabilities. Only the rectified components are returned, avoiding full regeneration and enabling efficient adversarial convergence.
"""
        return template

    def save_reprompt_explanations(self, reports, output_file='./explanations/re_prompt_explanations.txt', top_k=5, user_id_mapping=None):
        """
        Save re-prompt templates (Appendix C.4 format) for all users.
        """
        import json
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        reprompts = []
        for report in reports:
            reprompts.append(self.generate_reprompt_template(report, top_k=top_k, user_id_mapping=user_id_mapping))

        with open(output_file, 'w', encoding='utf-8') as f:
            for block in reprompts:
                f.write(block.strip() + "\n\n")

        json_output_file = os.path.splitext(output_file)[0] + '.json'
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(reprompts, f, ensure_ascii=False, indent=2)

    def save_all_explanations_to_single_file(self, reports, output_dir='./explanations'):
        """
        Save all users' explanations to a single file.
        
        Args:
            reports: List of explanation reports
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all users' JSON format to a single file
        json_file = os.path.join(output_dir, 'all_users_explanations.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        
        # Save all users' text format to a single file
        txt_file = os.path.join(output_dir, 'all_users_explanations.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("Social network bot classification explanation report - All users\n")
            f.write("=" * 60 + "\n\n")
            
            for i, report in enumerate(reports, 1):
                f.write(f"User {report['user_id']} bot classification explanation report\n")
                f.write("-" * 40 + "\n")
                f.write(f"Prediction: {report['prediction']}\n")
                f.write(f"Confidence: {report['confidence']:.3f}\n\n")
                f.write("SHAP explanation:\n")
                f.write(report['shap_explanation']['summary'] + "\n\n")
                f.write("GraphLIME explanation:\n")
                f.write(report['graphlime_explanation']['summary'] + "\n\n")
                f.write("PGExplainer structural explanation:\n")
                f.write(report['pgexplainer_explanation']['summary'] + "\n")
                f.write(f"Generated time: {report['timestamp']}\n")
                f.write("\n" + "=" * 60 + "\n\n")
        
        # print(f"✅ All user explanation reports saved:")
        # print(f"   - JSON: {json_file}")
        # print(f"   - TXT: {txt_file}")


def load_user_id_mapping(mapping_file_path):
    """
    Load user ID mapping file.
    
    Args:
        mapping_file_path: Path to the mapping file
    
    Returns:
        dict: User ID mapping dictionary {index: real_id}
    """
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Process mapping data based on file format
        if isinstance(mapping_data, dict):
            # If it's a direct mapping dictionary
            return mapping_data
        elif isinstance(mapping_data, list):
            # If it's a list format, assume each element contains index and real ID
            mapping = {}
            for item in mapping_data:
                if isinstance(item, dict) and 'index' in item and 'real_id' in item:
                    mapping[item['index']] = item['real_id']
                elif isinstance(item, dict) and 'node_id' in item and 'user_id' in item:
                    mapping[item['node_id']] = item['user_id']
            return mapping
        else:
            # print(f"⚠️ Unknown mapping file format: {type(mapping_data)}")
            return None
            
    except FileNotFoundError:
        # print(f"⚠️ Mapping file not found: {mapping_file_path}")
        return None
    except Exception as e:
        # print(f"❌ Failed to load mapping file: {e}")
        return None


def main():
    """Main function"""
    print("🤖 Social network bot classification model explainability analysis")
    print("=" * 50)
    
    # Configure GPU device
    if torch.cuda.is_available():
        device = 'cuda:0'  # Directly specify GPU 0
        # print("🚀 Using GPU 0")
    else:
        device = 'cpu'
        # print("⚠️ GPU not available, using CPU")
    
    # Configure paths (allow env overrides for pipeline use)
    model_path = os.getenv(
        "EXPLAINER_MODEL_PATH",
        './src/detector/models_quick/AllInOne1_rgcn_rgt_gcn_20250719/best_seed642414.pth'
    )
    data_dir = os.getenv(
        "EXPLAINER_DATA_DIR",
        './src/detector/processed_data/merged_cleaned_profiles'
    )
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Create explainer - use GPU 0
    explainer = BotExplainer(model_path, data_dir, device=device)
    
    # Configure analysis parameters
    num_users = 3  # Number of users to analyze, set to -1 to analyze all bot users
    top_n = 10  # Number of features returned by each explanation method (SHAP, GraphLIME, PGExplainer)
    save_individual_files = False  # Whether to save individual files, False means only save summary file
    
    # Configure feature quantities for natural language explanation
    shap_top_n = 3  # Number of features selected for SHAP part
    graphlime_top_n = 3  # Number of features selected for GraphLIME part
    
    # Load user ID mapping (if any)
    user_id_mapping = None
    # Automatically try to load node ID mapping file
    mapping_file_path = os.getenv(
        "EXPLAINER_MAPPING_PATH",
        './src/detector/processed_data/merged_cleaned_profiles/node_id_mapping.json'
    )
    
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                user_id_mapping = json.load(f)
            print(f"✅ Loaded user ID mapping from: {mapping_file_path}")
            print(f"   Total mappings: {len(user_id_mapping)}")
        except Exception as e:
            print(f"⚠️ Failed to load user ID mapping: {e}")
            user_id_mapping = None
    else:
        print(f"⚠️ User ID mapping file not found: {mapping_file_path}")
    
    # Analyze users
    reports = explainer.analyze_multiple_users(
        num_users=num_users, 
        top_n=top_n, 
        save_individual_files=save_individual_files
    )
    
    # Save explanations for all users to a single file
    explainer.save_all_explanations_to_single_file(reports)
    
    # Generate natural language explanation report (using configured feature quantities and user ID mapping)
    explainer.save_natural_language_explanations(
        reports, 
        shap_top_n=shap_top_n, 
        graphlime_top_n=graphlime_top_n,
        user_id_mapping=user_id_mapping
    )

    # Generate re-prompt templates aligned with Appendix C.4
    explainer.save_reprompt_explanations(
        reports,
        top_k=5,
        user_id_mapping=user_id_mapping
    )
    
    # print("\n🎉 Explainability analysis completed!")
    # print("📁 Results saved in ./explanations/ directory")


if __name__ == '__main__':
    main() 
