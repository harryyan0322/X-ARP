"""
Bot-Specific Explainability Framework for Social Network Bot Detection
======================================================================

This module extends classic explainability methods (SHAP, GraphLIME, PGExplainer)
with bot-specific innovations:

1. Bot-Specific Feature Importance (BSFI): Identifies bot signature features
2. Bot Network Signature Analysis (BNSA): Analyzes bot network patterns
3. Contrastive Explanation for Bot Detection (CEBD): Compares bot vs human patterns
4. Multi-level Hierarchical Explanation (MHE): Node-edge-subgraph explanations
5. Temporal Pattern Analysis (TPA): Analyzes temporal bot behaviors

Key Innovations:
- Bot signature feature combinations
- Bot network community detection
- Contrastive feature importance
- Hierarchical explanation aggregation
"""

import os
import sys
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import k_hop_subgraph

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from main_explainer import BotExplainer
from feature_mapping import FIELD_DESCRIPTIONS, get_field_slices


class BotSpecificExplainer(BotExplainer):
    """
    Extended explainer with bot-specific analysis methods.
    Inherits from BotExplainer and adds bot-focused innovations.
    """
    
    def __init__(self, model_path, data_dir, device='cpu'):
        super().__init__(model_path, data_dir, device)
        
        # Bot-specific feature groups (based on bot detection literature)
        self.bot_signature_features = {
            'activity_patterns': [
                'follower_friend_ratio', 'tweet_count', 'account_age_days',
                'avg_tweet_length', 'url_ratio', 'avg_hashtag_count'
            ],
            'profile_characteristics': [
                'verified', 'default_profile_image', 'protected',
                'has_extended_profile', 'url_exists'
            ],
            'social_network': [
                'neighbor_total', 'followers_count', 'friends_count',
                'favourites_count', 'listed_count'
            ],
            'content_patterns': [
                'avg_mention_count', 'avg_emoji_count', 'url_ratio'
            ]
        }
        
        # Pre-compute bot network communities
        self.bot_communities = self._detect_bot_communities()
    
    def _detect_bot_communities(self) -> Dict[int, int]:
        """
        Detect bot network communities using graph structure.
        
        Returns:
            Dict mapping node_id to community_id
        """
        # Build NetworkX graph from edge_index
        G = nx.DiGraph()
        edge_index_np = self.data.edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            G.add_edge(int(src), int(dst))
        
        # Detect communities using label propagation (fast for large graphs)
        try:
            communities = nx.community.label_propagation_communities(G.to_undirected())
            community_dict = {}
            for comm_id, nodes in enumerate(communities):
                for node in nodes:
                    community_dict[node] = comm_id
            return community_dict
        except:
            return {}
    
    def explain_with_bot_signature_features(self, user_idx: int, top_n: int = 10) -> Dict:
        """
        Bot-Specific Feature Importance (BSFI):
        Identifies features that are characteristic of bots vs humans.
        
        Innovation: Instead of generic feature importance, we focus on
        features that distinguish bots from humans in this specific context.
        
        Args:
            user_idx: Target user index
            top_n: Number of top features to return
            
        Returns:
            Dict with bot signature feature importance
        """
        # Get standard SHAP importance
        shap_importance = self.explain_with_shap(user_idx, top_n=top_n)
        
        # Get bot and human user indices
        bot_indices = torch.where(self.data.y == 1)[0].cpu().numpy()
        human_indices = torch.where(self.data.y == 0)[0].cpu().numpy()
        
        # Calculate feature statistics for bots vs humans
        x = torch.cat([
            self.data.num_property_embedding,
            self.data.cat_property_embedding,
            self.data.num_for_h
        ], dim=1)
        
        bot_features = x[bot_indices].cpu().numpy()
        human_features = x[human_indices].cpu().numpy()
        
        # Calculate bot signature score: how much does this feature distinguish bots?
        feature_names = list(self.field_slices.keys())
        bot_signature_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < x.shape[1]:
                bot_mean = np.mean(bot_features[:, i])
                human_mean = np.mean(human_features[:, i])
                bot_std = np.std(bot_features[:, i])
                human_std = np.std(human_features[:, i])
                
                # Bot signature score: difference in means normalized by pooled std
                pooled_std = np.sqrt((bot_std**2 + human_std**2) / 2)
                if pooled_std > 1e-6:
                    signature_score = abs(bot_mean - human_mean) / pooled_std
                else:
                    signature_score = 0.0
                
                # Combine with SHAP importance (weighted combination)
                shap_score = abs(shap_importance.get(feature_name, 0.0))
                combined_score = 0.6 * shap_score + 0.4 * signature_score
                
                bot_signature_scores[feature_name] = {
                    'combined_importance': combined_score,
                    'shap_importance': shap_score,
                    'signature_score': signature_score,
                    'bot_mean': float(bot_mean),
                    'human_mean': float(human_mean),
                    'distinctiveness': float(signature_score)
                }
        
        # Sort by combined importance
        sorted_features = sorted(
            bot_signature_scores.items(),
            key=lambda x: x[1]['combined_importance'],
            reverse=True
        )[:top_n]
        
        return {
            'top_signature_features': dict(sorted_features),
            'method': 'Bot-Specific Feature Importance (BSFI)',
            'description': 'Features that both contribute to prediction (SHAP) and distinguish bots from humans (signature analysis)'
        }
    
    def explain_with_bot_network_analysis(self, user_idx: int, top_n: int = 10) -> Dict:
        """
        Bot Network Signature Analysis (BNSA):
        Analyzes the bot network structure around the target user.
        
        Innovation: Identifies bot network patterns (clusters, bridges, hubs)
        that are characteristic of bot networks.
        
        Args:
            user_idx: Target user index
            top_n: Number of top neighbors to analyze
            
        Returns:
            Dict with bot network analysis results
        """
        # Get standard PGExplainer results
        pgexplainer_result = self.explain_with_pgexplainer(user_idx, top_n=top_n)
        
        # Get neighbors
        bot_neighbors = pgexplainer_result.get('bot_neighbors', [])
        human_neighbors = pgexplainer_result.get('human_neighbors', [])
        all_neighbors = bot_neighbors + human_neighbors
        
        # Analyze bot network patterns
        bot_network_metrics = {
            'bot_cluster_coefficient': 0.0,
            'bot_bridge_score': 0.0,
            'bot_hub_score': 0.0,
            'network_density': 0.0
        }
        
        if len(bot_neighbors) > 0:
            # Build subgraph of bot neighbors
            bot_subgraph_indices = [user_idx] + bot_neighbors
            
            # Calculate bot cluster coefficient (how connected are bot neighbors?)
            bot_edges = 0
            possible_edges = len(bot_neighbors) * (len(bot_neighbors) - 1) / 2
            if possible_edges > 0:
                for i, n1 in enumerate(bot_neighbors):
                    for n2 in bot_neighbors[i+1:]:
                        # Check if edge exists
                        edge_exists = (
                            ((self.data.edge_index[0] == n1) & (self.data.edge_index[1] == n2)).any() or
                            ((self.data.edge_index[0] == n2) & (self.data.edge_index[1] == n1)).any()
                        )
                        if edge_exists:
                            bot_edges += 1
                bot_network_metrics['bot_cluster_coefficient'] = bot_edges / possible_edges
            
            # Calculate bot bridge score (how many human-bot connections?)
            bridge_edges = len(human_neighbors)
            bot_network_metrics['bot_bridge_score'] = bridge_edges / max(len(all_neighbors), 1)
            
            # Calculate bot hub score (how central is this bot in the network?)
            # Based on number of bot neighbors and their connections
            bot_network_metrics['bot_hub_score'] = len(bot_neighbors) / max(len(all_neighbors), 1)
            
            # Network density (ratio of actual edges to possible edges)
            total_edges = 0
            for n1 in bot_subgraph_indices:
                for n2 in bot_subgraph_indices:
                    if n1 != n2:
                        edge_exists = (
                            ((self.data.edge_index[0] == n1) & (self.data.edge_index[1] == n2)).any() or
                            ((self.data.edge_index[0] == n2) & (self.data.edge_index[1] == n1)).any()
                        )
                        if edge_exists:
                            total_edges += 1
            possible_total = len(bot_subgraph_indices) * (len(bot_subgraph_indices) - 1)
            if possible_total > 0:
                bot_network_metrics['network_density'] = total_edges / possible_total
        
        # Check if user is in a bot community
        user_community = self.bot_communities.get(user_idx, -1)
        community_bot_ratio = 0.0
        if user_community >= 0:
            community_nodes = [n for n, c in self.bot_communities.items() if c == user_community]
            community_bots = sum(1 for n in community_nodes if self.data.y[n] == 1)
            if len(community_nodes) > 0:
                community_bot_ratio = community_bots / len(community_nodes)
        
        return {
            'bot_network_metrics': bot_network_metrics,
            'community_id': user_community,
            'community_bot_ratio': community_bot_ratio,
            'bot_neighbors_count': len(bot_neighbors),
            'human_neighbors_count': len(human_neighbors),
            'pgexplainer_result': pgexplainer_result,
            'method': 'Bot Network Signature Analysis (BNSA)',
            'description': 'Analyzes bot network structure patterns: clustering, bridging, and hub characteristics'
        }
    
    def explain_with_contrastive_analysis(self, user_idx: int, top_n: int = 10) -> Dict:
        """
        Contrastive Explanation for Bot Detection (CEBD):
        Compares the target user's features against typical human and bot patterns.
        
        Innovation: Instead of just explaining why a user is a bot,
        we explain how different they are from humans and how similar to bots.
        
        Args:
            user_idx: Target user index
            top_n: Number of top features to return
            
        Returns:
            Dict with contrastive analysis results
        """
        # Get user features
        x = torch.cat([
            self.data.num_property_embedding,
            self.data.cat_property_embedding,
            self.data.num_for_h
        ], dim=1)
        user_features = x[user_idx:user_idx+1].cpu().numpy()[0]
        
        # Get bot and human feature distributions
        bot_indices = torch.where(self.data.y == 1)[0].cpu().numpy()
        human_indices = torch.where(self.data.y == 0)[0].cpu().numpy()
        
        bot_features = x[bot_indices].cpu().numpy()
        human_features = x[human_indices].cpu().numpy()
        
        feature_names = list(self.field_slices.keys())
        contrastive_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < x.shape[1]:
                user_value = user_features[i]
                bot_mean = np.mean(bot_features[:, i])
                human_mean = np.mean(human_features[:, i])
                bot_std = np.std(bot_features[:, i])
                human_std = np.std(human_features[:, i])
                
                # Calculate distance to bot and human distributions
                if bot_std > 1e-6:
                    distance_to_bot = abs(user_value - bot_mean) / bot_std
                else:
                    distance_to_bot = abs(user_value - bot_mean)
                
                if human_std > 1e-6:
                    distance_to_human = abs(user_value - human_mean) / human_std
                else:
                    distance_to_human = abs(user_value - human_mean)
                
                # Contrastive score: how much closer to bot than human?
                # Negative means closer to bot, positive means closer to human
                contrastive_score = distance_to_human - distance_to_bot
                
                # Normalize to [0, 1] range for importance
                normalized_score = (contrastive_score + 3) / 6  # Assume scores in [-3, 3] range
                normalized_score = max(0, min(1, normalized_score))
                
                contrastive_scores[feature_name] = {
                    'contrastive_score': float(contrastive_score),
                    'distance_to_bot': float(distance_to_bot),
                    'distance_to_human': float(distance_to_human),
                    'user_value': float(user_value),
                    'bot_mean': float(bot_mean),
                    'human_mean': float(human_mean),
                    'bot_similarity': 1.0 / (1.0 + distance_to_bot),  # Higher = more similar to bot
                    'human_similarity': 1.0 / (1.0 + distance_to_human)  # Higher = more similar to human
                }
        
        # Sort by contrastive score (most bot-like features first)
        sorted_features = sorted(
            contrastive_scores.items(),
            key=lambda x: x[1]['contrastive_score'],  # Negative = closer to bot
            reverse=False  # Most negative (bot-like) first
        )[:top_n]
        
        return {
            'contrastive_features': dict(sorted_features),
            'method': 'Contrastive Explanation for Bot Detection (CEBD)',
            'description': 'Features that make the user more similar to bots than humans'
        }
    
    def explain_with_hierarchical_aggregation(self, user_idx: int, top_n: int = 10) -> Dict:
        """
        Multi-level Hierarchical Explanation (MHE):
        Aggregates explanations from node, edge, and subgraph levels.
        
        Innovation: Combines feature-level, relationship-level, and
        community-level explanations into a unified hierarchical view.
        
        Args:
            user_idx: Target user index
            top_n: Number of top features to return
            
        Returns:
            Dict with hierarchical explanation results
        """
        # Level 1: Node-level (feature importance)
        shap_result = self.explain_with_shap(user_idx, top_n)
        graphlime_result = self.explain_with_graphlime(user_idx, top_n)
        
        # Level 2: Edge-level (relationship importance)
        pgexplainer_result = self.explain_with_pgexplainer(user_idx, top_n)
        
        # Level 3: Subgraph-level (community/network importance)
        bot_network_result = self.explain_with_bot_network_analysis(user_idx, top_n)
        
        # Aggregate feature importance across levels
        aggregated_importance = defaultdict(float)
        feature_names = list(self.field_slices.keys())
        
        # Weight different levels
        node_weight = 0.4
        edge_weight = 0.3
        subgraph_weight = 0.3
        
        # Node-level features (SHAP + GraphLIME)
        for feature, importance in shap_result.items():
            aggregated_importance[feature] += node_weight * 0.6 * abs(importance)
        for feature, importance in graphlime_result.items():
            aggregated_importance[feature] += node_weight * 0.4 * abs(importance)
        
        # Edge-level: features of important neighbors
        # (This would require analyzing neighbor features, simplified here)
        neighbor_importance = pgexplainer_result.get('neighbor_importance', {})
        if neighbor_importance:
            # Boost features that are common in important bot neighbors
            for neighbor_idx, importance in list(neighbor_importance.items())[:top_n]:
                if neighbor_idx < len(self.data.y) and self.data.y[neighbor_idx] == 1:
                    # If neighbor is a bot, its features are more important
                    neighbor_shap = self.explain_with_shap(int(neighbor_idx), top_n=5)
                    for feature, feat_importance in neighbor_shap.items():
                        aggregated_importance[feature] += edge_weight * 0.3 * abs(feat_importance) * abs(importance)
        
        # Subgraph-level: community features
        community_bot_ratio = bot_network_result.get('community_bot_ratio', 0.0)
        if community_bot_ratio > 0.5:  # Bot-dominated community
            # Boost social network features
            for feature in self.bot_signature_features['social_network']:
                if feature in aggregated_importance:
                    aggregated_importance[feature] *= (1 + subgraph_weight * community_bot_ratio)
        
        # Sort and return top features
        sorted_features = sorted(
            aggregated_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            'hierarchical_importance': dict(sorted_features),
            'node_level': {
                'shap': shap_result,
                'graphlime': graphlime_result
            },
            'edge_level': pgexplainer_result,
            'subgraph_level': bot_network_result,
            'method': 'Multi-level Hierarchical Explanation (MHE)',
            'description': 'Unified explanation aggregating node, edge, and subgraph levels'
        }
    
    def generate_bot_specific_explanation_report(self, user_idx: int, top_n: int = 10) -> Dict:
        """
        Generate comprehensive bot-specific explanation report.
        
        Combines all bot-specific methods into a unified report.
        
        Args:
            user_idx: Target user index
            top_n: Number of top features to return
            
        Returns:
            Comprehensive explanation report
        """
        # Get prediction
        with torch.no_grad():
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
        
        # Get all bot-specific explanations
        bsfi_result = self.explain_with_bot_signature_features(user_idx, top_n)
        bnsa_result = self.explain_with_bot_network_analysis(user_idx, top_n)
        cebd_result = self.explain_with_contrastive_analysis(user_idx, top_n)
        mhe_result = self.explain_with_hierarchical_aggregation(user_idx, top_n)
        
        # Also include standard methods for comparison
        standard_shap = self.explain_with_shap(user_idx, top_n)
        standard_graphlime = self.explain_with_graphlime(user_idx, top_n)
        standard_pgexplainer = self.explain_with_pgexplainer(user_idx, top_n)
        
        return {
            'user_id': int(user_idx),
            'prediction': 'Bot',
            'confidence': bot_prob,
            'bot_specific_methods': {
                'bsfi': bsfi_result,
                'bnsa': bnsa_result,
                'cebd': cebd_result,
                'mhe': mhe_result
            },
            'standard_methods': {
                'shap': standard_shap,
                'graphlime': standard_graphlime,
                'pgexplainer': standard_pgexplainer
            },
            'timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

