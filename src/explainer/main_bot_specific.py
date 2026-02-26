#!/usr/bin/env python3
"""
Bot-Specific Explainability Main Script
========================================

This script demonstrates the usage of bot-specific explainability methods.
It extends the standard explainer with bot-focused innovations.
"""

import os
import sys
import torch
import json
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from bot_specific_explainer import BotSpecificExplainer


def main():
    """Main function for bot-specific explainability analysis"""
    print("🤖 Bot-Specific Explainability Analysis")
    print("=" * 60)
    
    # Configure GPU device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Configure paths (same as main_explainer.py)
    model_path = './src/detector/models_quick/AllInOne1_rgcn_rgt_gcn_20250719/best_seed642414.pth'
    data_dir = './src/detector/processed_data/merged_cleaned_profiles'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Create bot-specific explainer
    explainer = BotSpecificExplainer(model_path, data_dir, device=device)
    
    # Configure analysis parameters
    num_users = 3  # Number of users to analyze
    top_n = 10  # Number of top features
    
    # Get bot users
    bot_users = explainer.bot_users
    if len(bot_users) == 0:
        print("❌ No bot users found")
        return
    
    # Select users to analyze
    selected_users = bot_users[:num_users] if num_users > 0 else bot_users
    
    print(f"\n📊 Analyzing {len(selected_users)} bot users with bot-specific methods...")
    print("=" * 60)
    
    all_reports = []
    
    for i, user_idx in enumerate(selected_users):
        print(f"\n--- Analyzing user {i+1}/{len(selected_users)} (ID: {user_idx}) ---")
        
        # Generate bot-specific explanation report
        report = explainer.generate_bot_specific_explanation_report(user_idx, top_n)
        all_reports.append(report)
        
        # Print summary
        print(f"Confidence: {report['confidence']:.3f}")
        print(f"\nBSFI Top Features:")
        bsfi_features = report['bot_specific_methods']['bsfi']['top_signature_features']
        for j, (feature, info) in enumerate(list(bsfi_features.items())[:5], 1):
            print(f"  {j}. {feature}: {info['combined_importance']:.4f} "
                  f"(SHAP: {info['shap_importance']:.4f}, "
                  f"Signature: {info['signature_score']:.4f})")
        
        print(f"\nBNSA Network Metrics:")
        bnsa_metrics = report['bot_specific_methods']['bnsa']['bot_network_metrics']
        print(f"  Bot Cluster Coefficient: {bnsa_metrics['bot_cluster_coefficient']:.3f}")
        print(f"  Bot Bridge Score: {bnsa_metrics['bot_bridge_score']:.3f}")
        print(f"  Bot Hub Score: {bnsa_metrics['bot_hub_score']:.3f}")
        print(f"  Community Bot Ratio: {report['bot_specific_methods']['bnsa']['community_bot_ratio']:.3f}")
        
        print(f"\nCEBD Top Contrastive Features:")
        cebd_features = report['bot_specific_methods']['cebd']['contrastive_features']
        for j, (feature, info) in enumerate(list(cebd_features.items())[:5], 1):
            print(f"  {j}. {feature}: Bot Similarity {info['bot_similarity']:.3f}, "
                  f"Human Similarity {info['human_similarity']:.3f}")
    
    # Save results
    output_dir = './explanations'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'bot_specific_explanations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Bot-specific explanations saved to: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()

