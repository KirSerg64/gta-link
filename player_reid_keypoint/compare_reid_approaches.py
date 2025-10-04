"""
Compare Different Keypoint Matching Approaches for Player Re-Identification

This script runs multiple approaches and compares their performance:
1. Classical ORB with geometric filtering
2. LightGlue with SuperPoint (if available)
3. Hybrid approach with different filtering strategies

Author: CV Algorithm Developer
Date: 2025-10-04
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from typing import Dict, List
import pandas as pd
import argparse


def load_results(results_path: str) -> Dict:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def compare_approaches(results_paths: Dict[str, str], output_dir: str):
    """
    Compare different approaches
    
    Args:
        results_paths: Dictionary mapping approach name to results file path
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results = {}
    for name, path in results_paths.items():
        if os.path.exists(path):
            all_results[name] = load_results(path)
            print(f"✓ Loaded {name}: {len(all_results[name]['results'])} comparisons")
        else:
            print(f"✗ Missing {name}: {path}")
    
    if len(all_results) == 0:
        print("No results found to compare!")
        return
    
    # Calculate statistics for each approach
    stats_data = []
    
    for approach_name, results_data in all_results.items():
        results = results_data['results']
        
        intra_results = [r for r in results if r['is_same_tracklet']]
        inter_results = [r for r in results if not r['is_same_tracklet']]
        
        intra_matches = [r['num_filtered_matches'] for r in intra_results]
        inter_matches = [r['num_filtered_matches'] for r in inter_results]
        
        # Calculate metrics
        intra_mean = np.mean(intra_matches) if intra_matches else 0
        inter_mean = np.mean(inter_matches) if inter_matches else 0
        separation_ratio = intra_mean / (inter_mean + 1e-6)
        
        stats_data.append({
            'Approach': approach_name,
            'Intra-tracklet Mean': intra_mean,
            'Intra-tracklet Std': np.std(intra_matches) if intra_matches else 0,
            'Inter-tracklet Mean': inter_mean,
            'Inter-tracklet Std': np.std(inter_matches) if inter_matches else 0,
            'Separation Ratio': separation_ratio,
            'Num Comparisons': len(results)
        })
    
    # Create DataFrame
    df_stats = pd.DataFrame(stats_data)
    
    print("\n" + "="*100)
    print("COMPARISON OF APPROACHES")
    print("="*100)
    print(df_stats.to_string(index=False))
    print("="*100)
    
    # Save statistics to CSV
    stats_path = os.path.join(output_dir, "approach_comparison.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\n✓ Statistics saved to: {stats_path}")
    
    # Create comparison plots
    create_comparison_plots(all_results, output_dir)


def create_comparison_plots(all_results: Dict[str, Dict], output_dir: str):
    """Create comprehensive comparison plots"""
    
    # Prepare data for plotting
    plot_data = {
        'intra_matches': {},
        'inter_matches': {},
        'separation_ratios': {}
    }
    
    for approach_name, results_data in all_results.items():
        results = results_data['results']
        
        intra_results = [r for r in results if r['is_same_tracklet']]
        inter_results = [r for r in results if not r['is_same_tracklet']]
        
        intra_matches = [r['num_filtered_matches'] for r in intra_results]
        inter_matches = [r['num_filtered_matches'] for r in inter_results]
        
        plot_data['intra_matches'][approach_name] = intra_matches
        plot_data['inter_matches'][approach_name] = inter_matches
        
        intra_mean = np.mean(intra_matches) if intra_matches else 0
        inter_mean = np.mean(inter_matches) if inter_matches else 0
        plot_data['separation_ratios'][approach_name] = intra_mean / (inter_mean + 1e-6)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Box plot comparison - Intra-tracklet
    ax = axes[0, 0]
    data_to_plot = [plot_data['intra_matches'][name] for name in plot_data['intra_matches'].keys()]
    labels = list(plot_data['intra_matches'].keys())
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightgreen', 'lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Number of Matches')
    ax.set_title('Intra-tracklet Matches (Same Player)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Box plot comparison - Inter-tracklet
    ax = axes[0, 1]
    data_to_plot = [plot_data['inter_matches'][name] for name in plot_data['inter_matches'].keys()]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightgreen', 'lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Number of Matches')
    ax.set_title('Inter-tracklet Matches (Different Players)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Separation ratio comparison
    ax = axes[0, 2]
    approaches = list(plot_data['separation_ratios'].keys())
    ratios = [plot_data['separation_ratios'][name] for name in approaches]
    colors = ['green', 'blue', 'red'][:len(approaches)]
    
    bars = ax.bar(approaches, ratios, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Baseline (1.0x)')
    ax.axhline(y=1.5, color='orange', linestyle='--', label='Good threshold (1.5x)')
    ax.set_ylabel('Separation Ratio')
    ax.set_title('Separation Ratio (Intra / Inter)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}x',
               ha='center', va='bottom', fontweight='bold')
    
    # 4. Distribution comparison - overlaid histograms
    ax = axes[1, 0]
    for i, (name, matches) in enumerate(plot_data['intra_matches'].items()):
        ax.hist(matches, bins=30, alpha=0.5, label=f'{name} (Intra)', 
               color=['green', 'blue', 'red'][i])
    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Intra-tracklet Matches')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Mean comparison bar chart
    ax = axes[1, 1]
    x = np.arange(len(approaches))
    width = 0.35
    
    intra_means = [np.mean(plot_data['intra_matches'][name]) for name in approaches]
    inter_means = [np.mean(plot_data['inter_matches'][name]) for name in approaches]
    
    bars1 = ax.bar(x - width/2, intra_means, width, label='Intra-tracklet (Same)', 
                   color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, inter_means, width, label='Inter-tracklet (Different)', 
                   color='red', alpha=0.7)
    
    ax.set_ylabel('Average Number of Matches')
    ax.set_title('Mean Matches Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "SUMMARY\n" + "="*50 + "\n\n"
    
    for i, name in enumerate(approaches):
        intra_mean = np.mean(plot_data['intra_matches'][name])
        inter_mean = np.mean(plot_data['inter_matches'][name])
        ratio = plot_data['separation_ratios'][name]
        
        summary_text += f"{name}:\n"
        summary_text += f"  Intra: {intra_mean:.2f} matches\n"
        summary_text += f"  Inter: {inter_mean:.2f} matches\n"
        summary_text += f"  Ratio: {ratio:.2f}x\n"
        summary_text += f"  Status: {'✓ GOOD' if ratio > 1.5 else '~ OK' if ratio > 1.2 else '✗ POOR'}\n\n"
    
    # Find best approach
    best_approach = max(approaches, key=lambda x: plot_data['separation_ratios'][x])
    summary_text += f"\n{'='*50}\n"
    summary_text += f"BEST APPROACH: {best_approach}\n"
    summary_text += f"Ratio: {plot_data['separation_ratios'][best_approach]:.2f}x"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "approach_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    
    plt.show()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare Different Player Re-ID Keypoint Matching Approaches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input paths
    parser.add_argument(
        '--orb-results',
        type=str,
        default='outputs/keypoint_reid/keypoint_reid_results.json',
        help='Path to ORB approach results JSON file'
    )
    parser.add_argument(
        '--lightglue-results',
        type=str,
        default='outputs/lightglue_reid/lightglue_results.json',
        help='Path to LightGlue approach results JSON file'
    )
    parser.add_argument(
        '--custom-results',
        type=str,
        nargs='*',
        default=[],
        help='Additional custom results JSON files to compare (format: name:path name:path ...)'
    )
    
    # Output path
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/comparison',
        help='Directory to save comparison results and plots'
    )
    
    # Display options
    parser.add_argument(
        '--show-plot',
        action='store_true',
        default=False,
        help='Display plots interactively'
    )
    
    return parser.parse_args()


def main():
    """Main comparison function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Build results paths dictionary
    results_paths = {}
    
    # Add ORB results if file exists
    if os.path.exists(args.orb_results):
        results_paths['ORB + Filters'] = args.orb_results
    
    # Add LightGlue results if file exists
    if os.path.exists(args.lightglue_results):
        results_paths['LightGlue'] = args.lightglue_results
    
    # Add custom results
    for custom in args.custom_results:
        if ':' in custom:
            name, path = custom.split(':', 1)
            if os.path.exists(path):
                results_paths[name] = path
            else:
                print(f"Warning: Custom results file not found: {path}")
    
    if not results_paths:
        print("Error: No valid results files found!")
        print(f"  ORB results: {args.orb_results}")
        print(f"  LightGlue results: {args.lightglue_results}")
        return
    
    output_dir = args.output_dir
    
    print("="*100)
    print("KEYPOINT MATCHING APPROACH COMPARISON")
    print("="*100)
    
    compare_approaches(results_paths, output_dir)
    
    print("\n" + "="*100)
    print("✓ COMPARISON COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
