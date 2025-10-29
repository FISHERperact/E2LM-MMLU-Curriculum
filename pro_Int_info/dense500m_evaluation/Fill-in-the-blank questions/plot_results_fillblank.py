#!/usr/bin/env python3
"""
Plot evaluation results for Dense-500M-Arch1 model on Fill-in-the-blank MMLU
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(checkpoint_dir):
    """Load evaluation results for a specific checkpoint"""
    result_files = glob.glob(f"{checkpoint_dir}/results_*.json")
    
    if not result_files:
        return None
    
    with open(result_files[0], 'r') as f:
        data = json.load(f)
    
    return data

def extract_iter_number(checkpoint_name):
    """Extract iteration number from checkpoint name"""
    # iter_0002000 -> 2000
    try:
        return int(checkpoint_name.replace('iter_', ''))
    except:
        return 0

def main():
    results_dir = "/home2/yth/pro_Int_info/dense500m_evaluation/Fill-in-the-blank questions/results_fillblank"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run evaluation first")
        return
    
    # Collect results from all checkpoints
    checkpoints = []
    iter_numbers = []
    accuracies = []
    total_samples = []
    correct_counts = []
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"Error: {results_dir} is empty")
        return
    
    print("Collecting data...")
    
    for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: extract_iter_number(x.name)):
        checkpoint_name = checkpoint_dir.name
        results = load_results(checkpoint_dir)
        
        if results and 'accuracy' in results:
            checkpoints.append(checkpoint_name)
            iter_numbers.append(extract_iter_number(checkpoint_name))
            
            acc = results.get('accuracy', 0)
            total = results.get('total', 0)
            correct = results.get('correct', 0)
            
            accuracies.append(acc * 100)  # Convert to percentage
            total_samples.append(total)
            correct_counts.append(correct)
            
            print(f"  ✓ {checkpoint_name}: {acc:.2%} ({correct}/{total})")
    
    if not checkpoints:
        print("No evaluation results found")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint results")
    print("Generating plots...")
    
    # Create plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Accuracy vs Training Iterations
    ax1 = axes[0]
    ax1.plot(iter_numbers, accuracies, 'o-', linewidth=2.5, markersize=8, 
             label='Accuracy', color='#2E86AB')
    
    ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fill-in-the-blank MMLU Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Dense-500M-Arch1 Performance on Fill-in-the-blank MMLU\n(Training Progress)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels (adaptive interval)
    step = max(1, len(iter_numbers) // 8)
    for i in range(0, len(iter_numbers), step):
        ax1.annotate(f'{accuracies[i]:.1f}%', 
                    xy=(iter_numbers[i], accuracies[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, alpha=0.7)
    
    # Plot 2: First vs Last Checkpoint Comparison
    ax2 = axes[1]
    
    if len(checkpoints) >= 2:
        x = np.arange(2)
        width = 0.4
        
        first_data = [accuracies[0], correct_counts[0]]
        last_data = [accuracies[-1], correct_counts[-1]]
        
        # Accuracy bar chart
        bars1 = ax2.bar(x, [first_data[0], last_data[0]], width, 
                       color=['#E76F51', '#2A9D8F'], alpha=0.8)
        
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#E76F51')
        ax2.set_title('First vs Last Checkpoint Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels([checkpoints[0], checkpoints[-1]], rotation=0)
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        # Calculate improvements
        improvement = accuracies[-1] - accuracies[0]
        relative_improvement = (improvement / accuracies[0]) * 100 if accuracies[0] > 0 else 0
        
        ax2.text(0.5, 0.95, 
                f'Accuracy Improvement: +{improvement:.2f}% (Relative: +{relative_improvement:.1f}%)\n' +
                f'Correct Count: {correct_counts[0]} → {correct_counts[-1]} (+{correct_counts[-1]-correct_counts[0]})',
                transform=ax2.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "/home2/yth/pro_Int_info/dense500m_evaluation/Fill-in-the-blank questions/dense500m_fillblank_mmlu_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")
    
    # Print detailed statistics
    print("\n" + "=" * 80)
    print("Fill-in-the-blank MMLU Evaluation Statistics:")
    print("-" * 80)
    
    if len(accuracies) > 0:
        min_idx = accuracies.index(min(accuracies))
        max_idx = accuracies.index(max(accuracies))
        
        print(f"Lowest Accuracy:  {min(accuracies):.2f}% ({checkpoints[min_idx]}) - {correct_counts[min_idx]}/{total_samples[min_idx]}")
        print(f"Highest Accuracy: {max(accuracies):.2f}% ({checkpoints[max_idx]}) - {correct_counts[max_idx]}/{total_samples[max_idx]}")
        print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
        print(f"Std Deviation:    {np.std(accuracies):.2f}%")
        
        if len(accuracies) >= 2:
            improvement = accuracies[-1] - accuracies[0]
            improvement_pct = (improvement / accuracies[0]) * 100 if accuracies[0] > 0 else 0
            print(f"\nTraining Progress:")
            print(f"  Start ({checkpoints[0]}): {accuracies[0]:.2f}% ({correct_counts[0]}/{total_samples[0]})")
            print(f"  End   ({checkpoints[-1]}): {accuracies[-1]:.2f}% ({correct_counts[-1]}/{total_samples[-1]})")
            print(f"  Absolute Improvement: +{improvement:.2f}%")
            print(f"  Relative Improvement: +{improvement_pct:.1f}%")
            print(f"  Correct Count Gain:   +{correct_counts[-1] - correct_counts[0]}")
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

