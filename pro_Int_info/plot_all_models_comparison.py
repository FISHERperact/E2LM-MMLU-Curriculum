#!/usr/bin/env python3
"""
Comprehensive comparison plot for Dense-500M, Dense-1B, and Dense-3B models
Generates two plots:
1. Multiple-choice MMLU results comparison
2. Fill-in-the-blank MMLU results comparison
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_iter_number(checkpoint_name):
    """Extract iteration number from checkpoint name"""
    # iter_0002000 -> 2000
    try:
        return int(checkpoint_name.replace('iter_', ''))
    except:
        return 0

def load_mc_results(checkpoint_dir):
    """Load multiple-choice evaluation results"""
    result_files = glob.glob(f"{checkpoint_dir}/**/results_*.json", recursive=True)
    
    if not result_files:
        return None
    
    try:
        with open(result_files[0], 'r') as f:
            data = json.load(f)
        return data.get('results', {})
    except:
        return None

def load_fb_results(checkpoint_dir):
    """Load fill-in-the-blank evaluation results"""
    result_files = glob.glob(f"{checkpoint_dir}/results_*.json")
    
    if not result_files:
        return None
    
    try:
        with open(result_files[0], 'r') as f:
            data = json.load(f)
        return data
    except:
        return None

def collect_mc_data(results_dir, model_name):
    """Collect multiple-choice results for a model"""
    if not os.path.exists(results_dir):
        print(f"⚠️  Warning: {model_name} MC results not found at {results_dir}")
        return None, None, None
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"⚠️  Warning: {model_name} MC results directory is empty")
        return None, None, None
    
    iter_numbers = []
    accuracies = []
    checkpoints = []
    skipped = []
    
    for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: extract_iter_number(x.name)):
        checkpoint_name = checkpoint_dir.name
        
        # Skip iter_0050000 (problematic checkpoint)
        if checkpoint_name == 'iter_0050000':
            skipped.append(checkpoint_name)
            continue
        
        results = load_mc_results(checkpoint_dir)
        
        if results and 'mmlu' in results:
            checkpoints.append(checkpoint_name)
            iter_numbers.append(extract_iter_number(checkpoint_name))
            acc_norm = results['mmlu'].get('acc_norm,none', results['mmlu'].get('acc,none', 0))
            accuracies.append(acc_norm * 100)  # Convert to percentage
    
    if skipped:
        print(f"  ✓ {model_name} MC: {len(checkpoints)} checkpoints (skipped: {', '.join(skipped)})")
    else:
        print(f"  ✓ {model_name} MC: {len(checkpoints)} checkpoints")
    return iter_numbers, accuracies, checkpoints

def collect_fb_data(results_dir, model_name):
    """Collect fill-in-the-blank results for a model"""
    if not os.path.exists(results_dir):
        print(f"⚠️  Warning: {model_name} FB results not found at {results_dir}")
        return None, None, None
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"⚠️  Warning: {model_name} FB results directory is empty")
        return None, None, None
    
    iter_numbers = []
    accuracies = []
    checkpoints = []
    skipped = []
    
    for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: extract_iter_number(x.name)):
        checkpoint_name = checkpoint_dir.name
        
        # Skip iter_0050000 (problematic checkpoint)
        if checkpoint_name == 'iter_0050000':
            skipped.append(checkpoint_name)
            continue
        
        results = load_fb_results(checkpoint_dir)
        
        if results and 'accuracy' in results:
            checkpoints.append(checkpoint_name)
            iter_numbers.append(extract_iter_number(checkpoint_name))
            acc = results.get('accuracy', 0)
            accuracies.append(acc * 100)  # Convert to percentage
    
    if skipped:
        print(f"  ✓ {model_name} FB: {len(checkpoints)} checkpoints (skipped: {', '.join(skipped)})")
    else:
        print(f"  ✓ {model_name} FB: {len(checkpoints)} checkpoints")
    return iter_numbers, accuracies, checkpoints

def main():
    base_dir = "/home2/yth/pro_Int_info"
    
    # Define model configurations
    models = {
        'Dense-500M': {
            'mc_dir': f"{base_dir}/dense500m_evaluation/Multiple-choice questions/results_dense500m",
            'fb_dir': f"{base_dir}/dense500m_evaluation/Fill-in-the-blank questions/results_fillblank",
            'color': '#2E86AB',
            'marker': 'o'
        },
        'Dense-1B': {
            'mc_dir': f"{base_dir}/dense1b_evaluation/Multiple-choice questions/results_dense1b",
            'fb_dir': f"{base_dir}/dense1b_evaluation/Fill-in-the-blank questions/results_fillblank1b_batch",
            'color': '#A23B72',
            'marker': 's'
        },
        'Dense-3B': {
            'mc_dir': f"{base_dir}/dense3b_evaluation/Multiple-choice questions/results_dense3b",
            'fb_dir': f"{base_dir}/dense3b_evaluation/Fill-in-the-blank questions/results_fillblank3b_batch",
            'color': '#F18F01',
            'marker': '^'
        }
    }
    
    print("="*80)
    print("Collecting evaluation data for all models...")
    print("="*80)
    
    # Collect data for all models
    mc_data = {}
    fb_data = {}
    
    print("\nMultiple-Choice Results:")
    for model_name, config in models.items():
        iters, accs, ckpts = collect_mc_data(config['mc_dir'], model_name)
        if iters:
            mc_data[model_name] = {
                'iters': iters,
                'accs': accs,
                'checkpoints': ckpts,
                'color': config['color'],
                'marker': config['marker']
            }
    
    print("\nFill-in-the-Blank Results:")
    for model_name, config in models.items():
        iters, accs, ckpts = collect_fb_data(config['fb_dir'], model_name)
        if iters:
            fb_data[model_name] = {
                'iters': iters,
                'accs': accs,
                'checkpoints': ckpts,
                'color': config['color'],
                'marker': config['marker']
            }
    
    if not mc_data and not fb_data:
        print("\n❌ Error: No evaluation data found for any model")
        return
    
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ========================================================================
    # PLOT 1: Multiple-Choice Comparison
    # ========================================================================
    if mc_data:
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        
        for model_name, data in mc_data.items():
            ax1.plot(data['iters'], data['accs'], 
                    marker=data['marker'], 
                    linestyle='-', 
                    linewidth=2.5, 
                    markersize=8,
                    label=model_name,
                    color=data['color'],
                    alpha=0.8)
        
        ax1.set_xlabel('Training Iterations', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MMLU Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Multiple-Choice MMLU Evaluation Results\nDense-500M vs Dense-1B vs Dense-3B', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = "Performance Summary:\n"
        for model_name, data in mc_data.items():
            if data['accs']:
                final_acc = data['accs'][-1]
                initial_acc = data['accs'][0]
                improvement = final_acc - initial_acc
                stats_text += f"{model_name}: {initial_acc:.2f}% → {final_acc:.2f}% (+{improvement:.2f}%)\n"
        
        ax1.text(0.02, 0.98, stats_text.strip(),
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontweight='bold',
                family='monospace')
        
        plt.tight_layout()
        output_file_mc = f"{base_dir}/comparison_multiple_choice.png"
        plt.savefig(output_file_mc, dpi=300, bbox_inches='tight')
        print(f"\n✅ Multiple-choice comparison plot saved: {output_file_mc}")
        
        # Print MC statistics
        print("\nMultiple-Choice Statistics:")
        print("-" * 80)
        for model_name, data in mc_data.items():
            if data['accs']:
                print(f"\n{model_name}:")
                print(f"  Checkpoints: {len(data['checkpoints'])}")
                print(f"  Initial accuracy: {data['accs'][0]:.2f}%")
                print(f"  Final accuracy: {data['accs'][-1]:.2f}%")
                print(f"  Max accuracy: {max(data['accs']):.2f}%")
                print(f"  Improvement: +{data['accs'][-1] - data['accs'][0]:.2f}%")
    
    # ========================================================================
    # PLOT 2: Fill-in-the-Blank Comparison
    # ========================================================================
    if fb_data:
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        
        for model_name, data in fb_data.items():
            ax2.plot(data['iters'], data['accs'], 
                    marker=data['marker'], 
                    linestyle='-', 
                    linewidth=2.5, 
                    markersize=8,
                    label=model_name,
                    color=data['color'],
                    alpha=0.8)
        
        ax2.set_xlabel('Training Iterations', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Fill-in-the-blank Accuracy (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Fill-in-the-Blank MMLU Evaluation Results\nDense-500M vs Dense-1B vs Dense-3B', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = "Performance Summary:\n"
        for model_name, data in fb_data.items():
            if data['accs']:
                final_acc = data['accs'][-1]
                initial_acc = data['accs'][0]
                improvement = final_acc - initial_acc
                stats_text += f"{model_name}: {initial_acc:.2f}% → {final_acc:.2f}% (+{improvement:.2f}%)\n"
        
        ax2.text(0.02, 0.98, stats_text.strip(),
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontweight='bold',
                family='monospace')
        
        plt.tight_layout()
        output_file_fb = f"{base_dir}/comparison_fill_in_blank.png"
        plt.savefig(output_file_fb, dpi=300, bbox_inches='tight')
        print(f"\n✅ Fill-in-the-blank comparison plot saved: {output_file_fb}")
        
        # Print FB statistics
        print("\nFill-in-the-Blank Statistics:")
        print("-" * 80)
        for model_name, data in fb_data.items():
            if data['accs']:
                print(f"\n{model_name}:")
                print(f"  Checkpoints: {len(data['checkpoints'])}")
                print(f"  Initial accuracy: {data['accs'][0]:.2f}%")
                print(f"  Final accuracy: {data['accs'][-1]:.2f}%")
                print(f"  Max accuracy: {max(data['accs']):.2f}%")
                print(f"  Improvement: +{data['accs'][-1] - data['accs'][0]:.2f}%")
    
    print("\n" + "="*80)
    print("✅ All comparison plots generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    if mc_data:
        print(f"  • {base_dir}/comparison_multiple_choice.png")
    if fb_data:
        print(f"  • {base_dir}/comparison_fill_in_blank.png")
    print()

if __name__ == "__main__":
    main()

