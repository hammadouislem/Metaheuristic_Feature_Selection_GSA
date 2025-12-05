"""
Visualization script for GSA Feature Selection Results
Creates various plots to analyze the experimental results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Try to import seaborn (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib defaults")

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.style.use('default')

OUTPUT_DIR = "gsa_results"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

def load_all_results():
    """Load all run summaries from all datasets."""
    all_results = []
    summary_files = glob.glob(os.path.join(OUTPUT_DIR, "*", "*_gsa_runs_summary.csv"))
    
    for file in summary_files:
        df = pd.read_csv(file)
        all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_convergence_histories():
    """Load convergence history for all runs."""
    histories = {}
    history_files = glob.glob(os.path.join(OUTPUT_DIR, "*", "run_*_history.csv"))
    
    for file in history_files:
        # Extract dataset name from path
        parts = Path(file).parts
        dataset_dir = parts[-2]  # e.g., "arrhythmia_csv"
        dataset_name = dataset_dir.replace("_csv", ".csv")
        
        if dataset_name not in histories:
            histories[dataset_name] = []
        
        df = pd.read_csv(file)
        histories[dataset_name].append(df)
    
    return histories

def plot_accuracy_comparison(df_results):
    """Plot accuracy comparison across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    datasets = df_results['dataset'].unique()
    data_for_box = [df_results[df_results['dataset'] == d]['best_accuracy'].values 
                    for d in datasets]
    
    bp = axes[0].boxplot(data_for_box, tick_labels=datasets, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_title('Accuracy Distribution Across Datasets', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bar plot with error bars
    summary = df_results.groupby('dataset').agg({
        'best_accuracy': ['mean', 'std'],
        'selected_features_count': 'mean'
    }).round(4)
    
    x_pos = np.arange(len(datasets))
    means = [summary.loc[d, ('best_accuracy', 'mean')] for d in datasets]
    stds = [summary.loc[d, ('best_accuracy', 'std')] for d in datasets]
    
    bars = axes[1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1].set_ylabel('Mean Accuracy ± Std Dev', fontsize=12)
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_title('Mean Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: accuracy_comparison.png")
    plt.close()

def plot_feature_reduction(df_results):
    """Plot feature selection statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    datasets = df_results['dataset'].unique()
    
    # Get original feature counts (approximate from metadata or results)
    # We'll calculate reduction percentage
    summary = df_results.groupby('dataset').agg({
        'selected_features_count': ['mean', 'std', 'min', 'max']
    })
    
    # Bar plot: Mean selected features
    means = [summary.loc[d, ('selected_features_count', 'mean')] for d in datasets]
    stds = [summary.loc[d, ('selected_features_count', 'std')] for d in datasets]
    
    x_pos = np.arange(len(datasets))
    bars = axes[0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[0].set_ylabel('Number of Selected Features', fontsize=12)
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_title('Mean Selected Features per Dataset', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(mean)}',
                    ha='center', va='bottom', fontsize=9)
    
    # Scatter plot: Accuracy vs Selected Features
    for dataset in datasets:
        data = df_results[df_results['dataset'] == dataset]
        axes[1].scatter(data['selected_features_count'], data['best_accuracy'],
                       label=dataset, alpha=0.6, s=100)
    
    axes[1].set_xlabel('Number of Selected Features', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy vs Feature Count', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'feature_reduction.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: feature_reduction.png")
    plt.close()

def plot_convergence_curves(histories):
    """Plot convergence curves for each dataset."""
    n_datasets = len(histories)
    if n_datasets == 0:
        print("No convergence history files found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (dataset, history_list) in enumerate(histories.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot all runs
        for i, hist_df in enumerate(history_list):
            ax.plot(hist_df['iteration'], hist_df['best_accuracy'], 
                   alpha=0.3, linewidth=1, color='gray')
        
        # Plot mean convergence
        all_iterations = set()
        for hist_df in history_list:
            all_iterations.update(hist_df['iteration'].values)
        
        iterations = sorted(all_iterations)
        mean_accuracies = []
        std_accuracies = []
        
        for it in iterations:
            accs = []
            for hist_df in history_list:
                if it in hist_df['iteration'].values:
                    accs.append(hist_df[hist_df['iteration'] == it]['best_accuracy'].values[0])
            if accs:
                mean_accuracies.append(np.mean(accs))
                std_accuracies.append(np.std(accs))
            else:
                mean_accuracies.append(np.nan)
                std_accuracies.append(np.nan)
        
        # Filter out NaN values
        valid_indices = ~np.isnan(mean_accuracies)
        iterations_clean = [it for i, it in enumerate(iterations) if valid_indices[i]]
        mean_accuracies_clean = [m for i, m in enumerate(mean_accuracies) if valid_indices[i]]
        std_accuracies_clean = [s for i, s in enumerate(std_accuracies) if valid_indices[i]]
        
        ax.plot(iterations_clean, mean_accuracies_clean, 
               linewidth=2.5, color='#2ca02c', label='Mean', marker='o', markersize=4)
        ax.fill_between(iterations_clean, 
                       np.array(mean_accuracies_clean) - np.array(std_accuracies_clean),
                       np.array(mean_accuracies_clean) + np.array(std_accuracies_clean),
                       alpha=0.2, color='#2ca02c', label='±1 Std Dev')
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Best Accuracy', fontsize=11)
        ax.set_title(f'{dataset}\nConvergence Curve', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(histories), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'convergence_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: convergence_curves.png")
    plt.close()

def plot_runtime_analysis(df_results):
    """Plot runtime analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    datasets = df_results['dataset'].unique()
    
    # Box plot: Runtime
    data_for_box = [df_results[df_results['dataset'] == d]['time_seconds'].values 
                    for d in datasets]
    
    bp = axes[0].boxplot(data_for_box, tick_labels=datasets, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_title('Runtime Distribution Across Datasets', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Scatter: Runtime vs Accuracy
    for dataset in datasets:
        data = df_results[df_results['dataset'] == dataset]
        axes[1].scatter(data['time_seconds'], data['best_accuracy'],
                       label=dataset, alpha=0.6, s=100)
    
    axes[1].set_xlabel('Runtime (seconds)', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Runtime vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'runtime_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: runtime_analysis.png")
    plt.close()

def plot_summary_table(df_results):
    """Create a summary statistics table visualization."""
    summary = df_results.groupby('dataset').agg({
        'best_accuracy': ['mean', 'std', 'min', 'max'],
        'selected_features_count': ['mean', 'std'],
        'time_seconds': 'mean'
    }).round(4)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for dataset in summary.index:
        row = [
            dataset,
            f"{summary.loc[dataset, ('best_accuracy', 'mean')]:.4f}",
            f"{summary.loc[dataset, ('best_accuracy', 'std')]:.4f}",
            f"{summary.loc[dataset, ('best_accuracy', 'min')]:.4f}",
            f"{summary.loc[dataset, ('best_accuracy', 'max')]:.4f}",
            f"{summary.loc[dataset, ('selected_features_count', 'mean')]:.1f}",
            f"{summary.loc[dataset, ('time_seconds', 'mean')]:.2f}"
        ]
        table_data.append(row)
    
    columns = ['Dataset', 'Mean Acc', 'Std Acc', 'Min Acc', 'Max Acc', 
               'Mean Features', 'Mean Time (s)']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary Statistics Across All Datasets', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(VIS_DIR, 'summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: summary_table.png")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations for GSA Feature Selection Results")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading results...")
    df_results = load_all_results()
    
    if df_results.empty:
        print("❌ No results found! Please run the experiments first.")
        return
    
    print(f"✓ Loaded {len(df_results)} run results from {df_results['dataset'].nunique()} datasets")
    
    # Load convergence histories
    print("\n📈 Loading convergence histories...")
    histories = load_convergence_histories()
    print(f"✓ Loaded convergence data for {len(histories)} datasets")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    plot_accuracy_comparison(df_results)
    plot_feature_reduction(df_results)
    plot_runtime_analysis(df_results)
    plot_summary_table(df_results)
    
    if histories:
        plot_convergence_curves(histories)
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {VIS_DIR}/")
    print("=" * 60)
    
    # Print summary
    print("\n📋 Quick Summary:")
    summary = df_results.groupby('dataset').agg({
        'best_accuracy': ['mean', 'std'],
        'selected_features_count': 'mean'
    }).round(4)
    
    for dataset in summary.index:
        mean_acc = summary.loc[dataset, ('best_accuracy', 'mean')]
        std_acc = summary.loc[dataset, ('best_accuracy', 'std')]
        mean_feat = summary.loc[dataset, ('selected_features_count', 'mean')]
        print(f"  {dataset:20s}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}, "
              f"Features = {mean_feat:.1f}")

if __name__ == "__main__":
    main()

