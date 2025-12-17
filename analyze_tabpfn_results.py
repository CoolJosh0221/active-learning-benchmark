"""
Analyze TabPFN Active Learning Experimental Results

This script loads and analyzes the results from TabPFN experiments,
comparing performance across datasets and query strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyze TabPFN active learning experimental results"""
    
    def __init__(self, results_dir='../results', experiment_name='TabPFN'):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing result files
            experiment_name: Name of experiment to filter files
        """
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.aubc_data = None
        self.detail_data = None
        
    def load_results(self, datasets=None):
        """
        Load AUBC and detail results
        
        Args:
            datasets: List of dataset names to load (default: all)
        """
        print("Loading results...")
        
        # Load AUBC files
        aubc_files = list(self.results_dir.glob(f"*{self.experiment_name}*aubc.csv"))
        if not aubc_files:
            print(f"Warning: No AUBC files found for experiment '{self.experiment_name}'")
            return False
        
        aubc_dfs = []
        for f in aubc_files:
            try:
                df = pd.read_csv(f)
                aubc_dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if aubc_dfs:
            self.aubc_data = pd.concat(aubc_dfs, ignore_index=True)
            print(f"Loaded {len(self.aubc_data)} AUBC records from {len(aubc_files)} files")
        
        # Load detail files
        detail_files = list(self.results_dir.glob(f"*{self.experiment_name}*detail.csv"))
        if detail_files:
            detail_dfs = []
            for f in detail_files:
                try:
                    df = pd.read_csv(f)
                    detail_dfs.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            
            if detail_dfs:
                self.detail_data = pd.concat(detail_dfs, ignore_index=True)
                print(f"Loaded {len(self.detail_data)} detail records from {len(detail_files)} files")
        
        return True
    
    def compute_statistics(self):
        """Compute summary statistics across experiments"""
        if self.aubc_data is None:
            print("No AUBC data loaded!")
            return None
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        summary = self.aubc_data.groupby(['dataset', 'query_strategy']).agg({
            'aubc': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        print(summary)
        return summary
    
    def plot_aubc_comparison(self, datasets=None, save_path=None):
        """
        Create bar plot comparing AUBC across strategies for each dataset
        
        Args:
            datasets: List of datasets to plot (default: all)
            save_path: Path to save plot
        """
        if self.aubc_data is None:
            print("No data to plot!")
            return
        
        # Filter datasets if specified
        data = self.aubc_data.copy()
        if datasets:
            data = data[data['dataset'].isin(datasets)]
        
        unique_datasets = data['dataset'].unique()
        n_datasets = len(unique_datasets)
        
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]
        
        for idx, dataset in enumerate(unique_datasets):
            dataset_data = data[data['dataset'] == dataset]
            
            # Compute mean and std
            summary = dataset_data.groupby('query_strategy')['aubc'].agg(['mean', 'std'])
            summary = summary.sort_values('mean', ascending=False)
            
            # Plot
            ax = axes[idx]
            x = np.arange(len(summary))
            bars = ax.bar(x, summary['mean'], yerr=summary['std'], 
                         capsize=5, alpha=0.7, color='steelblue')
            
            # Highlight best strategy
            best_idx = summary['mean'].idxmax()
            bars[0].set_color('darkgreen')
            bars[0].set_alpha(0.9)
            
            ax.set_title(f'{dataset.capitalize()}\nBest: {summary.index[0]} ({summary["mean"].iloc[0]:.4f})',
                        fontweight='bold')
            ax.set_ylabel('AUBC (Area Under Budget Curve)')
            ax.set_xticks(x)
            ax.set_xticklabels(summary.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, summary['mean'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, dataset=None, strategies=None, save_path=None):
        """
        Plot learning curves showing performance over labeling budget
        
        Args:
            dataset: Dataset to plot (default: first available)
            strategies: List of strategies to plot (default: all)
            save_path: Path to save plot
        """
        if self.detail_data is None:
            print("No detail data available!")
            return
        
        data = self.detail_data.copy()
        
        # Filter dataset
        if dataset:
            data = data[data['dataset'] == dataset]
        else:
            dataset = data['dataset'].iloc[0]
            data = data[data['dataset'] == dataset]
        
        # Filter strategies
        if strategies:
            data = data[data['query_strategy'].isin(strategies)]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in data['query_strategy'].unique():
            strategy_data = data[data['query_strategy'] == strategy]
            
            # Group by budget and compute mean/std
            grouped = strategy_data.groupby('labeled_size')['accuracy'].agg(['mean', 'std'])
            
            x = grouped.index
            y_mean = grouped['mean']
            y_std = grouped['std']
            
            # Plot with confidence interval
            ax.plot(x, y_mean, marker='o', label=strategy, linewidth=2, markersize=4)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        ax.set_xlabel('Labeled Pool Size')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Learning Curves - {dataset.capitalize()}', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def statistical_comparison(self, dataset=None):
        """
        Perform statistical tests comparing strategies
        
        Args:
            dataset: Dataset to analyze (default: all)
        """
        if self.aubc_data is None:
            print("No data available!")
            return
        
        data = self.aubc_data.copy()
        if dataset:
            data = data[data['dataset'] == dataset]
        
        print("\n" + "="*80)
        print(f"STATISTICAL COMPARISON - {dataset if dataset else 'All Datasets'}")
        print("="*80)
        
        strategies = data['query_strategy'].unique()
        
        # Pairwise t-tests
        print("\nPairwise t-tests (p-values):")
        print("-" * 60)
        
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                data1 = data[data['query_strategy'] == strat1]['aubc']
                data2 = data[data['query_strategy'] == strat2]['aubc']
                
                if len(data1) > 1 and len(data2) > 1:
                    # Paired t-test (assuming same seeds)
                    t_stat, p_value = stats.ttest_rel(data1, data2)
                    
                    # Effect size (Cohen's d)
                    mean_diff = data1.mean() - data2.mean()
                    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    print(f"{strat1:15s} vs {strat2:15s}: p={p_value:.4f} {significance:3s}  "
                          f"(Δ={mean_diff:+.4f}, d={cohens_d:.2f})")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
        print("Cohen's d: |d|<0.2 (small), 0.2-0.5 (medium), >0.5 (large)")
    
    def generate_report(self, output_path='tabpfn_report.txt'):
        """Generate comprehensive text report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TabPFN Active Learning - Experimental Report\n")
            f.write("="*80 + "\n\n")
            
            if self.aubc_data is not None:
                # Overall statistics
                f.write("1. Overall Statistics\n")
                f.write("-"*80 + "\n")
                summary = self.aubc_data.groupby(['dataset', 'query_strategy']).agg({
                    'aubc': ['mean', 'std', 'count']
                }).round(4)
                f.write(str(summary) + "\n\n")
                
                # Best strategies per dataset
                f.write("\n2. Best Query Strategy per Dataset\n")
                f.write("-"*80 + "\n")
                for dataset in self.aubc_data['dataset'].unique():
                    dataset_data = self.aubc_data[self.aubc_data['dataset'] == dataset]
                    best = dataset_data.groupby('query_strategy')['aubc'].mean().idxmax()
                    best_score = dataset_data.groupby('query_strategy')['aubc'].mean().max()
                    f.write(f"{dataset:15s}: {best:15s} (AUBC={best_score:.4f})\n")
                
            f.write("\n" + "="*80 + "\n")
        
        print(f"Report saved to {output_path}")


def main():
    """Main analysis workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze TabPFN experiment results')
    parser.add_argument('--results_dir', default='../results', help='Results directory')
    parser.add_argument('--experiment', default='TabPFN', help='Experiment name')
    parser.add_argument('--datasets', nargs='+', help='Datasets to analyze')
    parser.add_argument('--output_dir', default='.', help='Output directory for plots')
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(
        results_dir=args.results_dir,
        experiment_name=args.experiment
    )
    
    # Load results
    if not analyzer.load_results(datasets=args.datasets):
        print("Failed to load results!")
        return
    
    # Compute statistics
    analyzer.compute_statistics()
    
    # Create plots
    print("\nGenerating plots...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # AUBC comparison
    analyzer.plot_aubc_comparison(
        datasets=args.datasets,
        save_path=output_dir / 'tabpfn_aubc_comparison.png'
    )
    
    # Learning curves for each dataset
    if analyzer.detail_data is not None:
        for dataset in (args.datasets or analyzer.detail_data['dataset'].unique()):
            analyzer.plot_learning_curves(
                dataset=dataset,
                save_path=output_dir / f'tabpfn_learning_curve_{dataset}.png'
            )
    
    # Statistical comparison
    for dataset in (args.datasets or analyzer.aubc_data['dataset'].unique()):
        analyzer.statistical_comparison(dataset=dataset)
    
    # Generate report
    analyzer.generate_report(output_path=output_dir / 'tabpfn_report.txt')
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
