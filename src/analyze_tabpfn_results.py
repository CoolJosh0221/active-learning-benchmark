#!/usr/bin/env python3
"""
Analyze TabPFN Active Learning Results
Generates summary statistics and plots from experiment CSV files.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Map internal strategy names to display names
STRATEGY_MAP = {
    'google_us_margin': 'Uncertainty Sampling (US)',
    'uniform': 'Random',
    'skal_coreset': 'Core-Set',
    'us_ent': 'Uncertainty Sampling (Entropy)',
    'dwus': 'Density-Weighted US',
    'kcenter': 'Core-Set (k-Center)'
}

def load_results(datasets, results_dir='.'):
    """Load all result files for specified datasets"""
    data = []
    
    for dataset in datasets:
        # Find all AUBC files for this dataset
        pattern = os.path.join(results_dir, f"{dataset}-*-aubc.csv")
        files = glob.glob(pattern)
        
        for file_path in files:
            # Parse filename to get strategy
            # Format: {dataset}-{strategy}-tabpfn-tabpfn-TabPFN-aubc.csv
            filename = os.path.basename(file_path)
            parts = filename.split('-tabpfn-')[0].split('-')
            
            # Extract strategy name (everything after dataset)
            strategy_key = '-'.join(parts[1:])
            
            # Load data
            try:
                df = pd.read_csv(file_path)
                df['dataset'] = dataset
                df['strategy_key'] = strategy_key
                df['strategy'] = STRATEGY_MAP.get(strategy_key, strategy_key)
                
                data.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)

def analyze_aubc(df):
    """Compute AUBC statistics"""
    stats = df.groupby(['dataset', 'strategy'])['res_tst_score'].agg(
        ['count', 'mean', 'std', 'min', 'max']
    ).reset_index()
    
    stats = stats.sort_values(['dataset', 'mean'], ascending=[True, False])
    return stats

def plot_aubc_comparison(df, output_dir):
    """Generate AUBC comparison boxplots"""
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        dataset_df = df[df['dataset'] == dataset]
        
        sns.boxplot(data=dataset_df, x='strategy', y='res_tst_score')
        plt.title(f'AUBC Distribution - {dataset.upper()}')
        plt.ylabel('Area Under Budget Curve (Test)')
        plt.xlabel('Query Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        outfile = os.path.join(output_dir, f'{dataset}_aubc_boxplot.png')
        plt.savefig(outfile)
        plt.close()
        print(f"Saved plot: {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Analyze TabPFN Results")
    parser.add_argument('--datasets', nargs='+', default=['splice', 'ionosphere', 'pol'],
                        help='Datasets to analyze')
    parser.add_argument('--dir', default='.', help='Directory containing results CSVs')
    parser.add_argument('--out', default='../results/analysis', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Loading results from {args.dir}...")
    df = load_results(args.datasets, args.dir)
    
    if df.empty:
        print("No results found matching the criteria.")
        return
        
    print("\nAUBC Summary Statistics:")
    print("-" * 80)
    stats = analyze_aubc(df)
    print(stats.to_string(index=False))
    print("-" * 80)
    
    # Save summary to CSV
    stats_file = os.path.join(args.out, 'aubc_summary.csv')
    stats.to_csv(stats_file, index=False)
    print(f"\nSummary saved to {stats_file}")
    
    print("\nGenerating plots...")
    plot_aubc_comparison(df, args.out)

if __name__ == "__main__":
    main()
