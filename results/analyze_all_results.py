#!/usr/bin/env python
"""
Comprehensive Analysis Script for TabPFN Active Learning Experiments

This script consolidates results from multiple JSON files and generates:
1. Summary statistics (CSV)
2. AUBC comparisons
3. Statistical significance tests
4. Learning curves data

Output Directory: analysis_YYYYMMDD/
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Configuration
RESULTS_DIR = Path(__file__).parent
OUTPUT_DIR = RESULTS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_all_results():
    """Load and consolidate all JSON result files."""
    all_results = []

    for json_file in sorted(RESULTS_DIR.glob("tabpfn_results_*.json")):
        print(f"Loading {json_file.name}...")
        try:
            with open(json_file) as f:
                data = json.load(f)

            if 'results' in data:
                for r in data['results']:
                    r['source_file'] = json_file.name
                    all_results.append(r)
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")

    return all_results


def extract_aubc_data(results):
    """Extract AUBC values from results."""
    aubc_data = []

    for r in results:
        if not r.get('success', False):
            continue

        # Calculate AUBC from history if available
        if 'history' in r and r['history']:
            hist = r['history']
            accuracies = [h.get('accuracy', h.get('E_tst_score', 0)) for h in hist]
            # AUBC is the area under the accuracy vs. labeled samples curve
            # Normalize by budget
            aubc = np.trapz(accuracies) / len(accuracies) if accuracies else 0
            final_acc = accuracies[-1] if accuracies else 0
        else:
            aubc = r.get('aubc', 0)
            final_acc = r.get('final_accuracy', 0)

        aubc_data.append({
            'dataset': r['dataset'],
            'strategy': r['strategy'],
            'trial': r['trial'],
            'seed': r.get('seed', 0),
            'aubc': aubc,
            'final_accuracy': final_acc,
            'source': r.get('source_file', 'unknown')
        })

    return pd.DataFrame(aubc_data)


def compute_summary_statistics(df):
    """Compute summary statistics by dataset and strategy."""
    summary = df.groupby(['dataset', 'strategy']).agg({
        'aubc': ['mean', 'std', 'count'],
        'final_accuracy': ['mean', 'std']
    }).round(4)

    summary.columns = ['aubc_mean', 'aubc_std', 'n_trials', 'acc_mean', 'acc_std']
    summary = summary.reset_index()

    return summary


def compute_pairwise_comparisons(df):
    """Compute pairwise statistical comparisons between strategies."""
    comparisons = []

    for dataset in df['dataset'].unique():
        ds_data = df[df['dataset'] == dataset]
        strategies = ds_data['strategy'].unique()

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                aubc1 = ds_data[ds_data['strategy'] == s1]['aubc'].values
                aubc2 = ds_data[ds_data['strategy'] == s2]['aubc'].values

                if len(aubc1) >= 2 and len(aubc2) >= 2:
                    # Paired t-test if same length, otherwise Welch's t-test
                    if len(aubc1) == len(aubc2):
                        t_stat, p_value = stats.ttest_rel(aubc1, aubc2)
                    else:
                        t_stat, p_value = stats.ttest_ind(aubc1, aubc2, equal_var=False)

                    # Cohen's d effect size
                    pooled_std = np.sqrt((np.std(aubc1)**2 + np.std(aubc2)**2) / 2)
                    cohens_d = (np.mean(aubc1) - np.mean(aubc2)) / pooled_std if pooled_std > 0 else 0

                    comparisons.append({
                        'dataset': dataset,
                        'strategy_1': s1,
                        'strategy_2': s2,
                        'mean_diff': np.mean(aubc1) - np.mean(aubc2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant_05': p_value < 0.05,
                        'significant_01': p_value < 0.01
                    })

    return pd.DataFrame(comparisons)


def extract_learning_curves(results):
    """Extract learning curve data (accuracy vs labeled samples)."""
    curves = []

    for r in results:
        if not r.get('success', False) or 'history' not in r:
            continue

        for i, h in enumerate(r['history']):
            curves.append({
                'dataset': r['dataset'],
                'strategy': r['strategy'],
                'trial': r['trial'],
                'step': i,
                'labeled_samples': h.get('al_round', 20 + i * 20),  # Estimated if not provided
                'accuracy': h.get('accuracy', h.get('E_tst_score', 0))
            })

    return pd.DataFrame(curves)


def generate_strategy_ranking(summary):
    """Generate strategy rankings by dataset."""
    rankings = []

    for dataset in summary['dataset'].unique():
        ds_data = summary[summary['dataset'] == dataset].copy()
        ds_data = ds_data.sort_values('aubc_mean', ascending=False)
        ds_data['rank'] = range(1, len(ds_data) + 1)

        for _, row in ds_data.iterrows():
            rankings.append({
                'dataset': dataset,
                'strategy': row['strategy'],
                'rank': row['rank'],
                'aubc_mean': row['aubc_mean'],
                'aubc_std': row['aubc_std']
            })

    return pd.DataFrame(rankings)


def main():
    print("=" * 60)
    print("TabPFN Active Learning Results Analysis")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load all results
    print("Step 1: Loading results...")
    results = load_all_results()
    print(f"  Loaded {len(results)} total experiment records")

    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    print(f"  Successful experiments: {len(successful)}")
    print()

    # Extract AUBC data
    print("Step 2: Extracting AUBC data...")
    aubc_df = extract_aubc_data(results)
    print(f"  AUBC data shape: {aubc_df.shape}")

    # Remove duplicates (keep latest by source file)
    aubc_df = aubc_df.sort_values('source').drop_duplicates(
        subset=['dataset', 'strategy', 'trial'], keep='last'
    )
    print(f"  After deduplication: {aubc_df.shape}")

    # Save raw AUBC data
    aubc_file = OUTPUT_DIR / "aubc_all_experiments.csv"
    aubc_df.to_csv(aubc_file, index=False)
    print(f"  Saved: {aubc_file.name}")
    print()

    # Compute summary statistics
    print("Step 3: Computing summary statistics...")
    summary = compute_summary_statistics(aubc_df)

    summary_file = OUTPUT_DIR / "summary_statistics.csv"
    summary.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file.name}")

    # Print summary
    print("\n  Summary Table:")
    print(summary.to_string(index=False))
    print()

    # Compute pairwise comparisons
    print("Step 4: Computing statistical comparisons...")
    comparisons = compute_pairwise_comparisons(aubc_df)

    if not comparisons.empty:
        comp_file = OUTPUT_DIR / "pairwise_comparisons.csv"
        comparisons.to_csv(comp_file, index=False)
        print(f"  Saved: {comp_file.name}")

        # Show significant comparisons
        sig = comparisons[comparisons['significant_05']]
        if not sig.empty:
            print("\n  Significant differences (p < 0.05):")
            for _, row in sig.iterrows():
                diff_dir = ">" if row['mean_diff'] > 0 else "<"
                print(f"    {row['dataset']}: {row['strategy_1']} {diff_dir} {row['strategy_2']} "
                      f"(p={row['p_value']:.4f}, d={row['cohens_d']:.3f})")
    print()

    # Generate rankings
    print("Step 5: Generating strategy rankings...")
    rankings = generate_strategy_ranking(summary)

    rank_file = OUTPUT_DIR / "strategy_rankings.csv"
    rankings.to_csv(rank_file, index=False)
    print(f"  Saved: {rank_file.name}")

    # Print rankings
    print("\n  Rankings by Dataset:")
    for dataset in rankings['dataset'].unique():
        print(f"\n  {dataset}:")
        ds_ranks = rankings[rankings['dataset'] == dataset]
        for _, row in ds_ranks.iterrows():
            print(f"    #{row['rank']}: {row['strategy']} (AUBC: {row['aubc_mean']:.4f} ± {row['aubc_std']:.4f})")
    print()

    # Extract learning curves
    print("Step 6: Extracting learning curves...")
    curves = extract_learning_curves(results)

    if not curves.empty:
        curves_file = OUTPUT_DIR / "learning_curves.csv"
        curves.to_csv(curves_file, index=False)
        print(f"  Saved: {curves_file.name}")
        print(f"  Total data points: {len(curves)}")
    else:
        print("  No learning curve data available in results")
    print()

    # Generate final report
    print("Step 7: Generating final report...")
    report_lines = [
        "# TabPFN Active Learning Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Datasets: {', '.join(sorted(aubc_df['dataset'].unique()))}",
        f"- Strategies: {', '.join(sorted(aubc_df['strategy'].unique()))}",
        "",
        "## AUBC Summary Statistics",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Strategy Rankings",
    ]

    for dataset in rankings['dataset'].unique():
        report_lines.append(f"\n### {dataset}")
        ds_ranks = rankings[rankings['dataset'] == dataset]
        for _, row in ds_ranks.iterrows():
            report_lines.append(f"- #{row['rank']}: **{row['strategy']}** (AUBC: {row['aubc_mean']:.4f} ± {row['aubc_std']:.4f})")

    report_lines.extend([
        "",
        "## Statistical Significance",
    ])

    if not comparisons.empty and not comparisons[comparisons['significant_05']].empty:
        for _, row in comparisons[comparisons['significant_05']].iterrows():
            diff_dir = ">" if row['mean_diff'] > 0 else "<"
            report_lines.append(
                f"- {row['dataset']}: {row['strategy_1']} {diff_dir} {row['strategy_2']} "
                f"(p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f})"
            )
    else:
        report_lines.append("- No statistically significant differences found (p < 0.05)")

    report_file = OUTPUT_DIR / "analysis_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_file.name}")

    print()
    print("=" * 60)
    print("Analysis Complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
