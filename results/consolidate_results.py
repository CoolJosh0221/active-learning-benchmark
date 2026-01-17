#!/usr/bin/env python
"""
Consolidate TabPFN Experiment Results

This script:
1. Collects all AUBC and detail CSV files from src/
2. Consolidates them into structured output
3. Generates comprehensive analysis

Output: results/analysis_YYYYMMDD/
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "results" / f"analysis_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(exist_ok=True)

# Strategy name mapping
STRATEGY_NAMES = {
    'google_us_margin': 'Uncertainty Sampling (US)',
    'uniform': 'Random',
    'skal_coreset': 'Core-Set',
    'skal_bald': 'BALD',
    'alipy_qbc': 'QBC',
    'us_ent': 'Entropy Sampling',
}


def parse_filename(filename):
    """Parse experiment info from filename."""
    # Format: {dataset}-{strategy}-tabpfn-tabpfn-TabPFN-{type}.csv
    parts = filename.replace('.csv', '').split('-')
    if len(parts) >= 5:
        dataset = parts[0]
        strategy = parts[1]
        result_type = parts[-1]  # 'aubc' or 'detail'
        return dataset, strategy, result_type
    return None, None, None


def load_all_aubc_files():
    """Load all AUBC CSV files."""
    aubc_data = []

    for csv_file in SRC_DIR.glob("*-aubc.csv"):
        dataset, strategy, _ = parse_filename(csv_file.name)
        if dataset is None:
            continue

        try:
            df = pd.read_csv(csv_file)
            # Standardize column names
            # Expected format: res_expno, res_trn_score, res_tst_score
            if 'res_tst_score' in df.columns:
                df = df.rename(columns={
                    'res_expno': 'trial',
                    'res_trn_score': 'train_aubc',
                    'res_tst_score': 'AUBC'
                })
            elif 'AUBC' not in df.columns and len(df.columns) >= 2:
                # Try to identify AUBC column (last numeric column usually)
                df.columns = ['trial', 'train_aubc', 'AUBC'][:len(df.columns)]

            df['dataset'] = dataset
            df['strategy_code'] = strategy
            df['strategy'] = STRATEGY_NAMES.get(strategy, strategy)
            aubc_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")

    if aubc_data:
        return pd.concat(aubc_data, ignore_index=True)
    return pd.DataFrame()


def load_all_detail_files():
    """Load all detail CSV files for learning curves.

    Detail files are log-format with lines like:
    [init] seed 0: |0|20|0.645|0.339|
    [update] seed 0: |0|21|0.655|0.000|0.305
    """
    detail_data = []

    for csv_file in SRC_DIR.glob("*-detail.csv"):
        dataset, strategy, _ = parse_filename(csv_file.name)
        if dataset is None:
            continue

        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Parse [init] and [update] lines
                if '[init]' in line or '[update]' in line:
                    # Extract seed and pipe-separated values
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        try:
                            # Parse: seed X: |trial|n_labeled|accuracy|...
                            seed_part = line.split('seed')[1].split(':')[0].strip()
                            seed = int(seed_part) if seed_part.isdigit() else 0
                            trial = int(parts[1]) if parts[1].strip().isdigit() else 0
                            n_labeled = int(parts[2]) if parts[2].strip().isdigit() else 0
                            accuracy = float(parts[3]) if parts[3].strip() else 0

                            detail_data.append({
                                'dataset': dataset,
                                'strategy_code': strategy,
                                'strategy': STRATEGY_NAMES.get(strategy, strategy),
                                'seed': seed,
                                'trial': trial,
                                'n_labeled': n_labeled,
                                'accuracy': accuracy,
                                'line_type': 'init' if '[init]' in line else 'update'
                            })
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")

    if detail_data:
        return pd.DataFrame(detail_data)
    return pd.DataFrame()


def compute_summary_stats(aubc_df):
    """Compute summary statistics."""
    if aubc_df.empty or 'AUBC' not in aubc_df.columns:
        return pd.DataFrame()

    summary = aubc_df.groupby(['dataset', 'strategy']).agg({
        'AUBC': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)

    summary.columns = ['mean', 'std', 'count', 'min', 'max']
    summary = summary.reset_index()

    return summary


def compute_pairwise_tests(aubc_df):
    """Compute pairwise statistical tests."""
    if aubc_df.empty or 'AUBC' not in aubc_df.columns:
        return pd.DataFrame()

    comparisons = []

    for dataset in aubc_df['dataset'].unique():
        ds_data = aubc_df[aubc_df['dataset'] == dataset]
        strategies = ds_data['strategy'].unique()

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                aubc1 = ds_data[ds_data['strategy'] == s1]['AUBC'].values
                aubc2 = ds_data[ds_data['strategy'] == s2]['AUBC'].values

                if len(aubc1) >= 2 and len(aubc2) >= 2:
                    # Paired t-test if same length
                    if len(aubc1) == len(aubc2):
                        t_stat, p_value = stats.ttest_rel(aubc1, aubc2)
                    else:
                        t_stat, p_value = stats.ttest_ind(aubc1, aubc2, equal_var=False)

                    # Cohen's d
                    pooled_std = np.sqrt((np.var(aubc1) + np.var(aubc2)) / 2)
                    cohens_d = (np.mean(aubc1) - np.mean(aubc2)) / pooled_std if pooled_std > 0 else 0

                    comparisons.append({
                        'dataset': dataset,
                        'strategy_1': s1,
                        'strategy_2': s2,
                        'n1': len(aubc1),
                        'n2': len(aubc2),
                        'mean_1': np.mean(aubc1),
                        'mean_2': np.mean(aubc2),
                        'mean_diff': np.mean(aubc1) - np.mean(aubc2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant_05': p_value < 0.05,
                        'significant_01': p_value < 0.01
                    })

    return pd.DataFrame(comparisons)


def compute_rankings(summary):
    """Compute strategy rankings by dataset."""
    if summary.empty:
        return pd.DataFrame()

    rankings = []
    for dataset in summary['dataset'].unique():
        ds_data = summary[summary['dataset'] == dataset].copy()
        ds_data = ds_data.sort_values('mean', ascending=False)

        for rank, (_, row) in enumerate(ds_data.iterrows(), 1):
            rankings.append({
                'dataset': dataset,
                'rank': rank,
                'strategy': row['strategy'],
                'aubc_mean': row['mean'],
                'aubc_std': row['std'],
                'n_trials': row['count']
            })

    return pd.DataFrame(rankings)


def compute_learning_curves(detail_df):
    """Aggregate learning curves by labeled pool size."""
    if detail_df.empty:
        return pd.DataFrame()

    # Check for required columns
    if 'accuracy' not in detail_df.columns or 'n_labeled' not in detail_df.columns:
        print("Warning: Missing required columns (accuracy, n_labeled)")
        return pd.DataFrame()

    # Aggregate by dataset, strategy, and n_labeled
    curves = detail_df.groupby(['dataset', 'strategy', 'n_labeled']).agg({
        'accuracy': ['mean', 'std', 'count']
    }).reset_index()

    curves.columns = ['dataset', 'strategy', 'labeled_samples', 'accuracy_mean', 'accuracy_std', 'n_trials']

    return curves


def generate_report(summary, comparisons, rankings):
    """Generate markdown report."""
    lines = [
        "# TabPFN Active Learning Benchmark Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Executive Summary",
        "",
        "This report presents the results of active learning experiments using TabPFN",
        "on three benchmark datasets: Splice, Ionosphere, and Pol.",
        "",
    ]

    # Overall findings
    if not summary.empty:
        lines.extend([
            "### Key Findings",
            "",
        ])

        for dataset in ['splice', 'ionosphere', 'pol']:
            ds_summary = summary[summary['dataset'] == dataset]
            if ds_summary.empty:
                continue

            best = ds_summary.loc[ds_summary['mean'].idxmax()]
            lines.append(f"- **{dataset.capitalize()}**: {best['strategy']} achieves highest AUBC ({best['mean']:.4f})")

    # Summary table
    if not summary.empty:
        lines.extend([
            "",
            "## AUBC Summary Statistics",
            "",
            "| Dataset | Strategy | Mean | Std | Trials |",
            "|---------|----------|------|-----|--------|",
        ])

        for _, row in summary.sort_values(['dataset', 'mean'], ascending=[True, False]).iterrows():
            lines.append(f"| {row['dataset']} | {row['strategy']} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")

    # Rankings
    if not rankings.empty:
        lines.extend([
            "",
            "## Strategy Rankings by Dataset",
            "",
        ])

        for dataset in rankings['dataset'].unique():
            lines.append(f"### {dataset.capitalize()}")
            lines.append("")
            ds_ranks = rankings[rankings['dataset'] == dataset]
            for _, row in ds_ranks.iterrows():
                lines.append(f"{row['rank']}. **{row['strategy']}** (AUBC: {row['aubc_mean']:.4f} ± {row['aubc_std']:.4f})")
            lines.append("")

    # Statistical comparisons
    if not comparisons.empty:
        sig = comparisons[comparisons['significant_05']]
        lines.extend([
            "## Statistical Significance (p < 0.05)",
            "",
        ])

        if not sig.empty:
            lines.append("| Dataset | Comparison | Mean Diff | p-value | Cohen's d |")
            lines.append("|---------|------------|-----------|---------|-----------|")

            for _, row in sig.iterrows():
                winner = row['strategy_1'] if row['mean_diff'] > 0 else row['strategy_2']
                loser = row['strategy_2'] if row['mean_diff'] > 0 else row['strategy_1']
                lines.append(
                    f"| {row['dataset']} | {winner} > {loser} | "
                    f"{abs(row['mean_diff']):.4f} | {row['p_value']:.4f} | {abs(row['cohens_d']):.3f} |"
                )
        else:
            lines.append("No statistically significant differences found between strategies.")

    lines.extend([
        "",
        "## Methodology",
        "",
        "- **Model**: TabPFN (Tabular Prior-Fitted Network)",
        "- **Initial labeled pool**: 20 samples",
        "- **Budget**: 200 queries",
        "- **Metric**: AUBC (Area Under Budget Curve)",
        "- **Statistical tests**: Paired t-test, Cohen's d effect size",
        "",
        "## Files Generated",
        "",
        "- `aubc_consolidated.csv`: All AUBC values per trial",
        "- `summary_statistics.csv`: Summary statistics by dataset/strategy",
        "- `pairwise_comparisons.csv`: Statistical test results",
        "- `strategy_rankings.csv`: Rankings by dataset",
        "- `learning_curves.csv`: Accuracy vs labeled samples (if available)",
        "",
    ])

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TabPFN Active Learning Results Consolidation")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load AUBC files
    print("Step 1: Loading AUBC CSV files...")
    aubc_df = load_all_aubc_files()
    print(f"  Loaded {len(aubc_df)} AUBC records")

    if not aubc_df.empty:
        aubc_file = OUTPUT_DIR / "aubc_consolidated.csv"
        aubc_df.to_csv(aubc_file, index=False)
        print(f"  Saved: {aubc_file.name}")

    # Load detail files
    print("\nStep 2: Loading detail CSV files...")
    detail_df = load_all_detail_files()
    print(f"  Loaded {len(detail_df)} detail records")

    if not detail_df.empty:
        detail_file = OUTPUT_DIR / "detail_consolidated.csv"
        detail_df.to_csv(detail_file, index=False)
        print(f"  Saved: {detail_file.name}")

    # Compute summary statistics
    print("\nStep 3: Computing summary statistics...")
    summary = compute_summary_stats(aubc_df)

    if not summary.empty:
        summary_file = OUTPUT_DIR / "summary_statistics.csv"
        summary.to_csv(summary_file, index=False)
        print(f"  Saved: {summary_file.name}")
        print("\n  Summary:")
        print(summary.to_string(index=False))

    # Compute pairwise comparisons
    print("\nStep 4: Computing statistical comparisons...")
    comparisons = compute_pairwise_tests(aubc_df)

    if not comparisons.empty:
        comp_file = OUTPUT_DIR / "pairwise_comparisons.csv"
        comparisons.to_csv(comp_file, index=False)
        print(f"  Saved: {comp_file.name}")

        sig = comparisons[comparisons['significant_05']]
        if not sig.empty:
            print("\n  Significant differences (p < 0.05):")
            for _, row in sig.iterrows():
                winner = row['strategy_1'] if row['mean_diff'] > 0 else row['strategy_2']
                loser = row['strategy_2'] if row['mean_diff'] > 0 else row['strategy_1']
                print(f"    {row['dataset']}: {winner} > {loser} (p={row['p_value']:.4f}, d={abs(row['cohens_d']):.3f})")

    # Compute rankings
    print("\nStep 5: Computing strategy rankings...")
    rankings = compute_rankings(summary)

    if not rankings.empty:
        rank_file = OUTPUT_DIR / "strategy_rankings.csv"
        rankings.to_csv(rank_file, index=False)
        print(f"  Saved: {rank_file.name}")

        print("\n  Rankings:")
        for dataset in rankings['dataset'].unique():
            print(f"\n  {dataset.capitalize()}:")
            ds_ranks = rankings[rankings['dataset'] == dataset]
            for _, row in ds_ranks.iterrows():
                print(f"    #{row['rank']}: {row['strategy']} ({row['aubc_mean']:.4f})")

    # Compute learning curves
    print("\nStep 6: Computing learning curves...")
    curves = compute_learning_curves(detail_df)

    if not curves.empty:
        curves_file = OUTPUT_DIR / "learning_curves.csv"
        curves.to_csv(curves_file, index=False)
        print(f"  Saved: {curves_file.name}")

    # Generate report
    print("\nStep 7: Generating report...")
    report = generate_report(summary, comparisons, rankings)
    report_file = OUTPUT_DIR / "final_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_file.name}")

    print("\n" + "=" * 70)
    print("Consolidation Complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
