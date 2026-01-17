# TabPFN Active Learning Benchmark Results
Generated: 2026-01-17 23:59

## Executive Summary

This report presents the results of active learning experiments using TabPFN
on three benchmark datasets: Splice, Ionosphere, and Pol.

### Key Findings

- **Splice**: Uncertainty Sampling (US) achieves highest AUBC (0.7704)
- **Ionosphere**: BALD achieves highest AUBC (0.9358)
- **Pol**: BALD achieves highest AUBC (0.9022)

## AUBC Summary Statistics

| Dataset | Strategy | Mean | Std | Trials |
|---------|----------|------|-----|--------|
| ionosphere | BALD | 0.9358 | 0.0171 | 10 |
| ionosphere | Uncertainty Sampling (US) | 0.9347 | 0.0167 | 10 |
| ionosphere | Core-Set | 0.9240 | 0.0132 | 10 |
| ionosphere | Random | 0.9165 | 0.0175 | 10 |
| pol | BALD | 0.9022 | 0.0088 | 10 |
| pol | Uncertainty Sampling (US) | 0.9017 | 0.0068 | 10 |
| pol | Random | 0.8638 | 0.0142 | 10 |
| pol | Core-Set | 0.7880 | 0.0515 | 10 |
| splice | Uncertainty Sampling (US) | 0.7704 | 0.0188 | 20 |
| splice | BALD | 0.7637 | 0.0147 | 10 |
| splice | Random | 0.7418 | 0.0224 | 20 |
| splice | Core-Set | 0.6899 | 0.0482 | 20 |

## Strategy Rankings by Dataset

### Ionosphere

1. **BALD** (AUBC: 0.9358 ± 0.0171)
2. **Uncertainty Sampling (US)** (AUBC: 0.9347 ± 0.0167)
3. **Core-Set** (AUBC: 0.9240 ± 0.0132)
4. **Random** (AUBC: 0.9165 ± 0.0175)

### Pol

1. **BALD** (AUBC: 0.9022 ± 0.0088)
2. **Uncertainty Sampling (US)** (AUBC: 0.9017 ± 0.0068)
3. **Random** (AUBC: 0.8638 ± 0.0142)
4. **Core-Set** (AUBC: 0.7880 ± 0.0515)

### Splice

1. **Uncertainty Sampling (US)** (AUBC: 0.7704 ± 0.0188)
2. **BALD** (AUBC: 0.7637 ± 0.0147)
3. **Random** (AUBC: 0.7418 ± 0.0224)
4. **Core-Set** (AUBC: 0.6899 ± 0.0482)

## Statistical Significance (p < 0.05)

| Dataset | Comparison | Mean Diff | p-value | Cohen's d |
|---------|------------|-----------|---------|-----------|
| pol | BALD > Core-Set | 0.1142 | 0.0001 | 3.258 |
| pol | BALD > Random | 0.0385 | 0.0000 | 3.432 |
| pol | Uncertainty Sampling (US) > Core-Set | 0.1137 | 0.0001 | 3.263 |
| pol | Random > Core-Set | 0.0758 | 0.0005 | 2.114 |
| pol | Uncertainty Sampling (US) > Random | 0.0380 | 0.0000 | 3.600 |
| ionosphere | Core-Set > Random | 0.0076 | 0.0331 | 0.514 |
| ionosphere | BALD > Random | 0.0193 | 0.0000 | 1.174 |
| ionosphere | Uncertainty Sampling (US) > Random | 0.0182 | 0.0001 | 1.122 |
| ionosphere | BALD > Core-Set | 0.0117 | 0.0005 | 0.808 |
| ionosphere | Uncertainty Sampling (US) > Core-Set | 0.0107 | 0.0049 | 0.746 |
| splice | Random > Core-Set | 0.0518 | 0.0000 | 1.415 |
| splice | BALD > Core-Set | 0.0737 | 0.0000 | 2.128 |
| splice | Uncertainty Sampling (US) > Core-Set | 0.0805 | 0.0000 | 2.257 |
| splice | BALD > Random | 0.0219 | 0.0036 | 1.195 |
| splice | Uncertainty Sampling (US) > Random | 0.0286 | 0.0000 | 1.420 |

## Methodology

- **Model**: TabPFN (Tabular Prior-Fitted Network)
- **Initial labeled pool**: 20 samples
- **Budget**: 200 queries
- **Metric**: AUBC (Area Under Budget Curve)
- **Statistical tests**: Paired t-test, Cohen's d effect size

## Files Generated

- `aubc_consolidated.csv`: All AUBC values per trial
- `summary_statistics.csv`: Summary statistics by dataset/strategy
- `pairwise_comparisons.csv`: Statistical test results
- `strategy_rankings.csv`: Rankings by dataset
- `learning_curves.csv`: Accuracy vs labeled samples (if available)
