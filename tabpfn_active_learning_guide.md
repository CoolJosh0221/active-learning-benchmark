# TabPFN Active Learning Reproduction Guide

## Overview

This guide will help you reproduce the active learning benchmark experiments using TabPFN on the top 3 datasets: **Splice**, **Ionosphere**, and **Pol**.

## Step 1: Environment Setup

### 1.1 Clone and Setup Repository

```bash
# Clone the benchmark repository
git clone https://github.com/ariapoy/active-learning-benchmark.git
cd active-learning-benchmark

# Create virtual environment
python3 -m venv act-env
source act-env/bin/activate

# Install base requirements
pip install -r requirements.txt
```

### 1.2 Install TabPFN

```bash
# Install TabPFN
pip install tabpfn

# Or install from source for latest version
# git clone https://github.com/automl/TabPFN.git
# cd TabPFN
# pip install -e .
```

### 1.3 Install Additional Dependencies

```bash
# Clone required active learning libraries
git clone https://github.com/ariapoy/active-learning.git
git clone https://github.com/ariapoy/ALiPy.git alipy-dev
cp -r alipy-dev/alipy alipy-dev/alipy_dev

git clone https://github.com/ariapoy/libact.git libact-dev
cd libact-dev
python setup.py build
python setup.py install
cd ..
cp -r libact-dev/libact libact-dev/libact_dev
```

### 1.4 Download Datasets

```bash
cd data
bash get_data_zhan21.sh  # This downloads all datasets
cd ..
```

## Step 2: Modify Code for TabPFN

### 2.1 Create TabPFN Model Wrapper

Create a new file: `src/models/tabpfn_model.py`

```python
"""
TabPFN model wrapper for active learning benchmark
"""
from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for TabPFN to work with the active learning benchmark.
    TabPFN has specific limitations:
    - Max 10,000 training samples
    - Max 100 features
    - Binary and multi-class classification
    """
    
    def __init__(self, N_ensemble_configurations=32, device='cpu', **kwargs):
        """
        Args:
            N_ensemble_configurations: Number of ensemble members (default: 32)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.N_ensemble_configurations = N_ensemble_configurations
        self.device = device
        self.kwargs = kwargs
        self.model = None
        
    def fit(self, X, y):
        """Fit TabPFN model"""
        # Check constraints
        if X.shape[0] > 10000:
            print(f"Warning: TabPFN supports max 10,000 samples. Got {X.shape[0]}. Using first 10,000.")
            X = X[:10000]
            y = y[:10000]
            
        if X.shape[1] > 100:
            print(f"Warning: TabPFN supports max 100 features. Got {X.shape[1]}. Using first 100.")
            X = X[:, :100]
        
        self.model = TabPFNClassifier(
            N_ensemble_configurations=self.N_ensemble_configurations,
            device=self.device,
            **self.kwargs
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if X.shape[1] > 100:
            X = X[:, :100]
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if X.shape[1] > 100:
            X = X[:, :100]
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Return accuracy score"""
        if X.shape[1] > 100:
            X = X[:, :100]
        return self.model.score(X, y)
    
    @property
    def classes_(self):
        """Return class labels"""
        return self.model.classes_ if self.model is not None else None
```

### 2.2 Update Config File

Modify `src/config.py` to add TabPFN:

```python
# Add this import at the top
from models.tabpfn_model import TabPFNWrapper

# Add TabPFN to the model dictionary
MODELS = {
    # ... existing models ...
    'tabpfn': TabPFNWrapper,
    'tabpfn_gpu': lambda: TabPFNWrapper(device='cuda'),
}

# Add TabPFN configurations for different query strategies
TABPFN_CONFIGS = {
    'US': {  # Uncertainty Sampling
        'model': 'tabpfn',
        'query_oriented_model': 'tabpfn',
        'task_oriented_model': 'tabpfn',
    },
    'QBC': {  # Query By Committee
        'model': 'tabpfn',
        'query_oriented_model': 'tabpfn',
        'task_oriented_model': 'tabpfn',
    },
    'BALD': {  # Bayesian Active Learning by Disagreement
        'model': 'tabpfn',
        'query_oriented_model': 'tabpfn',
        'task_oriented_model': 'tabpfn',
    },
    'Core-Set': {
        'model': 'tabpfn',
        'task_oriented_model': 'tabpfn',
    },
    'Random': {  # Random sampling baseline
        'model': 'tabpfn',
        'task_oriented_model': 'tabpfn',
    }
}
```

## Step 3: Create Experiment Script

Create `src/run_tabpfn_experiments.py`:

```python
#!/usr/bin/env python3
"""
Run TabPFN experiments on top 3 datasets
"""
import os
import sys
import subprocess
from pathlib import Path

# Top 3 datasets based on US superiority analysis
DATASETS = ['splice', 'ionosphere', 'pol']

# Query strategies to test
QUERY_STRATEGIES = [
    ('US', 'margin-zhan', 'google-zhan'),      # Uncertainty Sampling
    ('Random', 'random', 'random'),             # Random baseline
    ('QBC', 'qbc', 'google-zhan'),             # Query By Committee
    ('BALD', 'bald', 'google-zhan'),           # BALD (if available)
    ('Core-Set', 'core-set', 'google-zhan'),   # Core-Set
]

# Experimental settings
N_TRIALS = 10  # Number of independent trials
SEED_START = 0
INIT_LBL_SIZE = 20  # Initial labeled pool size
TST_SIZE = 0.4  # Test set size (40%)
BUDGET = 200  # Total labeling budget

def run_experiment(dataset, qs_display_name, qs_name, hs_name, seed):
    """
    Run a single experiment
    
    Args:
        dataset: Dataset name
        qs_display_name: Display name for query strategy
        qs_name: Query strategy implementation name
        hs_name: Hypothesis space name
        seed: Random seed
    """
    cmd = [
        'python', 'main.py',
        '--data_set', dataset,
        '--tool', 'google',  # or 'libact' depending on implementation
        '--qs_name', qs_name,
        '--hs_name', hs_name,
        '--gs_name', 'zhan',
        '--model', 'tabpfn',  # Use TabPFN
        '--seed', str(seed),
        '--n_trials', '1',
        '--init_lbl_size', str(INIT_LBL_SIZE),
        '--tst_size', str(TST_SIZE),
        '--budget', str(BUDGET),
        '--exp_name', 'TabPFN',  # Experiment name for output files
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {qs_display_name} on {dataset} (seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {qs_display_name} on {dataset} (seed={seed})")
        return True

def main():
    """Run all experiments"""
    print("="*80)
    print("TabPFN Active Learning Experiments")
    print("Datasets: Splice, Ionosphere, Pol")
    print(f"Trials per experiment: {N_TRIALS}")
    print("="*80)
    
    results = []
    
    for dataset in DATASETS:
        for qs_display_name, qs_name, hs_name in QUERY_STRATEGIES:
            for trial in range(N_TRIALS):
                seed = SEED_START + trial
                success = run_experiment(
                    dataset, qs_display_name, qs_name, hs_name, seed
                )
                results.append({
                    'dataset': dataset,
                    'query_strategy': qs_display_name,
                    'trial': trial,
                    'seed': seed,
                    'success': success
                })
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['dataset']}, {r['query_strategy']}, trial {r['trial']}")

if __name__ == '__main__':
    main()
```

### 3.1 Make the Script Executable

```bash
chmod +x src/run_tabpfn_experiments.py
```

## Step 4: Run Experiments

### 4.1 Quick Test (Single Experiment)

```bash
cd src
python main.py \
    --data_set splice \
    --tool google \
    --qs_name margin-zhan \
    --hs_name google-zhan \
    --gs_name zhan \
    --model tabpfn \
    --seed 0 \
    --n_trials 1 \
    --init_lbl_size 20 \
    --tst_size 0.4
```

### 4.2 Run Full Experiments

```bash
cd src
python run_tabpfn_experiments.py
```

### 4.3 Parallel Execution (Optional)

For faster execution, you can run experiments in parallel:

```bash
# Run different datasets in parallel
cd src

# Terminal 1
python run_tabpfn_experiments.py --dataset splice &

# Terminal 2
python run_tabpfn_experiments.py --dataset ionosphere &

# Terminal 3
python run_tabpfn_experiments.py --dataset pol &
```

## Step 5: Analysis and Results

### 5.1 Locate Results

Results will be saved in the `results/` directory:

- `*-aubc.csv`: Area Under Budget Curve (main metric)
- `*-detail.csv`: Detailed performance at each iteration

### 5.2 Analyze Results

```bash
cd results
python analysis.py
# Or use the Jupyter notebook
jupyter notebook analysis.ipynb
```

### 5.3 Create Custom Analysis Script

Create `results/analyze_tabpfn.py`:

```python
"""
Analyze TabPFN experimental results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(dataset_name, experiment_name='TabPFN'):
    """Load AUBC results for a dataset"""
    pattern = f"*{dataset_name}*{experiment_name}*aubc.csv"
    files = list(Path('.').glob(pattern))
    
    if not files:
        print(f"No results found for {dataset_name}")
        return None
    
    # Load and combine all result files
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def plot_comparison(datasets=['splice', 'ionosphere', 'pol']):
    """Plot performance comparison across datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(datasets):
        df = load_results(dataset)
        if df is None:
            continue
        
        # Group by query strategy and calculate mean AUBC
        summary = df.groupby('query_strategy')['aubc'].agg(['mean', 'std'])
        summary = summary.sort_values('mean', ascending=False)
        
        # Plot
        ax = axes[idx]
        summary['mean'].plot(kind='bar', ax=ax, yerr=summary['std'], capsize=5)
        ax.set_title(f'{dataset.capitalize()}')
        ax.set_xlabel('Query Strategy')
        ax.set_ylabel('AUBC')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tabpfn_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot to tabpfn_comparison.png")

def print_summary(datasets=['splice', 'ionosphere', 'pol']):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("TabPFN EXPERIMENTAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for dataset in datasets:
        df = load_results(dataset)
        if df is None:
            continue
        
        print(f"\n{dataset.upper()}")
        print("-" * 40)
        
        summary = df.groupby('query_strategy')['aubc'].agg(['mean', 'std', 'count'])
        summary = summary.sort_values('mean', ascending=False)
        
        print(summary.to_string())
        
        # Find best strategy
        best_strategy = summary.index[0]
        best_mean = summary.loc[best_strategy, 'mean']
        
        print(f"\nBest Strategy: {best_strategy}")
        print(f"Mean AUBC: {best_mean:.4f} ± {summary.loc[best_strategy, 'std']:.4f}")

if __name__ == '__main__':
    print_summary()
    plot_comparison()
```

## Step 6: Troubleshooting

### Common Issues and Solutions

#### Issue 1: TabPFN Training Sample Limit

**Problem**: Dataset has > 10,000 samples
**Solution**: The wrapper automatically truncates to 10,000 samples. Consider:

- Using a smaller initial pool
- Sampling from the unlabeled pool

#### Issue 2: TabPFN Feature Limit

**Problem**: Dataset has > 100 features
**Solution**: Apply feature selection before using TabPFN:

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X, y)
```

#### Issue 3: GPU Memory Issues

**Problem**: CUDA out of memory
**Solution**: Use CPU mode or reduce ensemble size:

```python
# Use CPU
model = TabPFNWrapper(device='cpu')

# Or reduce ensemble configurations
model = TabPFNWrapper(N_ensemble_configurations=16, device='cuda')
```

#### Issue 4: Import Errors

**Problem**: Cannot import libact or alipy
**Solution**: Ensure paths are correct:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libact-dev:$(pwd)/alipy-dev"
```

## Step 7: Expected Results

Based on your variance analysis, you should expect:

### Splice (Best Dataset)

- **Variance**: 6.26 (good consistency)
- **Expected US advantage**: +3.93% over median
- TabPFN should perform well due to stable patterns

### Ionosphere

- **Variance**: 11.00 (moderate consistency)
- **Expected US advantage**: +3.24% over median
- Good test case for TabPFN's uncertainty estimates

### Pol

- **Variance**: 16.07 (moderate consistency)
- **Expected US advantage**: +2.84% over median
- High absolute performance (~98% US score)

## Step 8: Next Steps

### Extend the Experiments

1. **Add more query strategies**:
   - DWUS (Density Weighted Uncertainty Sampling)
   - ALBL (Active Learning By Learning)
   - LAL (Learning Active Learning)

2. **Compare with baseline models**:
   - Run same experiments with SVM (original paper)
   - Compare TabPFN vs SVM performance

3. **Hyperparameter tuning**:
   - Vary `N_ensemble_configurations`
   - Test different initial pool sizes
   - Adjust labeling budget

4. **Statistical analysis**:
   - Perform paired t-tests between strategies
   - Calculate confidence intervals
   - Create learning curves

## Additional Resources

- TabPFN Paper: <https://arxiv.org/abs/2207.01848>
- TabPFN GitHub: <https://github.com/automl/TabPFN>
- Original AL Benchmark: <https://arxiv.org/abs/2306.08954>
- LibAct Documentation: <https://github.com/ntucllab/libact>

## Notes

1. TabPFN is pre-trained and doesn't require training, making it fast
2. TabPFN works best on small-to-medium tabular datasets
3. The model provides well-calibrated uncertainty estimates
4. Results may differ from SVM due to different model characteristics
