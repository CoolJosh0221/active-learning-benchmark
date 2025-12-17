# Project Context for Claude

## Project Overview

This is an **Active Learning Benchmark** project focused on reproducing and extending experiments from the paper ["An Expanded Benchmark that Rediscovers and Affirms the Edge of Uncertainty Sampling for Active Learning in Tabular Datasets"](https://openreview.net/pdf?id=855yo1Ubt2).

### Current Goal
Reproduce active learning experiments using **TabPFN** (Tabular Prior-Fitted Networks) on the **top 3 datasets** where Uncertainty Sampling (US) shows the strongest performance advantage:
1. **Splice** - Variance: 6.26, US advantage: +3.93% over median
2. **Ionosphere** - Variance: 11.00, US advantage: +3.24% over median  
3. **Pol** - Variance: 16.07, US advantage: +2.84% over median

### What Makes This Project Special
- Original paper used SVM (RBF kernel) as the base model
- We're extending it with **TabPFN**, a modern neural network pre-trained on synthetic tabular data
- Focus on datasets where US performs consistently well (low variance + high advantage)

## Repository Structure

```
active-learning-benchmark/
├── data/                          # Dataset storage
│   ├── get_data_zhan21.sh        # Download script for all datasets
│   └── [dataset files]           # LIBSVM format files
├── src/                          # Source code
│   ├── main.py                   # Main experiment runner (original)
│   ├── config.py                 # Configuration and model definitions
│   ├── models/                   # Model implementations
│   │   └── tabpfn_model.py      # TabPFN wrapper (NEW)
│   ├── run_tabpfn_experiments.py # Automated TabPFN experiments (NEW)
│   └── [query strategies]        # Various AL query strategy implementations
├── results/                      # Experiment results
│   ├── *-aubc.csv               # Area Under Budget Curve results
│   ├── *-detail.csv             # Detailed iteration results
│   └── analyze_tabpfn_results.py # Results analysis script (NEW)
├── libact-dev/                   # Active learning library (cloned)
├── alipy-dev/                    # Another AL library (cloned)
└── requirements.txt              # Python dependencies
```

## Key Files Created for This Project

### 1. `tabpfn_model.py` (src/models/)
**Purpose**: Wrapper to make TabPFN compatible with the AL benchmark framework

**Key Features**:
- Handles TabPFN constraints (max 10K samples, max 100 features)
- Auto-truncates when limits exceeded
- Implements sklearn's estimator interface
- Provides fast/accurate variants

**Important Classes**:
```python
TabPFNWrapper(N_ensemble_configurations=32, device='cpu')
TabPFNFast()  # 16 ensemble members
TabPFNAccurate()  # 64 ensemble members
```

### 2. `run_tabpfn_experiments.py` (src/)
**Purpose**: Automates running multiple experiments across datasets and query strategies

**Features**:
- Runs experiments on Splice, Ionosphere, Pol
- Tests multiple query strategies: US, Random, QBC, BALD, Core-Set
- Configurable trials, budget, initial pool size
- Comprehensive logging
- Quick test mode for validation

**Usage**:
```bash
# Quick test
python run_tabpfn_experiments.py --quick_test

# Full run
python run_tabpfn_experiments.py --trials 10

# Specific dataset
python run_tabpfn_experiments.py --datasets splice --strategies US Random
```

### 3. `analyze_tabpfn_results.py` (results/)
**Purpose**: Analyze and visualize experimental results

**Features**:
- Loads AUBC and detail CSV files
- Computes summary statistics
- Creates comparison plots
- Generates learning curves
- Performs statistical tests (paired t-tests, Cohen's d)
- Exports comprehensive reports

## Experimental Setup

### Core Settings
```python
init_lbl_size = 20      # Initial labeled pool size
tst_size = 0.4          # Test set size (40%)
budget = 200            # Total labeling budget
n_trials = 10           # Independent runs for statistical significance
```

### Query Strategies to Test
- **US** (Uncertainty Sampling) - The baseline that should perform best
- **Random** - Random sampling baseline
- **QBC** (Query By Committee) - Ensemble disagreement
- **BALD** (Bayesian Active Learning by Disagreement)
- **Core-Set** - Geometric diversity sampling

### Performance Metric
**AUBC** (Area Under Budget Curve): Integral of accuracy over labeling budget
- Higher = better
- More robust than final accuracy alone
- Captures both final performance and data efficiency

## TabPFN Specifics

### What is TabPFN?
- Pre-trained neural network for tabular classification
- Trained on 100K+ synthetic tabular datasets
- No hyperparameter tuning needed
- Fast inference (~0.1s per prediction)
- Well-calibrated uncertainty estimates

### Constraints (CRITICAL)
```python
MAX_SAMPLES = 10,000    # Hard limit
MAX_FEATURES = 100      # Hard limit
```
Our wrapper handles these automatically via truncation.

### Advantages for Active Learning
1. ✅ No training time (pre-trained)
2. ✅ Excellent uncertainty quantification
3. ✅ Strong performance on small datasets
4. ✅ No hyperparameter sensitivity

### Configuration Options
```python
# Standard (balanced)
TabPFNWrapper(N_ensemble_configurations=32, device='cpu')

# Fast (for quick testing)
TabPFNWrapper(N_ensemble_configurations=16, device='cpu')

# Accurate (for final results)
TabPFNWrapper(N_ensemble_configurations=64, device='cuda')
```

## Common Tasks

### Task 1: Add TabPFN Support to Existing Codebase

1. Copy `tabpfn_model.py` to `src/models/`
2. Update `src/config.py`:
```python
from models.tabpfn_model import TabPFNWrapper

MODELS = {
    # ... existing models ...
    'tabpfn': TabPFNWrapper,
}
```

### Task 2: Run Single Experiment
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
    --n_trials 1
```

### Task 3: Run All Experiments
```bash
cd src
python run_tabpfn_experiments.py \
    --datasets splice ionosphere pol \
    --strategies US Random QBC Core-Set \
    --trials 10 \
    --budget 200
```

### Task 4: Analyze Results
```bash
cd results
python analyze_tabpfn_results.py \
    --datasets splice ionosphere pol \
    --output_dir ./tabpfn_analysis
```

### Task 5: Compare TabPFN vs SVM
```bash
# Run with SVM (original paper)
python main.py --model svm --kernel rbf ...

# Run with TabPFN
python main.py --model tabpfn ...

# Compare in analysis
python analyze_tabpfn_results.py --compare_models svm tabpfn
```

## Expected Results

Based on variance analysis, we expect:

### Splice (Best Case)
- US should significantly outperform Random (+3-4%)
- Low variance across trials (consistent)
- Final accuracy ~92-93%

### Ionosphere (Medium Case)
- US maintains advantage (+2-3%)
- Moderate variance
- Final accuracy ~86-87%

### Pol (High Performance Case)
- Very high accuracy (~98%)
- US still shows advantage (+2-3%)
- Moderate variance

## Integration Points

### Where TabPFN Fits
```
Query Strategy → Model Training → Uncertainty Estimation → Sample Selection
                      ↓
                   TabPFN
                      ↓
              (no hyperparameters)
                      ↓
              predict_proba()
```

### How Active Learning Works Here
1. Start with small labeled pool (20 samples)
2. Train model on labeled data
3. Query strategy selects most informative unlabeled samples
4. Add to labeled pool, retrain
5. Repeat until budget exhausted
6. Evaluate on held-out test set

## Debugging Tips

### Issue: TabPFN not found
```bash
pip install tabpfn
```

### Issue: Dataset too large
```python
# Check dataset size
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# TabPFN auto-truncates, but you can also:
from sklearn.feature_selection import SelectKBest
X_selected = SelectKBest(k=100).fit_transform(X, y)
```

### Issue: CUDA out of memory
```python
# Switch to CPU
model = TabPFNWrapper(device='cpu')

# Or reduce ensemble size
model = TabPFNWrapper(N_ensemble_configurations=16, device='cuda')
```

### Issue: Import errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libact-dev:$(pwd)/alipy-dev"
```

### Issue: Results not found
```bash
# Check results directory
ls -la results/*TabPFN*

# Verify experiment name matches
grep "exp_name" src/run_tabpfn_experiments.py
```

## Data Format

### Input Format (LIBSVM)
```
+1 1:0.5 2:0.3 3:0.8
-1 1:0.2 2:0.9 3:0.1
```

### Output Format (AUBC CSV)
```csv
dataset,query_strategy,seed,trial,aubc,final_accuracy
splice,US,0,0,0.8523,0.9245
splice,Random,0,0,0.8123,0.8956
```

### Output Format (Detail CSV)
```csv
dataset,query_strategy,seed,trial,labeled_size,accuracy,query_time
splice,US,0,0,20,0.7234,0.123
splice,US,0,0,40,0.7856,0.145
```

## Research Context

### Original Paper Findings
- Uncertainty Sampling consistently strong across 28 datasets
- Simple methods often outperform complex ones
- Variance matters: some datasets have high method variability

### Our Extension
- Does TabPFN maintain US superiority?
- Is TabPFN's uncertainty estimation better than SVM?
- Can we identify datasets where TabPFN excels?

### Hypotheses to Test
1. ✅ US should remain best on these 3 datasets
2. ❓ TabPFN might improve over SVM due to better calibration
3. ❓ Lower variance possible due to no hyperparameter tuning

## Next Steps

### Immediate Tasks
1. ✅ Setup repository structure
2. ✅ Create TabPFN wrapper
3. ✅ Create experiment runner
4. ⏳ Run experiments on all 3 datasets
5. ⏳ Analyze results and compare with paper

### Future Extensions
1. Add more query strategies (DWUS, LAL, ALBL)
2. Test on additional datasets
3. Compare TabPFN vs other models (RF, GBDT)
4. Hyperparameter sensitivity analysis
5. Write up findings

## References

- TabPFN Paper: https://arxiv.org/abs/2207.01848
- TabPFN Code: https://github.com/automl/TabPFN
- AL Benchmark Paper: https://openreview.net/pdf?id=855yo1Ubt2
- Original Repo: https://github.com/ariapoy/active-learning-benchmark
- IJCAI 2021 Survey: https://ijcai-21.org/program-survey/

## Notes for Claude

- User (Cartier) is a high school student with strong competitive programming background
- Familiar with algorithms and optimization
- Working on AP Seminar research
- Interested in active learning and ML applications
- Prefers concise, technical explanations
- Values reproducibility and well-documented code

When helping with this project:
1. Be precise about technical details
2. Show code examples
3. Explain statistical concepts clearly
4. Suggest optimizations where relevant
5. Point out potential issues proactively
