# Project Context for Gemini

## Project Overview

**Active Learning Benchmark with TabPFN**

This project extends the active learning benchmark from the paper ["An Expanded Benchmark that Rediscovers and Affirms the Edge of Uncertainty Sampling for Active Learning in Tabular Datasets"](https://openreview.net/pdf?id=855yo1Ubt2).

### Mission
Reproduce active learning experiments using **TabPFN** on the top 3 datasets where Uncertainty Sampling (US) demonstrates the strongest and most consistent performance.

### Target Datasets
1. **Splice** (Priority 1)
   - Variance: 6.26 (HIGH consistency)
   - US advantage: +3.93% over median
   - Best overall choice

2. **Ionosphere** (Priority 2)
   - Variance: 11.00 (MODERATE consistency)
   - US advantage: +3.24% over median
   - Good test case

3. **Pol** (Priority 3)
   - Variance: 16.07 (MODERATE consistency)
   - US advantage: +2.84% over median
   - Highest absolute performance (~98%)

### Why TabPFN?
- **Pre-trained** neural network → no training time
- **No hyperparameters** → no tuning needed
- **Fast** inference → ~0.1s per prediction
- **Well-calibrated** uncertainties → better for active learning
- **Strong performance** on small tabular datasets

## Repository Structure

```
active-learning-benchmark/
│
├── data/                          # Dataset files
│   ├── get_data_zhan21.sh        # Download all datasets
│   └── [libsvm files]            # Dataset files in LIBSVM format
│
├── src/                          # Source code
│   ├── main.py                   # Original experiment runner
│   ├── config.py                 # Model and strategy configuration
│   │
│   ├── models/                   # Model implementations
│   │   └── tabpfn_model.py      # ⭐ NEW: TabPFN wrapper
│   │
│   ├── run_tabpfn_experiments.py # ⭐ NEW: Automated experiment runner
│   │
│   └── [query_strategies/]      # AL strategy implementations
│
├── results/                      # Experiment outputs
│   ├── *-aubc.csv               # Main results (Area Under Budget Curve)
│   ├── *-detail.csv             # Per-iteration details
│   └── analyze_tabpfn_results.py # ⭐ NEW: Results analysis
│
├── libact-dev/                   # Active learning library (external)
├── alipy-dev/                    # Another AL library (external)
│
└── requirements.txt              # Python dependencies
```

## Key New Files

### 1. `tabpfn_model.py`
**Location**: `src/models/`

**Purpose**: Makes TabPFN compatible with the active learning benchmark framework

**Core Class**:
```python
class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for TabPFN that handles:
    - Max 10,000 training samples (auto-truncates)
    - Max 100 features (auto-truncates)
    - sklearn-compatible interface
    """
    
    def __init__(self, N_ensemble_configurations=32, device='cpu'):
        # N_ensemble_configurations: 16 (fast), 32 (balanced), 64 (accurate)
        # device: 'cpu' or 'cuda'
        pass
```

**Variants**:
- `TabPFNFast()` - 16 ensemble members
- `TabPFNAccurate()` - 64 ensemble members

### 2. `run_tabpfn_experiments.py`
**Location**: `src/`

**Purpose**: Automates running experiments across datasets and query strategies

**Key Functions**:
```python
class ExperimentRunner:
    def run_single_experiment(dataset, strategy, seed)
    def run_all_experiments(datasets, strategies)
    def print_summary()
    def save_results()
```

**Command Line Usage**:
```bash
# Quick test (1 trial, 1 dataset)
python run_tabpfn_experiments.py --quick_test

# Full experiments (10 trials, all datasets)
python run_tabpfn_experiments.py

# Custom configuration
python run_tabpfn_experiments.py \
    --datasets splice ionosphere \
    --strategies US Random QBC \
    --trials 20 \
    --budget 300
```

### 3. `analyze_tabpfn_results.py`
**Location**: `results/`

**Purpose**: Analyzes and visualizes experimental results

**Features**:
- Load AUBC and detail CSV files
- Compute statistics (mean, std, min, max)
- Generate comparison plots
- Create learning curves
- Perform statistical tests
- Export comprehensive reports

**Usage**:
```bash
python analyze_tabpfn_results.py \
    --datasets splice ionosphere pol \
    --output_dir ./analysis_output
```

## Experimental Configuration

### Default Settings
```python
EXPERIMENTAL_CONFIG = {
    'init_lbl_size': 20,     # Initial labeled pool size
    'tst_size': 0.4,         # Test set size (40% of data)
    'budget': 200,           # Total labeling budget
    'n_trials': 10,          # Number of independent runs
    'seed_start': 0,         # Starting random seed
    'exp_name': 'TabPFN',    # Experiment identifier
}
```

### Query Strategies
```python
STRATEGIES = {
    'US': 'Uncertainty Sampling (margin-based)',
    'Random': 'Random sampling baseline',
    'QBC': 'Query By Committee',
    'BALD': 'Bayesian Active Learning by Disagreement',
    'Core-Set': 'Diversity-based core-set selection',
}
```

### Performance Metric
**AUBC** (Area Under Budget Curve):
- Integral of test accuracy over labeling iterations
- Range: [0, 1]
- Higher is better
- More robust than final accuracy alone

## TabPFN Technical Details

### Capabilities
✅ Binary classification
✅ Multi-class classification
✅ Fast inference (no training)
✅ Calibrated probabilities
✅ Handles missing features gracefully

### Constraints (CRITICAL)
⚠️ **Maximum 10,000 training samples**
⚠️ **Maximum 100 features**
⚠️ **No regression** (classification only)

*Note: Our wrapper automatically handles these constraints via truncation*

### Configuration Options
```python
# Balanced performance (recommended)
model = TabPFNWrapper(N_ensemble_configurations=32, device='cpu')

# Fast inference (for testing)
model = TabPFNWrapper(N_ensemble_configurations=16, device='cpu')

# Maximum accuracy (slower)
model = TabPFNWrapper(N_ensemble_configurations=64, device='cuda')
```

### Device Selection
- **CPU**: Always works, ~2-5x slower
- **CUDA**: Requires GPU, much faster
- Auto-fallback to CPU if CUDA unavailable

## Common Operations

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/ariapoy/active-learning-benchmark.git
cd active-learning-benchmark

# Create virtual environment
python3 -m venv act-env
source act-env/bin/activate  # On Windows: act-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install tabpfn

# Install AL libraries
git clone https://github.com/ariapoy/libact.git libact-dev
cd libact-dev && python setup.py install && cd ..

git clone https://github.com/ariapoy/ALiPy.git alipy-dev
cp -r alipy-dev/alipy alipy-dev/alipy_dev
```

### 2. Download Datasets
```bash
cd data
bash get_data_zhan21.sh
cd ..
```

### 3. Run Single Experiment
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

### 4. Run Batch Experiments
```bash
cd src

# All datasets, all strategies
python run_tabpfn_experiments.py

# Specific subset
python run_tabpfn_experiments.py \
    --datasets splice \
    --strategies US Random \
    --trials 5
```

### 5. Analyze Results
```bash
cd results

# Full analysis
python analyze_tabpfn_results.py

# Specific datasets
python analyze_tabpfn_results.py --datasets splice ionosphere
```

## Expected Results

### Success Criteria

**Splice Dataset**:
- US outperforms Random by ~3-4%
- Low variance across trials (σ < 0.02)
- Final accuracy: 92-93%
- AUBC: 0.85-0.87

**Ionosphere Dataset**:
- US outperforms Random by ~2-3%
- Moderate variance (σ < 0.03)
- Final accuracy: 86-87%
- AUBC: 0.83-0.85

**Pol Dataset**:
- US outperforms Random by ~2-3%
- Moderate variance (σ < 0.04)
- Final accuracy: ~98%
- AUBC: 0.95-0.97

### Performance Indicators

✅ **Good Results**:
- US consistently outperforms Random
- Low variance across trials
- Learning curves show steady improvement
- Statistical significance (p < 0.05)

⚠️ **Warning Signs**:
- High variance (σ > 0.05)
- Random sometimes beats US
- No improvement over iterations
- Inconsistent across seeds

## Troubleshooting

### Issue: TabPFN not found
```bash
pip install tabpfn
# Or
pip install git+https://github.com/automl/TabPFN.git
```

### Issue: CUDA out of memory
```python
# Solution 1: Use CPU
model = TabPFNWrapper(device='cpu')

# Solution 2: Reduce ensemble size
model = TabPFNWrapper(N_ensemble_configurations=16, device='cuda')

# Solution 3: Process in smaller batches
model.predict_proba(X[:1000])  # Process 1000 at a time
```

### Issue: Dataset has > 100 features
```python
# Automatic truncation (wrapper handles this)
# Or use feature selection:
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X, y)
```

### Issue: Import errors (libact/alipy)
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libact-dev:$(pwd)/alipy-dev"

# Or in Python:
import sys
sys.path.insert(0, './libact-dev')
sys.path.insert(0, './alipy-dev')
```

### Issue: No results files found
```bash
# Check results directory
ls -la results/*TabPFN*

# Verify experiment completed
tail -f src/experiment_log.txt  # Check log file

# Check experiment name
grep "exp_name" src/run_tabpfn_experiments.py
```

## File Formats

### Input: LIBSVM Format
```
<label> <index>:<value> <index>:<value> ...

Example:
+1 1:0.5 2:0.3 3:0.8
-1 1:0.2 2:0.9 3:0.1
```

### Output: AUBC CSV
```csv
dataset,query_strategy,seed,trial,aubc,final_accuracy,total_time
splice,US,0,0,0.8523,0.9245,123.45
splice,Random,0,0,0.8123,0.8956,98.76
```

### Output: Detail CSV
```csv
dataset,query_strategy,seed,trial,labeled_size,test_accuracy,query_time
splice,US,0,0,20,0.7234,0.123
splice,US,0,0,40,0.7856,0.145
splice,US,0,0,60,0.8123,0.167
```

## Active Learning Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Initialize                                           │
│    • Split data: train (60%) / test (40%)              │
│    • Create initial labeled pool (20 samples)          │
│    • Remaining training → unlabeled pool               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Active Learning Loop (until budget exhausted)       │
│    ┌──────────────────────────────────────────┐       │
│    │ a. Train TabPFN on labeled pool          │       │
│    │ b. Query strategy selects informative    │       │
│    │    samples from unlabeled pool           │       │
│    │ c. Add selected samples to labeled pool  │       │
│    │ d. Evaluate on test set                  │       │
│    │ e. Log results                           │       │
│    └──────────────────────────────────────────┘       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Analysis                                             │
│    • Compute AUBC                                       │
│    • Compare strategies                                 │
│    • Statistical significance tests                     │
│    • Generate plots and reports                         │
└─────────────────────────────────────────────────────────┘
```

## Research Context

### Original Paper Contributions
1. Expanded benchmark from 16 to 28 datasets
2. Re-evaluated 14+ active learning strategies
3. Found Uncertainty Sampling consistently strong
4. Identified importance of variance analysis

### Our Extension
1. Test with modern neural network (TabPFN)
2. Focus on high-confidence datasets (low variance)
3. Compare TabPFN vs SVM performance
4. Evaluate uncertainty estimation quality

### Research Questions
1. Does TabPFN maintain US superiority?
2. Is TabPFN's uncertainty better than SVM's?
3. Can we reduce variance with TabPFN?
4. Which datasets favor neural approaches?

## Key Statistics

### Performance Metrics
- **AUBC**: Primary metric (higher = better)
- **Final Accuracy**: Test accuracy at budget exhaustion
- **Learning Rate**: Slope of learning curve
- **Data Efficiency**: Samples needed for target accuracy

### Statistical Tests
- **Paired t-test**: Compare strategies (same seeds)
- **Cohen's d**: Effect size (small: 0.2, medium: 0.5, large: 0.8)
- **Confidence Intervals**: 95% CI for mean AUBC

## Development Guidelines

### Code Style
- Follow PEP 8
- Type hints where beneficial
- Comprehensive docstrings
- Error handling with try-except

### Testing Strategy
1. Quick test mode for validation
2. Single trial for debugging
3. Multiple trials (10+) for final results
4. Statistical significance testing

### Logging
- Use Python logging module
- Timestamp all events
- Log to file for long-running experiments
- Console output for progress tracking

## User Profile

**Name**: Cartier
**Level**: High school student (advanced)
**Strengths**:
- Competitive programming (USACO Platinum, AIME)
- Algorithmic problem solving
- Leadership (founded clubs, student council)
- Bilingual (English/Chinese)

**Context**:
- Working on AP Seminar research
- Interested in ML and active learning
- Values reproducibility and clear documentation
- Prefers technical, concise explanations

**Communication Style**:
- Direct and efficient
- Appreciates code examples
- Wants to understand underlying concepts
- Comfortable with technical terminology

## References

### Papers
- **AL Benchmark**: https://openreview.net/pdf?id=855yo1Ubt2
- **TabPFN**: https://arxiv.org/abs/2207.01848
- **IJCAI Survey**: https://ijcai-21.org/program-survey/

### Code
- **Original Benchmark**: https://github.com/ariapoy/active-learning-benchmark
- **TabPFN**: https://github.com/automl/TabPFN
- **LibAct**: https://github.com/ntucllab/libact

### Documentation
- **TabPFN Docs**: https://priorlabs.ai/
- **LibAct Docs**: https://libact.readthedocs.io/

## Notes for Gemini

When assisting with this project:

✅ **Do**:
- Be precise and technical
- Show code examples
- Explain statistical concepts clearly
- Suggest optimizations
- Point out potential issues proactively
- Provide reproducible solutions

❌ **Don't**:
- Over-explain basic concepts (user is advanced)
- Use overly verbose responses
- Assume user needs hand-holding
- Skip technical details

**Preferred Response Style**:
1. Quick summary of what you'll do
2. Code or commands to execute
3. Brief explanation of why
4. Potential issues to watch for
