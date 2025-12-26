# TabPFN Active Learning - Quick Start Guide

## 📋 Overview

Reproduce active learning experiments using TabPFN on the top 3 datasets where Uncertainty Sampling (US) shows the best performance:

- **Splice** (Variance: 6.26, US advantage: +3.93%)
- **Ionosphere** (Variance: 11.00, US advantage: +3.24%)
- **Pol** (Variance: 16.07, US advantage: +2.84%)

## 🚀 Quick Start (5 minutes)

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/CoolJosh0221/active-learning-benchmark.git
cd active-learning-benchmark

# Create virtual environment
python3 -m venv act-env
source act-env/bin/activate

# Install requirements
pip install -r requirements.txt
pip install tabpfn

# Install AL libraries
# Note: These manual copy steps are required for the imports in the benchmark code to work.

# 1. Setup ALiPy
git clone https://github.com/ariapoy/ALiPy.git alipy-dev
cp -r alipy-dev/alipy alipy-dev/alipy_dev

# 2. Setup libact
git clone https://github.com/CoolJosh0221/libact.git libact-dev
cd libact-dev && python setup.py install && cd ..
cp -r libact-dev/libact libact-dev/libact_dev

# 3. Setup active-learning (Google)
git clone https://github.com/ariapoy/active-learning.git

# 4. Setup scikit-activeml
git clone https://github.com/ariapoy/scikit-activeml.git scikit-activeml-dev
cp -r scikit-activeml-dev/skactiveml scikit-activeml-dev/skactiveml_dev
```

### 2. Add TabPFN Support

```bash
# Copy the TabPFN model wrapper to src/models/
cp /path/to/tabpfn_model.py src/models/

# Copy experiment scripts
cp /path/to/run_tabpfn_experiments.py src/
cp /path/to/analyze_tabpfn_results.py results/

# Make scripts executable
chmod +x src/run_tabpfn_experiments.py
chmod +x results/analyze_tabpfn_results.py
```

### 3. Download Datasets

```bash
cd data
bash get_data_zhan21.sh
cd ..
```

### 4. Run Quick Test

```bash
cd src
python run_tabpfn_experiments.py --quick_test
```

### 5. Run Full Experiments

```bash
# Run all experiments (will take several hours)
python run_tabpfn_experiments.py

# Or run specific datasets/strategies
python run_tabpfn_experiments.py --datasets splice --strategies US Random
```

### 6. Analyze Results

```bash
cd ../results
python analyze_tabpfn_results.py --datasets splice ionosphere pol
```

## 📊 What You'll Get

### Result Files

- `*-aubc.csv` - Area Under Budget Curve (main performance metric)
- `*-detail.csv` - Detailed accuracy at each labeling iteration
- `tabpfn_report.txt` - Summary statistics and rankings
- `tabpfn_aubc_comparison.png` - Bar chart comparing strategies
- `tabpfn_learning_curve_*.png` - Learning curves per dataset

### Key Metrics

- **AUBC** (Area Under Budget Curve): Higher is better
- **Final Accuracy**: Test set accuracy after full labeling budget
- **Data Efficiency**: How quickly accuracy improves

## 🔧 Customization

### Modify Experimental Settings

```bash
python run_tabpfn_experiments.py \
    --trials 20 \           # Number of independent runs
    --budget 300 \          # Total labeling budget
    --init_size 30          # Initial labeled pool size
```

### Add More Query Strategies

Edit `run_tabpfn_experiments.py`:

```python
QUERY_STRATEGIES = {
    'US': {...},
    'Random': {...},
    'Your-Strategy': {
        'name': 'your-strategy-name',
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Your description',
    },
}
```

### Use GPU Acceleration

Edit `tabpfn_model.py` or pass device parameter:

```python
model = TabPFNWrapper(device='cuda')
```

## ⚠️ Important Notes

### TabPFN Constraints

- **Max 10,000 training samples** - Automatically truncated
- **Max 100 features** - Uses first 100 features
- **Binary/Multi-class only** - No regression support

### Recommended Settings

```python
# Standard (balanced)
N_ensemble_configurations = 32

# Fast (less accurate)
N_ensemble_configurations = 16

# Accurate (slower)
N_ensemble_configurations = 64
```

## 🐛 Troubleshooting

### Problem: "No module named 'tabpfn'"

```bash
pip install tabpfn
```

### Problem: "CUDA out of memory"

```python
# Use CPU instead
model = TabPFNWrapper(device='cpu')
```

### Problem: "Dataset not found"

```bash
cd data
bash get_data_zhan21.sh
```

### Problem: Import errors for libact/alipy

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libact-dev:$(pwd)/alipy-dev"
```

## 📈 Expected Results

Based on the variance analysis, you should see:

### Splice Dataset

- ✅ US should outperform other strategies
- ✅ Low variance across trials (consistent results)
- ✅ ~+3.93% advantage over median

### Ionosphere Dataset

- ✅ US should show strong performance
- ✅ Moderate variance
- ✅ ~+3.24% advantage over median

### Pol Dataset

- ✅ Very high accuracy (~98%)
- ✅ US maintains advantage
- ✅ ~+2.84% advantage over median

## 📚 Key Files You Created

1. **tabpfn_model.py** - TabPFN wrapper for AL benchmark
2. **run_tabpfn_experiments.py** - Automated experiment runner
3. **analyze_tabpfn_results.py** - Results analysis and visualization
4. **tabpfn_active_learning_guide.md** - Complete documentation

## 🎯 Next Steps

1. **Compare with baseline**: Run same experiments with SVM
2. **Add more strategies**: DWUS, LAL, ALBL
3. **Hyperparameter tuning**: Try different ensemble sizes
4. **Statistical testing**: Perform significance tests
5. **Extend to more datasets**: Test on additional benchmarks

## 📖 References

- TabPFN Paper: <https://arxiv.org/abs/2207.01848>
- AL Benchmark Paper: <https://openreview.net/pdf?id=855yo1Ubt2>
- Original Repository: <https://github.com/ariapoy/active-learning-benchmark>

## 💡 Tips

1. **Start with quick test** to verify setup
2. **Use parallel execution** for faster experiments
3. **Monitor log files** to track progress
4. **Save intermediate results** in case of crashes
5. **Document any modifications** for reproducibility

---

**Need Help?** Check the full guide in `tabpfn_active_learning_guide.md` or raise an issue on GitHub.
