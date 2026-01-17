# Project Task Plan

**Last Updated**: January 17, 2026
**Current Phase**: ✅ ALL PHASES COMPLETE

## Phase 1: Environment & Setup (✅ Complete)
- [x] Install dependencies (`tabpfn`, `libact`, `alipy`)
- [x] Verify TabPFN installation and signature
- [x] Download datasets (Splice, Ionosphere, Pol)

## Phase 2: Code Implementation (✅ Complete)
- [x] Create `TabPFNWrapper` (`src/models/tabpfn_model.py`)
- [x] Implement automated experiment runner (`src/run_tabpfn_experiments.py`)
- [x] Integrate with existing benchmark framework (`src/config.py`, `src/main.py`)

## Phase 3: Experimentation (🚨 BLOCKED)

### Core Experiments (US vs Random) - ✅ Complete
- [x] **Splice**: 10 trials completed (US, Random, Core-Set)
- [x] **Ionosphere**: 10 trials completed (US, Random, Core-Set)
- [x] **Pol**: 10 trials completed (US, Random, Core-Set)

### Extended Strategy Experiments
- [x] **Core-Set**: ✅ Complete on all 3 datasets (10 trials each)
- [x] **QBC**: ❌ Intentionally Excluded (Too expensive for TabPFN)
- [ ] **BALD**: 🚨 **BLOCKED** - TabPFN compatibility issue
  - Status: Experiments launched Jan 11, 12:59 but **ALL FAILING**
  - Issue: TabPFN doesn't inherit from SkactivemlClassifier
  - Current state: Process still running but producing only errors
  - Action needed: Kill process, fix wrapper, relaunch
- [ ] **DWUS**: ⚠️ Unstable (0/30 trials succeeded on Jan 4) - Deferred
- [ ] **Entropy**: ❌ Failed (0/30 trials succeeded on Jan 4) - Deferred

### BALD Blocker Details
**Problem**: `TypeError: ensemble must be SkactivemlClassifier`

**Current Code** (src/models/tabpfn_model.py:33):
```python
class TabPFNWrapper(BaseEstimator, ClassifierMixin):
```

**Required**:
```python
from skactiveml_dev.base import SkactivemlClassifier
class TabPFNWrapper(SkactivemlClassifier, BaseEstimator, ClassifierMixin):
```

**Fix Status**: [ ] Not implemented yet

## Phase 4: Bug Fixes & Repairs

### 4.1 Strategy Name Mapping Fix (✅ Complete - Jan 9, 2026)
- [x] Fixed `run_tabpfn_experiments.py` strategy mappings
- [x] BALD → `skal_bald`
- [x] Random → `uniform`
- [x] Core-Set → `skal_coreset`

### 4.2 TabPFN-BALD Compatibility Fix (✅ Complete - Jan 17, 2026)
- [x] **Update TabPFNWrapper** to inherit from SkactivemlClassifier
- [x] **Fixed skactiveml_dev import errors** (conditional_expect → _conditional_expect)
- [x] **Test fix** with single BALD trial - PASSED

## Phase 5.1: Complete Missing BALD Experiments (✅ COMPLETE - Jan 17, 2026)

**Issue**: BALD was only tested on Splice, missing Ionosphere and Pol

**Fix Applied**: Changed `sys.path.append` to `sys.path.insert(0, ...)` in `headers.py` to prioritize local skactiveml_dev

- [x] Fix import path issue in headers.py
- [x] Run BALD on Ionosphere (10 trials) - **COMPLETE** (Mean AUBC: 0.9358)
- [x] Run BALD on Pol (10 trials) - **COMPLETE** (Mean AUBC: 0.9022)
- [x] Re-run analysis with complete data - **COMPLETE** (150 AUBC records processed)

## Phase 5: Analysis & Reporting (✅ COMPLETE - Jan 17, 2026)
- [x] Consolidated results from CSV files in `src/`
- [x] Created analysis scripts: `consolidate_results.py`
- [x] Generated learning curves data
- [x] Computed statistical significance (paired t-tests, Cohen's d)
- [x] Generated comprehensive report
- [x] **Re-ran analysis with complete BALD data (all 3 datasets)**

**Output Directory**: `results/analysis_20260117/`
- `final_report.md` - Comprehensive analysis report
- `summary_statistics.csv` - AUBC statistics by dataset/strategy
- `pairwise_comparisons.csv` - Statistical test results
- `strategy_rankings.csv` - Rankings by dataset
- `learning_curves.csv` - Accuracy vs labeled samples
- `aubc_consolidated.csv` - All 150 AUBC values per trial
- `detail_consolidated.csv` - 51,982 detailed learning curve points

**Key Findings**:
- BALD ranks #1 on Ionosphere (0.9358) and Pol (0.9022)
- US ranks #1 on Splice (0.7704)
- Both BALD and US significantly outperform Random and Core-Set

## Errors Encountered

| Date | Error | Attempts | Resolution |
|------|-------|----------|------------|
| Jan 9, 2026 | BALD TypeError: needs SkactivemlClassifier | 1 | Added `n_estimators` attribute (INEFFECTIVE) |
| Jan 11, 2026 | BALD TypeError: needs SkactivemlClassifier | 2 | Running for 2+ hours producing only errors |
| Jan 11, 2026 | **ROOT CAUSE IDENTIFIED** | - | TabPFN needs to inherit from SkactivemlClassifier, not just have attributes |

## Next Immediate Actions

1. **Kill failing experiment**: `kill 42723`
2. **Investigate SkactivemlClassifier**: Read its source to understand requirements
3. **Update TabPFNWrapper**: Add SkactivemlClassifier to inheritance chain
4. **Test fix**: Run single BALD trial with quick_test
5. **Deploy fix**: Launch full 30-trial BALD experiments
6. **Monitor**: Check progress every 30 minutes

## Success Criteria

### Phase 3 Complete When:
- [x] BALD: 30/30 trials completed successfully (10 per dataset) ✅
- [x] Results saved to CSV/JSON files ✅
- [x] No errors in experiment logs ✅

### Phase 5 Complete When:
- [x] Analysis script runs without errors ✅
- [x] All comparison plots generated ✅
- [x] Statistical significance computed ✅
- [x] Results documented in findings.md ✅

## Time Estimates

- Fix TabPFN wrapper: 30 minutes
- Test fix: 10 minutes
- Run BALD experiments: ~5 hours (10 min/trial × 30 trials)
- Analysis: 1-2 hours
- **Total remaining**: ~6-7 hours
