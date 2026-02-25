# Unit Testing Summary

## Overview

This project includes comprehensive unit tests for all utility modules in the AMR prediction pipeline. Tests cover classification metrics, class balancing, cross-validation, deep-learning helpers, ensemble methods, error handling, motif analysis, and output validation.

## Test Coverage

### Test Files

| File | Tests | Description |
|------|------:|-------------|
| `tests/test_class_balancing.py` | 10 | Class imbalance handling strategies |
| `tests/test_cross_validation.py` | 8 | Geographic-temporal cross-validation |
| `tests/test_dl_training.py` | 6 | PyTorch device detection, class weights, threading |
| `tests/test_ensemble_methods.py` | 12 | Ensemble init, prediction, stacking, evaluation |
| `tests/test_error_handling.py` | 41 | Exception hierarchy, logging, validators, safe math |
| `tests/test_evaluation.py` | 8 | Classification metrics & fold evaluation |
| `tests/test_motif_analysis.py` | 25 | k-mer extraction, attention regions, cross-model analysis |
| `tests/test_output_validation.py` | 15 | Result standardisation, filename templates, consistency |
| `tests/conftest.py` | — | Shared pytest fixtures |

**Total: 135 passing tests** ✅

## Code Coverage

| Module | Stmts | Miss | Coverage |
|--------|------:|-----:|---------:|
| `__init__.py` | 0 | 0 | 100% |
| `class_balancing.py` | 103 | 69 | 33% |
| `cross_validation.py` | 49 | 16 | 67% |
| `dl_training.py` | 53 | 15 | 72% |
| `ensemble_methods.py` | 203 | 69 | 66% |
| `error_handling.py` | 221 | 36 | 84% |
| `evaluation.py` | 83 | 54 | 35% |
| `motif_analysis.py` | 199 | 66 | 67% |
| `output_validation.py` | 154 | 65 | 58% |
| **TOTAL** | **1065** | **390** | **63%** |

## Running Tests

### Native (venv)

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_evaluation.py -v

# Run with coverage
python -m pytest tests/ --cov=utils --cov-report=term-missing --cov-report=html

# View HTML coverage report (macOS)
open htmlcov/index.html
```

### Docker

```bash
# Run full test suite inside Docker
docker compose run --rm test

# Run with coverage inside Docker
docker compose run --rm test python -m pytest tests/ --cov=utils --cov-report=term-missing
```

## Bugs Found & Fixed

### 1. **evaluation.py — Print Format Error** ✅

F-string formatting error in `evaluate_cross_validation_fold()` when AUC is `None`:

```python
# Before (BROKEN)
f"AUC={metrics['auc']:.3f if metrics['auc'] is not None else 'N/A'}"

# After (FIXED)
auc_str = f"{metrics['auc']:.3f}" if metrics['auc'] is not None else 'N/A'
print(f"AUC={auc_str}")
```

## Test Organisation

### Fixtures (`conftest.py`)

- `sample_binary_data` — Binary classification arrays
- `sample_multiclass_data` — Multi-class classification arrays
- `sample_imbalanced_data` — Imbalanced class distribution
- `sample_predictions` — Prediction examples

### Configuration (`pytest.ini`)

- Strict markers for test categorisation
- Short traceback format
- Colour output enabled

## What's Tested

### ✅ Classification Metrics (`test_evaluation.py`)
- F1-score, balanced accuracy, AUC-ROC
- Confusion matrix structure
- Multi-class metrics
- Zero-division edge cases

### ✅ Imbalance Handling (`test_class_balancing.py`)
- Strategy selection by imbalance ratio
- SMOTE neighbour parameter bounding
- Single-class and extreme-imbalance edge cases
- Multi-class support

### ✅ Cross-Validation (`test_cross_validation.py`)
- Geographic-temporal group isolation
- Reproducibility with random state
- Train/test split integrity
- All samples used across folds

### ✅ Deep-Learning Helpers (`test_dl_training.py`)
- CUDA / MPS / CPU device detection
- Class weight tensor computation
- PyTorch thread configuration
- Fold logging

### ✅ Ensemble Methods (`test_ensemble_methods.py`)
- Simple average, weighted average, majority vote
- Invalid method handling
- Missing model graceful skipping
- Stacking meta-learner training
- End-to-end ensemble from saved JSON results

### ✅ Error Handling (`test_error_handling.py`)
- Custom exception hierarchy (PipelineError, ValidationError, DataError, ModelError)
- Logger setup (console + file)
- `handle_errors` decorator (success, re-raise, keyboard interrupt)
- File / directory / DataFrame / config validators
- Feature matrix validation
- Safe division and log operations
- Memory usage warnings
- Progress tracker lifecycle
- ErrorContext context manager

### ✅ Motif Analysis (`test_motif_analysis.py`)
- k-mer motif extraction (basic, empty, non-DNA filtering)
- Weight-to-motif mapping
- Meaningful motif filtering (short, single-nucleotide, valid)
- CNN motif extraction (no conv layers edge case)
- Attention region detection
- Sequence similarity calculation
- Cross-model motif analysis
- Consensus motif ranking

### ✅ Output Validation (`test_output_validation.py`)
- Model result standardisation & missing field warnings
- Interpretability result standardisation
- Filename template rendering & fallback
- Output directory creation
- JSON serialisation (including NumPy types)
- Feature matrix column renaming
- Pipeline consistency checks

## Future Improvements

1. **Expand coverage** — `class_balancing.py` (33%) and `evaluation.py` (35%) have the most room for growth
2. **Integration tests** — Test Snakemake rule DAG connectivity end-to-end
3. **Performance tests** — Benchmark model training throughput
4. **CI/CD integration** — GitHub Actions with `docker compose run --rm test`

## Dependencies

- pytest ≥ 7.4.3
- pytest-cov
- pytest-xdist
- numpy
- scikit-learn
- imbalanced-learn
- torch (CPU)

See `requirements-dev.txt` for the complete list.
