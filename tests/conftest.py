"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_binary_data():
    """Fixture providing sample binary classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.binomial(1, 0.3, 100)
    return X, y


@pytest.fixture
def sample_multiclass_data():
    """Fixture providing sample multi-class classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.multinomial(1, [0.33, 0.33, 0.34], 100).argmax(axis=1)
    return X, y


@pytest.fixture
def sample_imbalanced_data():
    """Fixture providing sample imbalanced binary classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # 90 negative, 10 positive samples
    y = np.concatenate([np.zeros(90), np.ones(10)]).astype(int)
    np.random.shuffle(y)
    return X, y


@pytest.fixture
def sample_predictions():
    """Fixture providing sample predictions."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    return y_true, y_pred
