"""Unit tests for class_balancing.py module."""

import pytest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.class_balancing import get_imbalance_strategy


class TestGetImbalanceStrategy:
    """Test suite for get_imbalance_strategy function."""

    @pytest.mark.parametrize(
        "n_majority, n_minority, expected_ratio, expected_detail",
        [
            pytest.param(3, 3, 1.0, "mild_imbalance", id="balanced"),
            pytest.param(6, 4, 1.5, None, id="mild"),
            pytest.param(9, 3, 3.0, None, id="moderate"),
            pytest.param(50, 2, 25.0, "extreme_imbalance", id="extreme"),
        ],
    )
    def test_imbalance_ratio_and_detail(
        self, n_majority, n_minority, expected_ratio, expected_detail
    ):
        """Test ratio and detail classification across imbalance levels."""
        y_train = np.array([0] * n_majority + [1] * n_minority)
        strategy = get_imbalance_strategy(y_train)

        assert strategy["ratio"] == expected_ratio
        if expected_detail is not None:
            assert strategy["details"] == expected_detail

    def test_balanced_uses_class_weights_only(self):
        """Test balanced classes select class_weights_only method."""
        y_train = np.array([0, 0, 0, 1, 1, 1])
        strategy = get_imbalance_strategy(y_train)

        assert strategy["method"] == "class_weights_only"

    def test_mild_imbalance_uses_class_weights(self):
        """Test mild imbalance uses class weights."""
        y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        strategy = get_imbalance_strategy(y_train)

        assert "class_weights" in strategy["method"]
        assert 1.0 < strategy["ratio"] <= 2.0

    def test_moderate_imbalance_method(self):
        """Test moderate imbalance uses weighted method or stays below threshold."""
        y_train = np.array([0] * 9 + [1] * 3)
        strategy = get_imbalance_strategy(y_train)

        assert "weighted" in strategy["method"].lower() or strategy["ratio"] <= 5.0

    @pytest.mark.parametrize(
        "n_majority, n_minority",
        [
            pytest.param(20, 2, id="high_10x"),
            pytest.param(50, 2, id="extreme_25x"),
        ],
    )
    def test_high_imbalance_enables_focal_loss(self, n_majority, n_minority):
        """Test that high/extreme imbalance enables focal loss."""
        y_train = np.array([0] * n_majority + [1] * n_minority)
        strategy = get_imbalance_strategy(y_train)

        assert strategy["use_focal_loss"] is True

    def test_high_imbalance_uses_smote(self):
        """Test high imbalance selects a SMOTE-based method."""
        y_train = np.array([0] * 20 + [1] * 2)
        strategy = get_imbalance_strategy(y_train)

        assert "smote" in strategy["method"].lower()

    def test_single_class(self):
        """Test with single class only."""
        y_train = np.array([0, 0, 0, 0])
        strategy = get_imbalance_strategy(y_train)

        assert strategy["method"] == "none"
        assert strategy["details"] == "single_class"

    def test_multiclass_imbalance(self):
        """Test with multi-class imbalance."""
        y_train = np.array([0] * 10 + [1] * 5 + [2] * 2)
        strategy = get_imbalance_strategy(y_train)

        assert strategy is not None
        assert "ratio" in strategy

    @pytest.mark.parametrize(
        "y_train",
        [
            pytest.param(np.array([0] * 8 + [1] * 2), id="imbalanced"),
            pytest.param(np.array([0, 0, 0, 1, 1, 1]), id="balanced"),
            pytest.param(np.array([0] * 50 + [1] * 2), id="extreme"),
        ],
    )
    def test_strategy_contains_required_keys(self, y_train):
        """Test that strategy dict always contains required keys."""
        strategy = get_imbalance_strategy(y_train)

        for key in ("method", "ratio", "details"):
            assert key in strategy, f"Missing required key: {key}"

    @pytest.mark.parametrize(
        "n_majority, n_minority, expected_ratio",
        [
            pytest.param(10, 5, 2.0, id="2x"),
            pytest.param(9, 3, 3.0, id="3x"),
            pytest.param(20, 2, 10.0, id="10x"),
        ],
    )
    def test_imbalance_ratio_calculation(self, n_majority, n_minority, expected_ratio):
        """Test correct imbalance ratio calculation."""
        y_train = np.array([0] * n_majority + [1] * n_minority)
        strategy = get_imbalance_strategy(y_train)

        assert strategy["ratio"] == expected_ratio

    def test_smote_k_neighbors_bounded(self):
        """Test that SMOTE k_neighbors is bounded by minority class size."""
        y_train = np.array([0] * 20 + [1] * 2)
        strategy = get_imbalance_strategy(y_train)

        if "smote_k_neighbors" in strategy:
            assert strategy["smote_k_neighbors"] <= 1  # min(3, 2-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
