"""Unit tests for evaluation.py module."""

import pytest
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.evaluation import compute_comprehensive_metrics, evaluate_cross_validation_fold


class TestComputeComprehensiveMetrics:
    """Test suite for compute_comprehensive_metrics function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, expected_f1, expected_ba",
        [
            pytest.param(
                [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], 1.0, 1.0,
                id="perfect",
            ),
            pytest.param(
                [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], 0.0, 0.0,
                id="all_wrong",
            ),
        ],
    )
    def test_binary_prediction_quality(self, y_true, y_pred, expected_f1, expected_ba):
        """Test F1 and balanced accuracy for known binary predictions."""
        metrics = compute_comprehensive_metrics(
            np.array(y_true), np.array(y_pred)
        )

        assert metrics["f1"] == expected_f1
        assert metrics["balanced_accuracy"] == expected_ba
        assert metrics["auc"] is None  # no probabilities supplied

    def test_with_probabilities(self):
        """Test metrics with probability scores."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])

        metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)

        assert metrics["auc"] is not None, "AUC should be computed with probabilities"
        assert 0 <= metrics["auc"] <= 1, "AUC should be between 0 and 1"

    def test_confusion_matrix_format(self):
        """Test confusion matrix structure."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = compute_comprehensive_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]

        assert len(cm) == 2, "Binary classification should have 2x2 confusion matrix"
        assert len(cm[0]) == 2
        assert cm[0][0] == 1, "True negatives incorrect"
        assert cm[1][1] == 1, "True positives incorrect"

    def test_zero_division_handling(self):
        """Test handling of cases where metrics are undefined."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_comprehensive_metrics(y_true, y_pred)

        assert "f1" in metrics
        assert "balanced_accuracy" in metrics

    def test_multiclass_metrics(self):
        """Test metrics for multi-class classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])

        metrics = compute_comprehensive_metrics(y_true, y_pred, average="macro")

        assert "f1" in metrics
        assert "balanced_accuracy" in metrics
        assert len(metrics["confusion_matrix"]) == 3


class TestEvaluateCrossValidationFold:
    """Test suite for evaluate_cross_validation_fold function."""

    @pytest.mark.parametrize(
        "fold_idx",
        [pytest.param(0, id="fold_0"), pytest.param(1, id="fold_1"), pytest.param(4, id="fold_4")],
    )
    def test_fold_index_tracking(self, fold_idx):
        """Test that fold index is correctly stored in result."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = evaluate_cross_validation_fold(y_true, y_pred, fold_idx=fold_idx)

        assert result["fold"] == fold_idx
        assert "f1" in result
        assert result["f1"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
