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
    
    def test_perfect_predictions_binary(self):
        """Test metrics with perfect binary predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['f1'] == 1.0, "Perfect predictions should have F1=1.0"
        assert metrics['balanced_accuracy'] == 1.0, "Perfect predictions should have balanced_accuracy=1.0"
        assert metrics['auc'] is None, "AUC should be None without probabilities"
    
    def test_random_predictions_binary(self):
        """Test metrics with random binary predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # All wrong
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['f1'] == 0.0, "All-wrong predictions should have F1=0.0"
        assert 'confusion_matrix' in metrics
        assert len(metrics['confusion_matrix']) == 2
    
    def test_with_probabilities(self):
        """Test metrics with probability scores."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)
        
        assert metrics['auc'] is not None, "AUC should be computed with probabilities"
        assert 0 <= metrics['auc'] <= 1, "AUC should be between 0 and 1"
    
    def test_confusion_matrix_format(self):
        """Test confusion matrix structure."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        cm = metrics['confusion_matrix']
        
        assert len(cm) == 2, "Binary classification should have 2x2 confusion matrix"
        assert len(cm[0]) == 2
        assert cm[0][0] == 1, "True negatives incorrect"  # TN=1
        assert cm[1][1] == 1, "True positives incorrect"  # TP=1
    
    def test_zero_division_handling(self):
        """Test handling of cases where metrics are undefined."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # All same class
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Should not raise error
        assert 'f1' in metrics
        assert 'balanced_accuracy' in metrics
    
    def test_multiclass_metrics(self):
        """Test metrics for multi-class classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        
        metrics = compute_comprehensive_metrics(y_true, y_pred, average='macro')
        
        assert 'f1' in metrics
        assert 'balanced_accuracy' in metrics
        assert len(metrics['confusion_matrix']) == 3


class TestEvaluateCrossValidationFold:
    """Test suite for evaluate_cross_validation_fold function."""
    
    def test_fold_evaluation(self, capsys):
        """Test single fold evaluation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        
        result = evaluate_cross_validation_fold(y_true, y_pred, fold_idx=0)
        
        assert 'fold' in result
        assert result['fold'] == 0
        assert 'f1' in result
        assert result['f1'] == 1.0
    
    def test_fold_index_tracking(self):
        """Test that fold index is correctly tracked."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result1 = evaluate_cross_validation_fold(y_true, y_pred, fold_idx=0)
        result2 = evaluate_cross_validation_fold(y_true, y_pred, fold_idx=1)
        
        assert result1['fold'] == 0
        assert result2['fold'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
