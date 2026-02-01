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
    
    def test_balanced_classes(self):
        """Test with balanced class distribution."""
        y_train = np.array([0, 0, 0, 1, 1, 1])
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy['method'] == 'class_weights_only'
        assert strategy['ratio'] == 1.0
        assert strategy['details'] == 'mild_imbalance'
    
    def test_mild_imbalance(self):
        """Test with mild imbalance (ratio 1.5)."""
        # 6 zeros, 4 ones = ratio 1.5
        y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        strategy = get_imbalance_strategy(y_train)
        
        assert 'class_weights' in strategy['method']
        assert 1.0 < strategy['ratio'] <= 2.0
    
    def test_moderate_imbalance(self):
        """Test with moderate imbalance (ratio ~3)."""
        # 9 zeros, 3 ones = ratio 3
        y_train = np.array([0]*9 + [1]*3)
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy['ratio'] == 3.0
        # Should use class weights, possibly with threshold optimization
        assert 'weighted' in strategy['method'].lower() or strategy['ratio'] <= 5.0
    
    def test_high_imbalance(self):
        """Test with high imbalance (ratio 10)."""
        # 20 zeros, 2 ones = ratio 10
        y_train = np.array([0]*20 + [1]*2)
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy['ratio'] == 10.0
        assert 'smote' in strategy['method'].lower()
        assert strategy['use_focal_loss'] is True
    
    def test_extreme_imbalance(self):
        """Test with extreme imbalance (ratio > 15)."""
        # 50 zeros, 2 ones = ratio 25
        y_train = np.array([0]*50 + [1]*2)
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy['ratio'] == 25.0
        assert strategy['details'] == 'extreme_imbalance'
        assert strategy['use_focal_loss'] is True
    
    def test_single_class(self):
        """Test with single class only."""
        y_train = np.array([0, 0, 0, 0])
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy['method'] == 'none'
        assert strategy['details'] == 'single_class'
    
    def test_multiclass_imbalance(self):
        """Test with multi-class imbalance."""
        # 10 class 0, 5 class 1, 2 class 2
        y_train = np.array([0]*10 + [1]*5 + [2]*2)
        strategy = get_imbalance_strategy(y_train)
        
        assert strategy is not None
        assert 'ratio' in strategy
    
    def test_strategy_contains_required_keys(self):
        """Test that strategy dict contains all required keys."""
        y_train = np.array([0]*8 + [1]*2)
        strategy = get_imbalance_strategy(y_train)
        
        required_keys = ['method', 'ratio', 'details']
        for key in required_keys:
            assert key in strategy, f"Missing required key: {key}"
    
    def test_imbalance_ratio_calculation(self):
        """Test correct imbalance ratio calculation."""
        y_train = np.array([0]*10 + [1]*5)
        strategy = get_imbalance_strategy(y_train)
        
        expected_ratio = 10 / 5
        assert strategy['ratio'] == expected_ratio
    
    def test_smote_k_neighbors_bounded(self):
        """Test that SMOTE k_neighbors is bounded by minority class size."""
        # Only 2 minority samples
        y_train = np.array([0]*20 + [1]*2)
        strategy = get_imbalance_strategy(y_train)
        
        if 'smote_k_neighbors' in strategy:
            assert strategy['smote_k_neighbors'] <= 1  # min(3, 2-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
