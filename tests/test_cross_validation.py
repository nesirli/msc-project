"""Unit tests for cross_validation.py module."""

import pytest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.cross_validation import GeographicTemporalKFold


class TestGeographicTemporalKFold:
    """Test suite for GeographicTemporalKFold class."""

    @pytest.mark.parametrize(
        "n_splits, random_state",
        [
            pytest.param(5, 42, id="5_folds"),
            pytest.param(3, 0, id="3_folds"),
            pytest.param(10, 123, id="10_folds"),
        ],
    )
    def test_initialization(self, n_splits, random_state):
        """Test proper initialization of CV splitter with various params."""
        cv = GeographicTemporalKFold(n_splits=n_splits, random_state=random_state)

        assert cv.n_splits == n_splits
        assert cv.random_state == random_state

    def test_split_without_groups_raises_error(self):
        """Test that split() raises error without groups parameter."""
        cv = GeographicTemporalKFold(n_splits=5)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        with pytest.raises(ValueError, match="groups parameter"):
            list(cv.split(X, y))

    def test_basic_split_structure(self):
        """Test that split returns correct train/test indices."""
        cv = GeographicTemporalKFold(n_splits=5, random_state=42)
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

        folds = list(cv.split(X, y, groups))

        assert len(folds) == 5, "Should have 5 folds"
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0, "Train indices should not be empty"
            assert len(test_idx) > 0, "Test indices should not be empty"

    def test_no_group_overlap(self):
        """Test that samples from same group are not split across folds."""
        cv = GeographicTemporalKFold(n_splits=5, random_state=42)
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups & test_groups
            assert len(overlap) == 0, f"Groups {overlap} appear in both train and test"

    def test_all_samples_used(self):
        """Test that all samples are used across all folds."""
        cv = GeographicTemporalKFold(n_splits=5, random_state=42)
        n_samples = 10
        X = np.arange(n_samples).reshape(-1, 1)
        y = np.array([0, 1] * 5)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

        all_test_indices = []
        for train_idx, test_idx in cv.split(X, y, groups):
            all_test_indices.extend(test_idx)

        assert set(all_test_indices) == set(range(n_samples)), "Not all samples used in testing"

    def test_insufficient_groups_raises_error(self):
        """Test that error is raised if n_groups < n_splits."""
        cv = GeographicTemporalKFold(n_splits=5)
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3])

        with pytest.raises(ValueError, match="Number of location-year groups"):
            list(cv.split(X, y, groups))

    def test_reproducibility_with_random_state(self):
        """Test that CV splits are reproducible with same random state."""
        X = np.arange(20).reshape(-1, 1)
        y = np.array([0, 1] * 10)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])

        cv1 = GeographicTemporalKFold(n_splits=5, random_state=42)
        cv2 = GeographicTemporalKFold(n_splits=5, random_state=42)

        folds1 = list(cv1.split(X, y, groups))
        folds2 = list(cv2.split(X, y, groups))

        assert len(folds1) == len(folds2)
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            assert np.array_equal(train1, train2), "Train indices differ"
            assert np.array_equal(test1, test2), "Test indices differ"

    @pytest.mark.parametrize(
        "shuffle",
        [pytest.param(True, id="shuffle_on"), pytest.param(False, id="shuffle_off")],
    )
    def test_shuffle_produces_valid_splits(self, shuffle):
        """Test that both shuffle modes produce valid 5-fold splits."""
        X = np.arange(20).reshape(-1, 1)
        y = np.array([0, 1] * 10)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])

        cv = GeographicTemporalKFold(n_splits=5, shuffle=shuffle, random_state=42)
        folds = list(cv.split(X, y, groups))

        assert len(folds) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
