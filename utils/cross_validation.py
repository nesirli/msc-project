"""
Shared cross-validation strategies for AMR prediction pipeline.
"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class GeographicTemporalKFold(BaseCrossValidator):
    """
    K-Fold cross-validation that respects geographic and temporal structure.
    Ensures strains from the same location-year combination are not split across training/validation.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups parameter (location-year combinations) is required for geographic-temporal CV")
        
        groups = np.array(groups)
        y = np.array(y) if y is not None else None
        
        # Get unique groups (location-year combinations)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(f"Number of location-year groups ({n_groups}) < n_splits ({self.n_splits})")
        
        # Set random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        # Shuffle groups if requested
        if self.shuffle:
            rng.shuffle(unique_groups)
        
        # Split groups into k folds
        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[:n_groups % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_groups = unique_groups[start:stop]
            
            # Find all samples belonging to test groups
            test_mask = np.isin(groups, test_groups)
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]
            
            yield train_indices, test_indices
            current = stop
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_cross_validator(location_year_groups=None, cv_folds=5, random_state=42):
    """
    Get appropriate cross-validator based on available grouping information.
    
    Args:
        location_year_groups: Array of location-year combinations for each sample
        cv_folds: Number of CV folds
        random_state: Random state for reproducibility
        
    Returns:
        Cross-validator instance and groups array
    """
    if location_year_groups is not None:
        # Check if we have enough unique groups for geographic-temporal CV
        unique_groups = len(np.unique(location_year_groups))
        if unique_groups >= cv_folds:
            try:
                cv_splitter = GeographicTemporalKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                print("Using geographic-temporal-aware cross-validation")
                return cv_splitter, location_year_groups
            except ValueError as e:
                print(f"Warning: Cannot use geographic-temporal CV ({e}), falling back to stratified CV")
        else:
            print(f"Warning: Only {unique_groups} location-year groups available for {cv_folds}-fold CV, falling back to stratified CV")
    
    # Fallback to stratified CV
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    print("Using stratified cross-validation")
    return cv_splitter, None


def load_location_year_groups(train_metadata):
    """
    Create location-year groups from training metadata.
    
    Args:
        train_metadata: DataFrame with 'Location' and 'Year' columns
        
    Returns:
        Array of location-year combinations
    """
    location_year = (train_metadata['Location'].fillna('unknown').astype(str) + '_' + 
                    train_metadata['Year'].fillna('unknown').astype(str)).values
    return location_year