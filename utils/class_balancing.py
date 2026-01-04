"""
Standardized class balancing strategies for all models.
Implements tiered approach based on imbalance severity.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN


def get_imbalance_strategy(y_train):
    """
    Determine optimal imbalance handling strategy based on class distribution.
    
    Args:
        y_train: Training labels
        
    Returns:
        dict: Strategy configuration with method and parameters
    """
    y_train = np.array(y_train)
    class_counts = np.bincount(y_train)
    
    if len(class_counts) < 2:
        return {"method": "none", "ratio": 1.0, "details": "single_class"}
    
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)
    imbalance_ratio = majority_count / minority_count
    
    print(f"Class distribution: {dict(enumerate(class_counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f} (majority/minority)")
    
    if imbalance_ratio <= 2.0:
        # BALANCED: Light weighting only
        strategy = {
            "method": "class_weights_only",
            "ratio": imbalance_ratio,
            "details": "mild_imbalance",
            "sampling": None,
            "use_focal_loss": False
        }
        print("📊 Strategy: Class weights only (mild imbalance)")
        
    elif imbalance_ratio <= 5.0:
        # MODERATE: Class weights + threshold optimization  
        strategy = {
            "method": "weighted_threshold_opt",
            "ratio": imbalance_ratio,
            "details": "moderate_imbalance",
            "sampling": None,
            "use_focal_loss": False,
            "optimize_threshold": True
        }
        print("⚖️ Strategy: Class weights + threshold optimization (moderate imbalance)")
        
    elif imbalance_ratio <= 15.0:
        # HIGH: SMOTE + class weights
        strategy = {
            "method": "smote_with_weights", 
            "ratio": imbalance_ratio,
            "details": "high_imbalance",
            "sampling": "smote",
            "use_focal_loss": True,
            "optimize_threshold": True,
            "smote_k_neighbors": min(3, minority_count - 1)
        }
        print("🔄 Strategy: SMOTE + class weights + focal loss (high imbalance)")
        
    else:
        # EXTREME: Combination approach
        strategy = {
            "method": "hybrid_extreme",
            "ratio": imbalance_ratio,
            "details": "extreme_imbalance", 
            "sampling": "smote_enn",  # SMOTE + Edited Nearest Neighbors
            "use_focal_loss": True,
            "optimize_threshold": True,
            "ensemble_method": "balanced_bagging",
            "smote_k_neighbors": max(1, min(2, minority_count - 1)),
            "under_sample_ratio": 0.3  # Keep 30% of majority class
        }
        print("🚨 Strategy: Hybrid extreme (SMOTE+ENN + focal loss + ensemble)")
    
    return strategy


def apply_sampling_strategy(X_train, y_train, strategy):
    """
    Apply the determined sampling strategy to training data.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        strategy: Strategy dict from get_imbalance_strategy()
        
    Returns:
        X_resampled, y_resampled: Resampled training data
    """
    if strategy["sampling"] is None:
        print("No sampling applied - using original data")
        return X_train, y_train
    
    try:
        if strategy["sampling"] == "smote":
            k_neighbors = strategy.get("smote_k_neighbors", 3)
            if k_neighbors < 1:
                print("Warning: Not enough minority samples for SMOTE, skipping sampling")
                return X_train, y_train
                
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {len(X_train)} -> {len(X_resampled)} samples")
            
        elif strategy["sampling"] == "smote_enn":
            k_neighbors = strategy.get("smote_k_neighbors", 2)
            if k_neighbors < 1:
                print("Warning: Not enough minority samples for SMOTE-ENN, falling back to undersampling")
                undersampler = RandomUnderSampler(sampling_strategy=strategy.get("under_sample_ratio", 0.5), 
                                                random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            else:
                smote_enn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
                X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
                print(f"SMOTE-ENN applied: {len(X_train)} -> {len(X_resampled)} samples")
                
        else:
            print(f"Unknown sampling method: {strategy['sampling']}")
            return X_train, y_train
            
        print(f"New class distribution: {dict(enumerate(np.bincount(y_resampled)))}")
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Warning: Sampling failed ({e}), using original data")
        return X_train, y_train


def compute_balanced_class_weights(y_train, method='balanced'):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        method: 'balanced' or 'balanced_subsample'
        
    Returns:
        Dictionary of class weights and scale_pos_weight for tree models
    """
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    
    # Compute sklearn-style class weights
    if len(classes) == 1:
        print(f"Warning: Only single class {classes[0]} present in training data")
        class_weights = {classes[0]: 1.0}
        scale_pos_weight = 1.0
    else:
        class_weights_array = compute_class_weight(method, classes=classes, y=y_train)
        class_weights = dict(zip(classes, class_weights_array))
        
        # For binary classification, compute scale_pos_weight for tree models
        if len(classes) == 2 and 0 in classes and 1 in classes:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        else:
            scale_pos_weight = 1.0
    
    print(f"Class distribution: {Counter(y_train)}")
    print(f"Class weights: {class_weights}")
    print(f"Scale pos weight: {scale_pos_weight}")
    
    return class_weights, scale_pos_weight


def get_tree_model_weights(y_train):
    """Get class weights for tree models (XGBoost, LightGBM)."""
    class_weights, scale_pos_weight = compute_balanced_class_weights(y_train)
    return scale_pos_weight


def get_deep_model_weights(y_train):
    """Get class weights for deep learning models (CNN, DNABERT)."""
    class_weights, _ = compute_balanced_class_weights(y_train)
    
    # Convert to tensor-compatible format
    if len(class_weights) == 2 and 0 in class_weights and 1 in class_weights:
        weight_tensor = [class_weights[0], class_weights[1]]
    else:
        # Multi-class or single class case
        weight_tensor = list(class_weights.values())
    
    return weight_tensor


def validate_class_balance_consistency():
    """
    Validate that all models use consistent class balancing approach.
    This should be called during pipeline validation.
    """
    print("Validating class balance consistency across all models...")
    
    # Mock data for testing
    y_test = np.array([0, 0, 0, 1, 1])
    
    # Test tree model weights
    tree_weight = get_tree_model_weights(y_test)
    print(f"Tree model scale_pos_weight: {tree_weight}")
    
    # Test deep model weights  
    deep_weights = get_deep_model_weights(y_test)
    print(f"Deep model class weights: {deep_weights}")
    
    # Check consistency
    expected_ratio = 3/2  # 3 negatives, 2 positives
    tree_ratio = tree_weight
    deep_ratio = deep_weights[1] / deep_weights[0] if len(deep_weights) == 2 else 1.0
    
    print(f"Expected ratio: {expected_ratio:.3f}")
    print(f"Tree ratio: {tree_ratio:.3f}")
    print(f"Deep ratio: {deep_ratio:.3f}")
    
    if abs(tree_ratio - expected_ratio) < 0.1 and abs(deep_ratio - 1.0) < 0.1:
        print("✅ Class balancing is consistent across models")
    else:
        print("❌ Class balancing inconsistency detected!")
    
    return tree_weight, deep_weights


if __name__ == "__main__":
    validate_class_balance_consistency()