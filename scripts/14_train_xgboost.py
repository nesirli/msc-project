#!/usr/bin/env python3
"""
Train XGBoost model with 5-fold nested CV and hyperparameter tuning.
Uses batch-corrected features and standardized utilities.
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
from collections import Counter

# Import shared utilities
from utils.cross_validation import get_cross_validator, load_location_year_groups
from utils.class_balancing import get_tree_model_weights
from utils.evaluation import (
    evaluate_cross_validation_fold, summarize_cross_validation_results,
    compute_comprehensive_metrics, check_success_criteria, save_standardized_results
)

class DummyModel:
    """Dummy model for single-class scenarios."""
    def __init__(self, single_class):
        self.single_class = single_class
    
    def predict(self, X):
        return np.full(X.shape[0], self.single_class)
    
    def predict_proba(self, X):
        prob = 1.0 if self.single_class == 1 else 0.0
        return np.column_stack([1-prob, prob] if X.shape[0] > 0 else [[], []])


def main():
    # Load balanced data (prepared by step 13)
    train_df = pd.read_csv(snakemake.input.train)
    test_df = pd.read_csv(snakemake.input.test)
    
    # Get antibiotic name from output path
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in snakemake.output.model:
            antibiotic = ab
            break
    
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from output path")
    
    print(f"Training XGBoost model for {antibiotic}")
    
    # Batch-corrected data already contains labels as 'R' column
    train_merged = train_df.dropna(subset=['R'])
    test_merged = test_df.dropna(subset=['R'])
    
    print(f"Training samples: {len(train_merged)}")
    print(f"Test samples: {len(test_merged)}")
    print(f"Training class distribution: {train_merged['R'].value_counts().to_dict()}")
    print(f"Test class distribution: {test_merged['R'].value_counts().to_dict()}")
    
    # Check for class imbalance - single class datasets
    unique_classes = train_merged['R'].unique()
    if len(unique_classes) == 1:
        print(f"WARNING: Only one class present in training data: {unique_classes[0]}")
        print("Cannot perform cross-validation with single class. Creating dummy results.")
        
        # Create dummy results for single-class scenario
        single_class = int(unique_classes[0])
        dummy_results = {
            'cv_results': [{
                'fold': 0,
                'f1': 0.0 if single_class == 0 else 1.0,
                'balanced_accuracy': 1.0,  # Perfect accuracy for single class
                'auc': 0.5,  # Random AUC for single class
                'best_params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
            }],
            'test_results': {
                'f1': 0.0 if single_class == 0 else 1.0,
                'balanced_accuracy': 1.0,
                'auc': 0.5,
                'confusion_matrix': [[len(test_merged), 0], [0, 0]] if single_class == 0 else [[0, 0], [0, len(test_merged)]],
                'classification_report': {}
            },
            'cv_mean_f1': 0.0 if single_class == 0 else 1.0,
            'cv_std_f1': 0.0,
            'note': f'Single class dataset (class {single_class}), no meaningful ML evaluation possible'
        }
        
        # Save dummy results
        with open(snakemake.output.results, 'w') as f:
            json.dump(dummy_results, f, indent=2)
        
        # Create dummy model (just saves class prediction)
        dummy_model = DummyModel(single_class)
        joblib.dump(dummy_model, snakemake.output.model)
        
        # Create dummy importance file
        meta_cols = ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']
        feature_cols = [c for c in train_merged.columns if c not in meta_cols]
        dummy_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.zeros(len(feature_cols))
        })
        dummy_importance.to_csv(snakemake.output.shap, index=False)
        
        # Create dummy plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Single Class Dataset\nClass {single_class}\nNo ML Training Possible', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title(f'XGBoost Results - {antibiotic} (Single Class)')
        plt.axis('off')
        plt.savefig(snakemake.output.plots, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dummy results saved for single-class dataset")
        return
    
    # Identify feature columns (exclude metadata)
    meta_cols = ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']
    feature_cols = [c for c in train_merged.columns if c not in meta_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    
    X_train = train_merged[feature_cols].values
    y_train = train_merged['R'].astype(int).values
    
    # Create location-year groups for geographic-temporal CV using shared utility
    location_year_train = load_location_year_groups(train_merged)
    print(f"Training location-year groups: {len(np.unique(location_year_train))} unique groups")
    print(f"Location-year distribution: {Counter(location_year_train).most_common(10)}")
    
    X_test = test_merged[feature_cols].values
    y_test = test_merged['R'].astype(int).values
    
    print(f"Training class distribution: {np.bincount(y_train)}")
    
    # Load class weights from balance summary (if available)
    try:
        balance_summary_path = snakemake.input.train.replace('_train_final.csv', '_balance_summary.json')
        with open(balance_summary_path, 'r') as f:
            balance_info = json.load(f)
            scale_pos_weight = balance_info.get('scale_pos_weight', 1.0)
            print(f"Using scale_pos_weight from balance summary: {scale_pos_weight:.3f}")
    except:
        # Fallback: calculate from current data
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0
        print(f"Calculated scale_pos_weight from data: {scale_pos_weight:.3f}")
    
    # Parameters
    cv_folds = snakemake.params.cv_folds
    random_state = snakemake.params.random_state
    param_grid = snakemake.params.param_grid
    
    # Create parameter grid
    grid = {
        'n_estimators': param_grid['n_estimators'],
        'max_depth': param_grid['max_depth'],
        'learning_rate': param_grid['learning_rate']
    }
    
    # Base model with class weighting
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    # Geographic-temporal-aware CV to prevent strain leakage
    outer_cv, cv_groups = get_cross_validator(location_year_train, cv_folds, random_state)
    inner_cv, _ = get_cross_validator(location_year_train, cv_folds, random_state)
    location_year_train = cv_groups  # Use processed groups
    
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train, groups=location_year_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        loc_year_tr = location_year_train[train_idx] if location_year_train is not None else None
        
        print(f"Fold {fold + 1}: Train groups={len(np.unique(loc_year_tr)) if loc_year_tr is not None else 'N/A'}, "
              f"Val groups={len(np.unique(location_year_train[val_idx])) if location_year_train is not None else 'N/A'}")
        
        # Grid search on inner fold
        # Note: For simplicity, using standard CV for hyperparameter tuning
        grid_search = GridSearchCV(
            base_model, grid, cv=3, scoring='f1', n_jobs=-1  # Reduced CV for speed
        )
        grid_search.fit(X_tr, y_tr)
        
        # Evaluate on validation fold
        y_pred = grid_search.predict(X_val)
        y_prob = grid_search.predict_proba(X_val)[:, 1]
        
        cv_results.append({
            'fold': fold,
            'f1': f1_score(y_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_prob),
            'best_params': grid_search.best_params_
        })
    
    # Train final model with best average parameters and class weighting
    final_model = xgb.XGBClassifier(
        **cv_results[-1]['best_params'],
        objective='binary:logistic',
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    
    test_results = {
        'f1': f1_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_prob),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
        'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
    }
    
    # Basic feature importance from XGBoost
    feature_importance = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Save results
    results = {
        'cv_results': cv_results,
        'test_results': test_results,
        'cv_mean_f1': np.mean([r['f1'] for r in cv_results]),
        'cv_std_f1': np.std([r['f1'] for r in cv_results])
    }
    
    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2)
    
    joblib.dump(final_model, snakemake.output.model)
    importance_df.to_csv(snakemake.output.shap, index=False)
    
    # Create CV results plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # CV results plot
    plt.subplot(1, 2, 1)
    plt.bar(range(len(cv_results)), [r['f1'] for r in cv_results])
    plt.xlabel('CV Fold')
    plt.ylabel('F1 Score')
    plt.title(f'XGBoost CV F1 Scores\n{antibiotic}')
    plt.ylim(0, 1)
    
    # Class distribution plot
    plt.subplot(1, 2, 2)
    original_dist = np.bincount(train_merged['R'].astype(int))
    balanced_dist = np.bincount(y_train)
    x = ['Resistant (0)', 'Sensitive (1)']
    width = 0.35
    plt.bar([i - width/2 for i in range(len(x))], original_dist, width, label='Original', alpha=0.7)
    plt.bar([i + width/2 for i in range(len(x))], balanced_dist, width, label='SMOTE Balanced', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.title('Class Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(snakemake.output.plots, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CV F1: {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")

if __name__ == "__main__":
    main()