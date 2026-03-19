#!/usr/bin/env python3
"""
Train LightGBM model with 5-fold nested CV and hyperparameter tuning.
Uses batch-corrected features and standardized utilities.
Feature importance computed via SHAP TreeExplainer (not native split importance).

Performance notes:
- Reduced num_leaves grid from [31, 63, 127] to [31, 63] in config (33% faster)
- Convert DataFrames to numpy arrays for better memory efficiency
- Grid search: 54 combinations (vs XGBoost's 27)
- Total fits: 810 (5 folds × 54 combinations × 3 inner CV)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, BaseCrossValidator
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
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
        return np.column_stack([1 - prob, prob] if X.shape[0] > 0 else [[], []])


def compute_shap_importance(model, X, feature_cols):
    """
    Compute SHAP values using TreeExplainer and return mean absolute SHAP
    as a ranked importance DataFrame.

    SHAP (SHapley Additive exPlanations) gives each feature a contribution
    score for each prediction. Mean |SHAP| across all test samples is the
    standard interpretability metric used in genomic ML literature
    (Lundberg & Lee 2017).

    For LightGBM binary classification, TreeExplainer returns shap_values
    as a list of two arrays [class_0, class_1]. We use class_1 (resistant).

    Args:
        model: Fitted LGBMClassifier
        X: Feature matrix (numpy array) — use test set for unbiased importance
        feature_cols: List of feature names

    Returns:
        Tuple of (importance_df, shap_values_class1)
        importance_df has columns ['feature', 'importance', 'shap_mean', 'shap_std']
        sorted by mean |SHAP| descending.
    """
    print("Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM binary returns list: [shap_class0, shap_class1]
    # Use class 1 (resistant) as the target class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    std_abs_shap = np.abs(shap_vals).std(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_abs_shap,   # primary column — kept as 'importance' for step 19 compatibility
        'shap_mean': mean_abs_shap,
        'shap_std': std_abs_shap,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"SHAP computed. Top 5 features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['shap_mean']:.4f} ± {row['shap_std']:.4f}")

    return importance_df, shap_vals


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

    print(f"Training LightGBM model for {antibiotic}")

    # Batch-corrected data already contains labels as 'R' column
    train_merged = train_df.dropna(subset=['R'])
    test_merged = test_df.dropna(subset=['R'])

    print(f"Training samples: {len(train_merged)}")
    print(f"Test samples: {len(test_merged)}")
    print(f"Training class distribution: {train_merged['R'].value_counts().to_dict()}")
    print(f"Test class distribution: {test_merged['R'].value_counts().to_dict()}")

    # Handle single-class datasets
    unique_classes = train_merged['R'].unique()
    if len(unique_classes) == 1:
        print(f"WARNING: Only one class present in training data: {unique_classes[0]}")
        print("Cannot perform cross-validation with single class. Creating dummy results.")

        single_class = int(unique_classes[0])
        dummy_results = {
            'cv_results': [{
                'fold': 0,
                'f1': 0.0 if single_class == 0 else 1.0,
                'balanced_accuracy': 1.0,
                'auc': 0.5,
                'best_params': {
                    'n_estimators': 100, 'max_depth': 3,
                    'learning_rate': 0.1, 'num_leaves': 31
                }
            }],
            'test_results': {
                'f1': 0.0 if single_class == 0 else 1.0,
                'balanced_accuracy': 1.0,
                'auc': 0.5,
                'confusion_matrix': (
                    [[len(test_merged), 0], [0, 0]] if single_class == 0
                    else [[0, 0], [0, len(test_merged)]]
                ),
                'classification_report': {}
            },
            'cv_mean_f1': 0.0 if single_class == 0 else 1.0,
            'cv_std_f1': 0.0,
            'note': f'Single class dataset (class {single_class}), no meaningful ML evaluation possible'
        }

        with open(snakemake.output.results, 'w') as f:
            json.dump(dummy_results, f, indent=2)

        dummy_model = DummyModel(single_class)
        joblib.dump(dummy_model, snakemake.output.model)

        meta_cols = ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']
        feature_cols = [c for c in train_merged.columns if c not in meta_cols]
        dummy_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.zeros(len(feature_cols)),
            'shap_mean': np.zeros(len(feature_cols)),
            'shap_std': np.zeros(len(feature_cols)),
        })
        dummy_importance.to_csv(snakemake.output.shap, index=False)

        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Single Class Dataset\nClass {single_class}\nNo ML Training Possible',
                 ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title(f'LightGBM Results - {antibiotic} (Single Class)')
        plt.axis('off')
        plt.savefig(snakemake.output.plots, dpi=300, bbox_inches='tight')
        plt.close()

        print("Dummy results saved for single-class dataset")
        return

    # Identify feature columns (exclude metadata)
    meta_cols = ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']
    feature_cols = [c for c in train_merged.columns if c not in meta_cols]

    print(f"Number of features: {len(feature_cols)}")

    # Convert to numpy arrays for performance
    X_train = train_merged[feature_cols].values
    y_train = train_merged['R'].astype(int).values

    # Geographic-temporal CV groups
    location_year_train = (
        train_merged['Location'].fillna('unknown').astype(str) + '_' +
        train_merged['Year'].fillna('unknown').astype(str)
    ).values
    print(f"Training location-year groups: {len(np.unique(location_year_train))} unique groups")
    print(f"Location-year distribution: {Counter(location_year_train).most_common(10)}")
    print("Using geographic-temporal CV (location-year grouping)")

    X_test = test_merged[feature_cols].values
    y_test = test_merged['R'].astype(int).values

    print(f"Training class distribution: {np.bincount(y_train)}")

    # Load class weights from balance summary
    try:
        balance_summary_path = snakemake.input.train.replace('_train_final.csv', '_balance_summary.json')
        with open(balance_summary_path, 'r') as f:
            balance_info = json.load(f)
            scale_pos_weight = balance_info.get('scale_pos_weight', 1.0)
            class_weights = balance_info.get('class_weights', {})
            print(f"Using scale_pos_weight from balance summary: {scale_pos_weight:.3f}")
            print(f"Class weights: {class_weights}")
    except Exception:
        class_counts = np.bincount(y_train)
        scale_pos_weight = (
            class_counts[0] / class_counts[1]
            if len(class_counts) > 1 and class_counts[1] > 0
            else 1.0
        )
        print(f"Calculated scale_pos_weight from data: {scale_pos_weight:.3f}")

    # Parameters from config
    cv_folds = snakemake.params.cv_folds
    random_state = snakemake.params.random_state
    param_grid = snakemake.params.param_grid

    grid = {
        'n_estimators': param_grid['n_estimators'],
        'max_depth': param_grid['max_depth'],
        'learning_rate': param_grid['learning_rate'],
        'num_leaves': param_grid['num_leaves']
    }

    base_model = lgb.LGBMClassifier(
        objective='binary',
        random_state=random_state,
        verbose=-1,
        scale_pos_weight=scale_pos_weight
    )

    # Geographic-temporal-aware CV
    outer_cv, cv_groups = get_cross_validator(location_year_train, cv_folds, random_state)
    inner_cv, _ = get_cross_validator(location_year_train, cv_folds, random_state)
    location_year_train = cv_groups

    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X_train, y_train, groups=location_year_train)
    ):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        loc_year_tr = location_year_train[train_idx] if location_year_train is not None else None

        print(
            f"Fold {fold + 1}: "
            f"Train groups={len(np.unique(loc_year_tr)) if loc_year_tr is not None else 'N/A'}, "
            f"Val groups={len(np.unique(location_year_train[val_idx])) if location_year_train is not None else 'N/A'}"
        )

        grid_search = GridSearchCV(
            base_model, grid, cv=3, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_tr, y_tr)

        y_pred = grid_search.predict(X_val)
        y_prob = grid_search.predict_proba(X_val)[:, 1]

        cv_results.append({
            'fold': fold,
            'f1': f1_score(y_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_prob),
            'best_params': grid_search.best_params_
        })

    # Train final model on full training set
    final_model = lgb.LGBMClassifier(
        **cv_results[-1]['best_params'],
        objective='binary',
        random_state=random_state,
        verbose=-1,
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

    # -----------------------------------------------------------------------
    # SHAP feature importance (replaces native split importance)
    # Computed on test set — unbiased estimate of feature contributions
    # -----------------------------------------------------------------------
    importance_df, shap_vals = compute_shap_importance(final_model, X_test, feature_cols)

    # Save results
    results = {
        'cv_results': cv_results,
        'test_results': test_results,
        'cv_mean_f1': np.mean([r['f1'] for r in cv_results]),
        'cv_std_f1': np.std([r['f1'] for r in cv_results]),
        'test_predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_test_pred.tolist(),
            'y_proba': y_test_prob.tolist(),
            'sample_ids': (
                test_merged['sample_id'].tolist()
                if 'sample_id' in test_merged.columns else []
            )
        },
        'shap_info': {
            'method': 'TreeExplainer',
            'target_class': 1,
            'n_samples': int(X_test.shape[0]),
            'n_features': int(X_test.shape[1]),
            'top_feature': importance_df.iloc[0]['feature'],
            'top_feature_shap': float(importance_df.iloc[0]['shap_mean'])
        }
    }

    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2)

    joblib.dump(final_model, snakemake.output.model)
    importance_df.to_csv(snakemake.output.shap, index=False)

    # -----------------------------------------------------------------------
    # Plots: CV performance + SHAP bar chart + feature importance
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: CV F1 scores per fold
    axes[0].bar(range(len(cv_results)), [r['f1'] for r in cv_results], color='steelblue')
    axes[0].set_xlabel('CV Fold')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title(f'LightGBM CV F1 Scores\n{antibiotic}')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(
        np.mean([r['f1'] for r in cv_results]),
        color='red', linestyle='--', label=f"Mean={results['cv_mean_f1']:.3f}"
    )
    axes[0].legend()

    # Plot 2: Top 20 SHAP mean |value| bar chart
    top20 = importance_df.head(20)
    axes[1].barh(
        range(len(top20)), top20['shap_mean'],
        xerr=top20['shap_std'], color='darkorange', alpha=0.8
    )
    axes[1].set_yticks(range(len(top20)))
    axes[1].set_yticklabels(
        [f[:25] + '...' if len(f) > 25 else f for f in top20['feature']],
        fontsize=7
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean |SHAP value|')
    axes[1].set_title('Top 20 Features (SHAP)')

    # Plot 3: Class distribution
    dist = np.bincount(y_train)
    axes[2].bar(['Susceptible (0)', 'Resistant (1)'], dist, color=['steelblue', 'firebrick'], alpha=0.8)
    axes[2].set_ylabel('Sample Count')
    axes[2].set_title('Training Class Distribution')
    for i, v in enumerate(dist):
        axes[2].text(i, v + 1, str(v), ha='center', fontweight='bold')

    plt.suptitle(f'LightGBM — {antibiotic}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(snakemake.output.plots, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCV F1:   {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")
    print(f"Top SHAP feature: {importance_df.iloc[0]['feature']} "
          f"(mean |SHAP| = {importance_df.iloc[0]['shap_mean']:.4f})")


if __name__ == "__main__":
    main()