"""
Standardized evaluation metrics for all models.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve
)
import json


def compute_comprehensive_metrics(y_true, y_pred, y_proba=None, average='binary'):
    """
    Compute comprehensive evaluation metrics for model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)
        average: Averaging strategy for multi-class ('binary', 'macro', 'micro')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # AUC score
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    auc_score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_true, y_proba)
            else:
                # Multi-class
                auc_score = roc_auc_score(y_true, y_proba, average=average, multi_class='ovr')
            metrics['auc'] = auc_score
        except ValueError as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
    except Exception as e:
        print(f"Warning: Could not generate classification report: {e}")
        metrics['classification_report'] = None
    
    return metrics


def evaluate_cross_validation_fold(y_true, y_pred, y_proba=None, fold_idx=0):
    """
    Evaluate a single cross-validation fold.
    
    Args:
        y_true: True labels for this fold
        y_pred: Predicted labels for this fold
        y_proba: Predicted probabilities for this fold
        fold_idx: Fold index for logging
        
    Returns:
        Dictionary of fold metrics
    """
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)
    metrics['fold'] = fold_idx
    
    print(f"Fold {fold_idx + 1}: F1={metrics['f1']:.3f}, "
          f"Balanced Acc={metrics['balanced_accuracy']:.3f}, "
          f"AUC={metrics['auc']:.3f if metrics['auc'] is not None else 'N/A'}")
    
    return metrics


def summarize_cross_validation_results(cv_results):
    """
    Summarize results across all CV folds.
    
    Args:
        cv_results: List of fold result dictionaries
        
    Returns:
        Summary dictionary with mean and std metrics
    """
    if not cv_results:
        return {}
    
    # Extract metrics from all folds
    f1_scores = [result['f1'] for result in cv_results if result['f1'] is not None]
    bal_acc_scores = [result['balanced_accuracy'] for result in cv_results if result['balanced_accuracy'] is not None]
    auc_scores = [result['auc'] for result in cv_results if result['auc'] is not None and not np.isnan(result['auc'])]
    
    summary = {
        'cv_mean_f1': np.mean(f1_scores) if f1_scores else 0.0,
        'cv_std_f1': np.std(f1_scores) if f1_scores else 0.0,
        'cv_mean_balanced_accuracy': np.mean(bal_acc_scores) if bal_acc_scores else 0.0,
        'cv_std_balanced_accuracy': np.std(bal_acc_scores) if bal_acc_scores else 0.0,
        'cv_mean_auc': np.mean(auc_scores) if auc_scores else None,
        'cv_std_auc': np.std(auc_scores) if auc_scores else None,
        'n_folds': len(cv_results)
    }
    
    print(f"\nCross-Validation Summary:")
    print(f"Mean F1: {summary['cv_mean_f1']:.3f} ± {summary['cv_std_f1']:.3f}")
    print(f"Mean Balanced Accuracy: {summary['cv_mean_balanced_accuracy']:.3f} ± {summary['cv_std_balanced_accuracy']:.3f}")
    if summary['cv_mean_auc'] is not None:
        print(f"Mean AUC: {summary['cv_mean_auc']:.3f} ± {summary['cv_std_auc']:.3f}")
    
    return summary


def check_success_criteria(test_metrics, cv_summary, f1_threshold=0.85, balanced_acc_threshold=0.85):
    """
    Check if model meets success criteria from project proposal.
    
    Args:
        test_metrics: Test set evaluation metrics
        cv_summary: Cross-validation summary
        f1_threshold: Minimum F1 score (0.85 per proposal)
        balanced_acc_threshold: Minimum balanced accuracy (0.85 per proposal)
        
    Returns:
        Dictionary with success criteria results
    """
    test_f1 = test_metrics.get('f1', 0.0)
    test_bal_acc = test_metrics.get('balanced_accuracy', 0.0)
    cv_f1 = cv_summary.get('cv_mean_f1', 0.0)
    cv_bal_acc = cv_summary.get('cv_mean_balanced_accuracy', 0.0)
    
    criteria = {
        'test_f1_meets_criteria': test_f1 >= f1_threshold,
        'test_balanced_acc_meets_criteria': test_bal_acc >= balanced_acc_threshold,
        'cv_f1_meets_criteria': cv_f1 >= f1_threshold,
        'cv_balanced_acc_meets_criteria': cv_bal_acc >= balanced_acc_threshold,
        'overall_success': (test_f1 >= f1_threshold and cv_f1 >= f1_threshold),
        'discrepancy_analysis_needed': (test_bal_acc >= balanced_acc_threshold and test_f1 < f1_threshold)
    }
    
    print(f"\n=== SUCCESS CRITERIA EVALUATION ===")
    print(f"Test F1: {test_f1:.3f} ({'✅' if criteria['test_f1_meets_criteria'] else '❌'} >= {f1_threshold})")
    print(f"CV F1: {cv_f1:.3f} ({'✅' if criteria['cv_f1_meets_criteria'] else '❌'} >= {f1_threshold})")
    print(f"Test Balanced Acc: {test_bal_acc:.3f} ({'✅' if criteria['test_balanced_acc_meets_criteria'] else '❌'} >= {balanced_acc_threshold})")
    print(f"CV Balanced Acc: {cv_bal_acc:.3f} ({'✅' if criteria['cv_balanced_acc_meets_criteria'] else '❌'} >= {balanced_acc_threshold})")
    
    if criteria['overall_success']:
        print("🎉 SUCCESS: Model meets all criteria!")
    elif criteria['discrepancy_analysis_needed']:
        print("⚠️  DISCREPANCY: Balanced accuracy meets threshold but F1 does not - analysis needed")
    else:
        print("❌ FAILURE: Model does not meet success criteria")
    
    return criteria


def save_standardized_results(cv_results, test_metrics, cv_summary, success_criteria, output_path):
    """
    Save results in standardized format across all models.
    
    Args:
        cv_results: Cross-validation results
        test_metrics: Test set metrics
        cv_summary: CV summary statistics
        success_criteria: Success criteria evaluation
        output_path: Path to save results JSON
    """
    results = {
        'cv_results': cv_results,
        'test_results': test_metrics,
        'cv_summary': cv_summary,
        'success_criteria': success_criteria,
        'pipeline_version': '1.0',
        'evaluation_timestamp': np.datetime64('now').astype(str)
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    return results