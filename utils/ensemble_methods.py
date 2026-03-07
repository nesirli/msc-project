"""
Ensemble model combination methods for AMR prediction.
Combines predictions from XGBoost, LightGBM, CNN, Sequence CNN, and DNABERT models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from scipy.stats import mode
from typing import Dict, List, Tuple, Any
import warnings


class ModelEnsemble:
    """Ensemble methods for combining multiple AMR prediction models."""
    
    def __init__(self, models: Dict[str, Any], model_weights: Dict[str, float] = None):
        """
        Initialize ensemble with trained models.
        
        Args:
            models: Dictionary of {model_name: model_object}
            model_weights: Optional weights for each model based on CV performance
        """
        self.models = models
        self.model_weights = model_weights or {name: 1.0 for name in models.keys()}
        self.ensemble_methods = {
            'simple_average': self._simple_average,
            'weighted_average': self._weighted_average,
            'majority_vote': self._majority_vote,
            'weighted_vote': self._weighted_vote,
            'stacking': self._stacking_ensemble,
            'rank_average': self._rank_average
        }
    
    def predict_ensemble(self, X_data: Dict[str, np.ndarray], method: str = 'weighted_average') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions using specified method.
        
        Args:
            X_data: Dictionary of {model_name: input_data}
            method: Ensemble method to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if method not in self.ensemble_methods:
            raise ValueError(f"Unknown ensemble method: {method}. Available: {list(self.ensemble_methods.keys())}")
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            if model_name in X_data:
                X = X_data[model_name]
                
                try:
                    # Get predictions
                    if hasattr(model, 'predict'):
                        preds = model.predict(X)
                    else:
                        # For dummy models or edge cases
                        preds = np.zeros(X.shape[0])
                    
                    # Get probabilities
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if len(proba.shape) > 1 and proba.shape[1] > 1:
                            proba = proba[:, 1]  # Positive class probability
                        else:
                            proba = proba.flatten()
                    else:
                        # Fallback: use predictions as probabilities
                        proba = preds.astype(float)
                    
                    model_predictions[model_name] = preds
                    model_probabilities[model_name] = proba
                    
                except Exception as e:
                    warnings.warn(f"Error getting predictions from {model_name}: {e}")
                    # Use dummy predictions
                    model_predictions[model_name] = np.zeros(X.shape[0])
                    model_probabilities[model_name] = np.zeros(X.shape[0])
        
        # Apply ensemble method
        ensemble_preds, ensemble_proba = self.ensemble_methods[method](model_predictions, model_probabilities)
        
        return ensemble_preds, ensemble_proba
    
    def _simple_average(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Simple average of model probabilities."""
        if not probabilities:
            return np.array([]), np.array([])
        
        # Stack probabilities
        proba_matrix = np.column_stack(list(probabilities.values()))
        
        # Simple average
        avg_proba = np.mean(proba_matrix, axis=1)
        avg_preds = (avg_proba > 0.5).astype(int)
        
        return avg_preds, avg_proba
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted average based on model performance."""
        if not probabilities:
            return np.array([]), np.array([])
        
        # Get weights for available models
        weights = np.array([self.model_weights.get(name, 1.0) for name in probabilities.keys()])
        weights = weights / np.sum(weights)  # Normalize
        
        # Stack probabilities
        proba_matrix = np.column_stack(list(probabilities.values()))
        
        # Weighted average
        weighted_proba = np.average(proba_matrix, axis=1, weights=weights)
        weighted_preds = (weighted_proba > 0.5).astype(int)
        
        return weighted_preds, weighted_proba
    
    def _majority_vote(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Simple majority vote of predictions."""
        if not predictions:
            return np.array([]), np.array([])
        
        # Stack predictions
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Majority vote
        majority_preds = mode(pred_matrix, axis=1)[0].flatten()
        
        # Use average probabilities for probability output
        if probabilities:
            proba_matrix = np.column_stack(list(probabilities.values()))
            avg_proba = np.mean(proba_matrix, axis=1)
        else:
            avg_proba = majority_preds.astype(float)
        
        return majority_preds, avg_proba
    
    def _weighted_vote(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted vote based on model confidence and performance."""
        if not predictions or not probabilities:
            return np.array([]), np.array([])
        
        # Get weights
        model_names = list(predictions.keys())
        weights = np.array([self.model_weights.get(name, 1.0) for name in model_names])
        
        # Weight predictions by model performance and confidence
        pred_matrix = np.column_stack(list(predictions.values()))
        proba_matrix = np.column_stack(list(probabilities.values()))
        
        # Calculate confidence (distance from 0.5)
        confidence_matrix = np.abs(proba_matrix - 0.5)
        
        # Combine performance weights with confidence
        combined_weights = weights * confidence_matrix.T
        
        # Weighted vote
        weighted_votes = np.sum(pred_matrix * combined_weights.T, axis=1)
        total_weights = np.sum(combined_weights.T, axis=1)
        total_weights[total_weights == 0] = 1  # Avoid division by zero
        
        weighted_preds = (weighted_votes / total_weights > 0.5).astype(int)
        weighted_proba = weighted_votes / total_weights
        
        return weighted_preds, weighted_proba
    
    def _stacking_ensemble(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Stacking ensemble using logistic regression meta-learner."""
        if not probabilities:
            return np.array([]), np.array([])
        
        # Use probabilities as features for meta-learner
        X_meta = np.column_stack(list(probabilities.values()))
        
        # If we don't have a trained meta-learner, fall back to weighted average
        if not hasattr(self, 'meta_learner') or self.meta_learner is None:
            warnings.warn("No trained meta-learner found, falling back to weighted average")
            return self._weighted_average(predictions, probabilities)
        
        # Predict using meta-learner
        try:
            stacked_preds = self.meta_learner.predict(X_meta)
            stacked_proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        except:
            # Fallback to weighted average if stacking fails
            return self._weighted_average(predictions, probabilities)
        
        return stacked_preds, stacked_proba
    
    def _rank_average(self, predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Rank-based average to handle different probability scales."""
        if not probabilities:
            return np.array([]), np.array([])
        
        # Convert probabilities to ranks
        proba_matrix = np.column_stack(list(probabilities.values()))
        n_samples = proba_matrix.shape[0]
        
        # Convert to ranks (0 to 1 scale)
        rank_matrix = np.zeros_like(proba_matrix)
        for i, col in enumerate(proba_matrix.T):
            sorted_indices = np.argsort(col)
            ranks = np.zeros_like(col)
            ranks[sorted_indices] = np.arange(len(col)) / (len(col) - 1) if len(col) > 1 else 0.5
            rank_matrix[:, i] = ranks
        
        # Average ranks with weights
        weights = np.array([self.model_weights.get(name, 1.0) for name in probabilities.keys()])
        weights = weights / np.sum(weights)
        
        rank_avg = np.average(rank_matrix, axis=1, weights=weights)
        rank_preds = (rank_avg > 0.5).astype(int)
        
        return rank_preds, rank_avg
    
    def train_stacking_meta_learner(self, X_meta: np.ndarray, y_true: np.ndarray):
        """
        Train meta-learner for stacking ensemble.
        
        Args:
            X_meta: Meta-features (model probabilities) from cross-validation
            y_true: True labels
        """
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        try:
            self.meta_learner.fit(X_meta, y_true)
            print(f"Trained meta-learner with {X_meta.shape[1]} model features")
        except Exception as e:
            warnings.warn(f"Failed to train meta-learner: {e}")
            self.meta_learner = None
    
    def evaluate_ensemble_methods(self, X_data: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all ensemble methods on test data.
        
        Args:
            X_data: Dictionary of {model_name: input_data}
            y_true: True labels
            
        Returns:
            Dictionary of {method: {metric: value}}
        """
        results = {}
        
        for method_name in self.ensemble_methods.keys():
            try:
                preds, proba = self.predict_ensemble(X_data, method=method_name)
                
                if len(preds) > 0 and len(y_true) > 0:
                    # Calculate metrics
                    f1 = f1_score(y_true, preds, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true, preds)
                    
                    try:
                        auc = roc_auc_score(y_true, proba)
                    except:
                        auc = None
                    
                    results[method_name] = {
                        'f1': f1,
                        'balanced_accuracy': bal_acc,
                        'auc': auc,
                        'n_predictions': len(preds)
                    }
                else:
                    results[method_name] = {
                        'f1': 0.0,
                        'balanced_accuracy': 0.5,
                        'auc': None,
                        'n_predictions': 0
                    }
            
            except Exception as e:
                warnings.warn(f"Error evaluating {method_name}: {e}")
                results[method_name] = {
                    'f1': 0.0,
                    'balanced_accuracy': 0.5,
                    'auc': None,
                    'error': str(e)
                }
        
        return results


def create_ensemble_from_results(results_dir: str, antibiotic: str, models: List[str] = None) -> ModelEnsemble:
    """
    Create ensemble from saved model results.
    
    Args:
        results_dir: Results directory path
        antibiotic: Antibiotic name
        models: List of model names to include
        
    Returns:
        Configured ModelEnsemble
    """
    from pathlib import Path
    import json
    
    if models is None:
        models = ['xgboost', 'lightgbm', 'cnn', 'sequence_cnn', 'dnabert']
    
    # Load model performance to calculate weights
    model_weights = {}
    loaded_models = {}
    
    for model_name in models:
        results_file = Path(results_dir) / 'models' / model_name / f'{antibiotic}_results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract F1 score for weighting
                cv_f1 = results.get('cv_mean_f1', 0.0)
                test_f1 = results.get('test_results', {}).get('f1', 0.0)
                
                # Use average of CV and test F1, with minimum weight
                avg_f1 = (cv_f1 + test_f1) / 2.0
                model_weights[model_name] = max(avg_f1, 0.1)  # Minimum weight to avoid zero
                
                # Create dummy model object (actual models would need to be loaded from pickle)
                class DummyModel:
                    def __init__(self, results):
                        self.results = results
                    
                    def predict(self, X):
                        # Dummy prediction - in real use, load actual model
                        return np.zeros(X.shape[0])
                    
                    def predict_proba(self, X):
                        # Dummy probabilities
                        return np.random.random(X.shape[0])
                
                loaded_models[model_name] = DummyModel(results)
                
            except Exception as e:
                warnings.warn(f"Could not load results for {model_name}: {e}")
    
    if not loaded_models:
        warnings.warn(f"No models loaded for {antibiotic}")
        return None
    
    print(f"Created ensemble for {antibiotic} with {len(loaded_models)} models")
    print(f"Model weights: {model_weights}")
    
    return ModelEnsemble(loaded_models, model_weights)


def evaluate_ensemble_performance(results_dir: str, antibiotics: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate ensemble methods across all antibiotics.
    
    Args:
        results_dir: Results directory path
        antibiotics: List of antibiotics to evaluate
        
    Returns:
        Comprehensive ensemble evaluation results
    """
    if antibiotics is None:
        antibiotics = ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']
    
    ensemble_results = {
        'antibiotics': {},
        'method_comparison': {},
        'best_methods': {},
        'summary': {}
    }
    
    all_method_scores = {method: [] for method in ['simple_average', 'weighted_average', 'majority_vote', 'weighted_vote', 'rank_average']}
    
    for antibiotic in antibiotics:
        print(f"\nEvaluating ensemble methods for {antibiotic}...")
        
        ensemble = create_ensemble_from_results(results_dir, antibiotic)
        
        if ensemble is not None:
            # Create dummy test data for evaluation
            n_test_samples = 10  # Would use real test data in practice
            X_dummy = {
                'xgboost': np.random.random((n_test_samples, 179)),
                'lightgbm': np.random.random((n_test_samples, 179)),
                'cnn': np.random.random((n_test_samples, 1000)),
                'sequence_cnn': np.random.random((n_test_samples, 4, 250)),
                'dnabert': np.random.random((n_test_samples, 250))
            }
            y_dummy = np.random.randint(0, 2, n_test_samples)
            
            # Evaluate ensemble methods
            method_results = ensemble.evaluate_ensemble_methods(X_dummy, y_dummy)
            ensemble_results['antibiotics'][antibiotic] = method_results
            
            # Track best method for this antibiotic
            best_method = max(method_results.keys(), key=lambda k: method_results[k].get('f1', 0))
            ensemble_results['best_methods'][antibiotic] = {
                'method': best_method,
                'f1': method_results[best_method].get('f1', 0),
                'balanced_accuracy': method_results[best_method].get('balanced_accuracy', 0.5)
            }
            
            # Accumulate scores for method comparison
            for method, scores in method_results.items():
                if method in all_method_scores:
                    all_method_scores[method].append(scores.get('f1', 0))
    
    # Calculate average performance across antibiotics
    for method, scores in all_method_scores.items():
        if scores:
            ensemble_results['method_comparison'][method] = {
                'mean_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'min_f1': np.min(scores),
                'max_f1': np.max(scores)
            }
    
    # Overall summary
    if ensemble_results['method_comparison']:
        best_overall_method = max(ensemble_results['method_comparison'].keys(), 
                                key=lambda k: ensemble_results['method_comparison'][k]['mean_f1'])
        ensemble_results['summary'] = {
            'best_overall_method': best_overall_method,
            'best_mean_f1': ensemble_results['method_comparison'][best_overall_method]['mean_f1'],
            'antibiotics_evaluated': len([ab for ab in antibiotics if ab in ensemble_results['antibiotics']])
        }
    
    return ensemble_results


if __name__ == "__main__":
    # Example usage
    print("=== ENSEMBLE METHOD EVALUATION ===")
    
    # Evaluate ensemble performance
    results = evaluate_ensemble_performance("results")
    
    print(f"\nBest overall ensemble method: {results['summary'].get('best_overall_method', 'None')}")
    print(f"Mean F1 score: {results['summary'].get('best_mean_f1', 0):.3f}")
    
    # Show method comparison
    if results['method_comparison']:
        print("\nMethod Comparison (F1 scores):")
        for method, scores in results['method_comparison'].items():
            print(f"  {method}: {scores['mean_f1']:.3f} ± {scores['std_f1']:.3f}")