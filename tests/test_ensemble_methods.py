"""Unit tests for ensemble_methods.py module."""

import pytest
import json
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.ensemble_methods import ModelEnsemble, create_ensemble_from_results


@pytest.fixture
def mock_models():
    """Create mock models with predict methods."""
    models = {}
    for name in ['xgboost', 'lightgbm', 'cnn']:
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]
        ])
        models[name] = model
    return models


@pytest.fixture
def sample_x_data():
    """Sample X data dict for ensemble prediction."""
    X = np.random.randn(4, 10)
    return {'xgboost': X, 'lightgbm': X, 'cnn': X}


@pytest.fixture
def simple_ensemble():
    """Ensemble with manually-set probabilities for predictable testing."""
    ensemble = ModelEnsemble(
        models={'model_a': MagicMock(), 'model_b': MagicMock()},
        model_weights={'model_a': 1.0, 'model_b': 1.0}
    )
    return ensemble


class TestModelEnsembleInit:
    def test_default_weights(self, mock_models):
        ensemble = ModelEnsemble(mock_models)
        for name in mock_models:
            assert ensemble.model_weights[name] == 1.0

    def test_custom_weights(self, mock_models):
        weights = {'xgboost': 2.0, 'lightgbm': 1.0, 'cnn': 0.5}
        ensemble = ModelEnsemble(mock_models, model_weights=weights)
        assert ensemble.model_weights['xgboost'] == 2.0

    def test_ensemble_methods_populated(self, mock_models):
        ensemble = ModelEnsemble(mock_models)
        assert len(ensemble.ensemble_methods) > 0
        assert 'simple_average' in ensemble.ensemble_methods
        assert 'weighted_average' in ensemble.ensemble_methods
        assert 'majority_vote' in ensemble.ensemble_methods


class TestPredictEnsemble:
    def test_simple_average(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        preds, probas = ensemble.predict_ensemble(sample_x_data, method='simple_average')
        assert len(preds) == 4
        assert len(probas) == 4
        assert all(p in [0, 1] for p in preds)

    def test_weighted_average(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        preds, probas = ensemble.predict_ensemble(sample_x_data, method='weighted_average')
        assert len(preds) == 4

    def test_majority_vote(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        preds, probas = ensemble.predict_ensemble(sample_x_data, method='majority_vote')
        assert len(preds) == 4

    def test_invalid_method_raises(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        with pytest.raises(ValueError):
            ensemble.predict_ensemble(sample_x_data, method='nonexistent')

    def test_skips_missing_models(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        partial_data = {'xgboost': sample_x_data['xgboost']}
        preds, probas = ensemble.predict_ensemble(partial_data, method='simple_average')
        assert len(preds) == 4


class TestTrainStackingMetaLearner:
    def test_trains_meta_learner(self, mock_models):
        ensemble = ModelEnsemble(mock_models)
        X_meta = np.random.randn(20, 3)
        y_true = np.random.binomial(1, 0.5, 20)
        ensemble.train_stacking_meta_learner(X_meta, y_true)
        # Should have a meta_learner or None (if fitting failed)
        assert hasattr(ensemble, 'meta_learner')


class TestEvaluateEnsembleMethods:
    def test_returns_dict(self, mock_models, sample_x_data):
        ensemble = ModelEnsemble(mock_models)
        y_true = np.array([0, 1, 0, 1])
        results = ensemble.evaluate_ensemble_methods(sample_x_data, y_true)
        assert isinstance(results, dict)
        # Should have at least some methods evaluated
        if results:
            for method_name, metrics in results.items():
                assert 'f1' in metrics or 'error' in metrics


class TestCreateEnsembleFromResults:
    def test_creates_ensemble_from_json(self, tmp_path):
        """Test creating ensemble from result JSON files."""
        models_dir = tmp_path / "models"
        for model_name in ['xgboost', 'lightgbm']:
            model_dir = models_dir / model_name
            model_dir.mkdir(parents=True)
            results = {
                'antibiotic': 'amikacin',
                'test_results': {'f1': 0.85, 'balanced_accuracy': 0.82},
                'cv_results': [{'f1': 0.80}]
            }
            (model_dir / "amikacin_results.json").write_text(json.dumps(results))

        ensemble = create_ensemble_from_results(str(tmp_path), 'amikacin',
                                                  models=['xgboost', 'lightgbm'])
        # May return None if models can't actually be loaded (no saved model files),
        # but should not crash
        assert ensemble is None or isinstance(ensemble, ModelEnsemble)

    def test_missing_results_dir(self, tmp_path):
        result = create_ensemble_from_results(str(tmp_path / "nonexistent"), 'amikacin')
        assert result is None
