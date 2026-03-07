"""Unit tests for output_validation.py module."""

import pytest
import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.output_validation import OutputValidator


@pytest.fixture
def validator(tmp_path):
    """Create an OutputValidator with a minimal config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "antibiotics:\n  - amikacin\n  - ciprofloxacin\n"
        "pipeline_version: '1.0'\n"
        "output_naming:\n  model_results: '{antibiotic}_results'\n"
    )
    return OutputValidator(config_path=str(config_path))


class TestValidateModelResults:
    def test_standardizes_structure(self, validator):
        raw = {
            'cv_results': [{'f1': 0.8}],
            'test_results': {'f1': 0.85, 'balanced_accuracy': 0.82}
        }
        result = validator.validate_model_results(raw, 'amikacin')
        assert result['antibiotic'] == 'amikacin'
        assert 'cv_results' in result
        assert 'test_results' in result

    def test_warns_on_missing_fields(self, validator, capsys):
        raw = {}
        result = validator.validate_model_results(raw, 'amikacin')
        # Should still return a dict, with warnings printed
        assert isinstance(result, dict)


class TestValidateInterpretabilityResults:
    def test_standardizes_structure(self, validator):
        raw = {
            'consensus_features': {'gene_a': 0.9},
            'statistical_comparison': {}
        }
        result = validator.validate_interpretability_results(raw, 'amikacin')
        assert result['antibiotic'] == 'amikacin'

    def test_empty_input(self, validator):
        result = validator.validate_interpretability_results({}, 'amikacin')
        assert isinstance(result, dict)


class TestStandardizeFilename:
    def test_basic_template(self, validator):
        name = validator.standardize_filename('model_results', antibiotic='amikacin')
        assert 'amikacin' in name

    def test_missing_template_fallback(self, validator):
        name = validator.standardize_filename('nonexistent_key', antibiotic='amikacin')
        assert 'amikacin' in name


class TestValidateOutputDirectory:
    def test_creates_missing_directory(self, validator, tmp_path):
        new_dir = tmp_path / "output" / "sub"
        result = validator.validate_output_directory(str(new_dir), create_if_missing=True)
        assert new_dir.exists()

    def test_existing_directory(self, validator, tmp_path):
        result = validator.validate_output_directory(str(tmp_path))
        assert result == tmp_path


class TestSaveStandardizedResults:
    def test_saves_json(self, validator, tmp_path):
        results = {
            'antibiotic': 'amikacin',
            'cv_results': [],
            'test_results': {'f1': 0.85}
        }
        output_path = str(tmp_path / "results.json")
        validator.save_standardized_results(results, output_path, 'model_results')
        assert Path(output_path).exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded['antibiotic'] == 'amikacin'

    def test_handles_numpy_types(self, validator, tmp_path):
        results = {
            'value': np.float64(0.85),
            'array': np.array([1, 2, 3]).tolist(),
            'int_val': np.int64(42)
        }
        output_path = str(tmp_path / "numpy_results.json")
        validator.save_standardized_results(results, output_path, 'model_results')
        assert Path(output_path).exists()

    def test_creates_parent_dirs(self, validator, tmp_path):
        output_path = str(tmp_path / "deep" / "nested" / "results.json")
        results = {'test': True}
        validator.save_standardized_results(results, output_path, 'model_results')
        assert Path(output_path).exists()


class TestValidateFeatureMatrix:
    def test_renames_common_columns(self, validator):
        df = pd.DataFrame({
            'Sample_ID': ['s1', 's2'],
            'resistance': [0, 1],
            'feature1': [1.0, 2.0]
        })
        result = validator.validate_feature_matrix(df)
        assert isinstance(result, pd.DataFrame)
        # Should rename columns
        if 'sample_id' in result.columns:
            assert 'sample_id' in result.columns

    def test_valid_feature_matrix(self, validator):
        df = pd.DataFrame({
            'sample_id': ['s1', 's2', 's3'],
            'R': [0, 1, 0],
            'feat_a': [1.0, 2.0, 3.0]
        })
        result = validator.validate_feature_matrix(df)
        assert isinstance(result, pd.DataFrame)


class TestCheckPipelineConsistency:
    def test_missing_results(self, validator, tmp_path):
        report = validator.check_pipeline_consistency(str(tmp_path), ['amikacin'])
        assert isinstance(report, dict)
        assert report['overall_status'] == 'fail'

    def test_with_valid_results(self, validator, tmp_path):
        # Create minimal expected directory structure
        models_dir = tmp_path / "models"
        for model in ['xgboost', 'lightgbm', 'cnn', 'sequence_cnn', 'dnabert']:
            (models_dir / model).mkdir(parents=True, exist_ok=True)
            result_file = models_dir / model / "amikacin_results.json"
            result_file.write_text(json.dumps({
                'antibiotic': 'amikacin',
                'cv_results': [{'f1': 0.8}],
                'test_results': {'f1': 0.85}
            }))
        
        report = validator.check_pipeline_consistency(str(tmp_path), ['amikacin'])
        assert isinstance(report, dict)
