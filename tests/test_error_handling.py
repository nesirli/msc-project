"""Unit tests for error_handling.py module."""

import pytest
import logging
import tempfile
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.error_handling import (
    PipelineError, ValidationError, DataError, ModelError,
    setup_logging, handle_errors, validate_input_file, validate_output_dir,
    validate_dataframe, validate_model_results, validate_feature_matrix,
    safe_divide, safe_log, validate_config, create_error_summary,
    ProgressTracker, ErrorContext, check_memory_usage,
)


# --- Exception hierarchy ---

class TestExceptionHierarchy:
    def test_pipeline_error_is_exception(self):
        assert issubclass(PipelineError, Exception)

    def test_validation_error_is_pipeline_error(self):
        assert issubclass(ValidationError, PipelineError)

    def test_data_error_is_pipeline_error(self):
        assert issubclass(DataError, PipelineError)

    def test_model_error_is_pipeline_error(self):
        assert issubclass(ModelError, PipelineError)

    def test_pipeline_error_message(self):
        err = PipelineError("test message")
        assert str(err) == "test message"


# --- setup_logging ---

class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'amr_pipeline'

    def test_with_log_file(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "test.log")
        logger = setup_logging(log_file=log_file)
        assert isinstance(logger, logging.Logger)
        # Log file parent directory should be created
        assert (tmp_path / "subdir").exists()


# --- handle_errors decorator ---

class TestHandleErrors:
    def test_success_passthrough(self):
        @handle_errors()
        def good_func():
            return 42
        assert good_func() == 42

    def test_reraise_wraps_in_pipeline_error(self):
        @handle_errors(reraise=True)
        def bad_func():
            raise ValueError("oops")
        with pytest.raises(PipelineError):
            bad_func()

    def test_no_reraise_returns_default(self):
        @handle_errors(reraise=False, default_return=-1)
        def bad_func():
            raise ValueError("oops")
        assert bad_func() == -1

    def test_keyboard_interrupt_always_raised(self):
        @handle_errors(reraise=False, default_return=None)
        def interrupted():
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            interrupted()


# --- validate_input_file ---

class TestValidateInputFile:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        result = validate_input_file(str(f))
        assert result == f

    def test_missing_required_file(self, tmp_path):
        with pytest.raises(ValidationError):
            validate_input_file(str(tmp_path / "nonexistent.csv"), required=True)

    def test_missing_optional_file(self, tmp_path):
        result = validate_input_file(str(tmp_path / "nonexistent.csv"), required=False)
        assert result is None or isinstance(result, Path)

    def test_invalid_extension(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValidationError):
            validate_input_file(str(f), extensions=['.csv', '.json'])

    def test_valid_extension(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        result = validate_input_file(str(f), extensions=['.json'])
        assert result == f


# --- validate_output_dir ---

class TestValidateOutputDir:
    def test_create_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "sub"
        result = validate_output_dir(str(new_dir), create=True)
        assert new_dir.exists()

    def test_no_create_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            validate_output_dir(str(tmp_path / "nope"), create=False)

    def test_existing_directory(self, tmp_path):
        result = validate_output_dir(str(tmp_path))
        assert result == tmp_path


# --- validate_dataframe ---

class TestValidateDataframe:
    def test_valid_dataframe(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = validate_dataframe(df, required_columns=['a', 'b'])
        assert isinstance(result, pd.DataFrame)

    def test_none_raises(self):
        with pytest.raises(DataError):
            validate_dataframe(None)

    def test_wrong_type_raises(self):
        with pytest.raises(DataError):
            validate_dataframe("not a dataframe")

    def test_too_few_rows(self):
        df = pd.DataFrame({'a': []})
        with pytest.raises(DataError):
            validate_dataframe(df, min_rows=1)

    def test_missing_columns(self):
        df = pd.DataFrame({'a': [1]})
        with pytest.raises(DataError):
            validate_dataframe(df, required_columns=['a', 'b', 'c'])


# --- validate_model_results ---

class TestValidateModelResults:
    def test_valid_results(self):
        results = {
            'cv_results': [{'f1': 0.8}],
            'test_results': {'f1': 0.85, 'balanced_accuracy': 0.8}
        }
        validated = validate_model_results(results, 'amikacin')
        assert isinstance(validated, dict)

    def test_missing_cv_results(self):
        results = {'test_results': {'f1': 0.85}}
        with pytest.raises(ValidationError):
            validate_model_results(results, 'amikacin')

    def test_missing_test_results(self):
        results = {'cv_results': [{'f1': 0.8}]}
        with pytest.raises(ValidationError):
            validate_model_results(results, 'amikacin')


# --- validate_feature_matrix ---

class TestValidateFeatureMatrix:
    def test_valid_matrix(self):
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)
        result = validate_feature_matrix(X, y)
        assert len(result) == 3  # X, y, feature_names

    def test_mismatched_lengths(self):
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 30)
        with pytest.raises(DataError):
            validate_feature_matrix(X, y)

    def test_1d_array_raises(self):
        X = np.array([1, 2, 3])
        y = np.array([0, 1, 0])
        with pytest.raises(DataError):
            validate_feature_matrix(X, y)

    def test_feature_names_validation(self):
        X = np.random.randn(10, 3)
        y = np.array([0, 1] * 5)
        result = validate_feature_matrix(X, y, feature_names=['a', 'b', 'c'])
        assert result[2] == ['a', 'b', 'c']


# --- safe_divide / safe_log ---

class TestSafeMath:
    def test_safe_divide_normal(self):
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_safe_divide_by_zero_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_safe_log_normal(self):
        result = safe_log(np.e)
        assert abs(result - 1.0) < 1e-6

    def test_safe_log_zero(self):
        assert safe_log(0) == 0.0

    def test_safe_log_negative(self):
        assert safe_log(-1) == 0.0


# --- validate_config ---

class TestValidateConfig:
    def test_valid_config(self):
        config = {'a': 1, 'b': 2}
        result = validate_config(config, required_keys=['a', 'b'])
        assert result == config

    def test_missing_key(self):
        config = {'a': 1}
        with pytest.raises(ValidationError):
            validate_config(config, required_keys=['a', 'b'])

    def test_not_dict(self):
        with pytest.raises(ValidationError):
            validate_config("not a dict", required_keys=['a'])


# --- create_error_summary ---

class TestCreateErrorSummary:
    def test_empty_errors(self):
        summary = create_error_summary([])
        assert summary['total_errors'] == 0

    def test_multiple_errors(self):
        errors = [ValueError("a"), TypeError("b"), ValueError("c")]
        summary = create_error_summary(errors)
        assert summary['total_errors'] == 3
        assert summary['error_types']['ValueError'] == 2
        assert summary['error_types']['TypeError'] == 1


# --- ProgressTracker ---

class TestProgressTracker:
    def test_initialization(self):
        tracker = ProgressTracker(total_steps=10, description="Test")
        assert tracker.current_step == 0

    def test_update(self):
        tracker = ProgressTracker(total_steps=10)
        tracker.update(5)
        assert tracker.current_step == 5

    def test_finish(self):
        tracker = ProgressTracker(total_steps=10)
        tracker.update(5)
        tracker.finish()
        # Should not raise


# --- ErrorContext ---

class TestErrorContext:
    def test_no_error(self):
        with ErrorContext("test_op"):
            pass  # Should complete fine

    def test_suppresses_when_no_reraise(self):
        with ErrorContext("test_op", reraise=False):
            raise ValueError("suppressed")
        # Should not propagate

    def test_reraises_when_configured(self):
        with pytest.raises(ValueError):
            with ErrorContext("test_op", reraise=True):
                raise ValueError("should wrap")

    def test_keyboard_interrupt_always_propagates(self):
        with pytest.raises(KeyboardInterrupt):
            with ErrorContext("test_op", reraise=False):
                raise KeyboardInterrupt()


# --- check_memory_usage ---

class TestCheckMemoryUsage:
    def test_small_data(self):
        # Should not raise for small data
        check_memory_usage(data_size_mb=100, available_memory_gb=16.0)

    def test_large_data_warns(self):
        # Should warn but not raise
        check_memory_usage(data_size_mb=14000, available_memory_gb=16.0)
