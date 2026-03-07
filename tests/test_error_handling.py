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
    @pytest.mark.parametrize(
        "child, parent",
        [
            pytest.param(PipelineError, Exception, id="pipeline_is_exception"),
            pytest.param(ValidationError, PipelineError, id="validation_is_pipeline"),
            pytest.param(DataError, PipelineError, id="data_is_pipeline"),
            pytest.param(ModelError, PipelineError, id="model_is_pipeline"),
        ],
    )
    def test_subclass_hierarchy(self, child, parent):
        assert issubclass(child, parent)

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

    @pytest.mark.parametrize(
        "filename, content, extensions, should_pass",
        [
            pytest.param("data.txt", "hello", [".csv", ".json"], False, id="invalid_ext"),
            pytest.param("data.json", "{}", [".json"], True, id="valid_ext"),
            pytest.param("data.csv", "a,b", [".csv", ".json"], True, id="valid_csv_ext"),
        ],
    )
    def test_extension_validation(self, tmp_path, filename, content, extensions, should_pass):
        """Test that file extension validation accepts/rejects correctly."""
        f = tmp_path / filename
        f.write_text(content)
        if should_pass:
            result = validate_input_file(str(f), extensions=extensions)
            assert result == f
        else:
            with pytest.raises(ValidationError):
                validate_input_file(str(f), extensions=extensions)


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

    @pytest.mark.parametrize(
        "input_val, kwargs, error_msg",
        [
            pytest.param(None, {}, "None input", id="none"),
            pytest.param("not a dataframe", {}, "wrong type", id="wrong_type"),
            pytest.param(
                pd.DataFrame({'a': []}), {"min_rows": 1}, "too few rows", id="empty_df"
            ),
            pytest.param(
                pd.DataFrame({'a': [1]}),
                {"required_columns": ['a', 'b', 'c']},
                "missing cols",
                id="missing_columns",
            ),
        ],
    )
    def test_invalid_dataframe_raises(self, input_val, kwargs, error_msg):
        """Test that invalid inputs raise DataError."""
        with pytest.raises(DataError):
            validate_dataframe(input_val, **kwargs)


# --- validate_model_results ---

class TestValidateModelResults:
    def test_valid_results(self):
        results = {
            'cv_results': [{'f1': 0.8}],
            'test_results': {'f1': 0.85, 'balanced_accuracy': 0.8}
        }
        validated = validate_model_results(results, 'amikacin')
        assert isinstance(validated, dict)

    @pytest.mark.parametrize(
        "results",
        [
            pytest.param({'test_results': {'f1': 0.85}}, id="missing_cv"),
            pytest.param({'cv_results': [{'f1': 0.8}]}, id="missing_test"),
        ],
    )
    def test_missing_required_key_raises(self, results):
        """Test that missing cv_results or test_results raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_model_results(results, 'amikacin')


# --- validate_feature_matrix ---

class TestValidateFeatureMatrix:
    def test_valid_matrix(self):
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)
        result = validate_feature_matrix(X, y)
        assert len(result) == 3  # X, y, feature_names

    @pytest.mark.parametrize(
        "X, y",
        [
            pytest.param(np.random.randn(50, 10), np.random.binomial(1, 0.5, 30), id="length_mismatch"),
            pytest.param(np.array([1, 2, 3]), np.array([0, 1, 0]), id="1d_array"),
        ],
    )
    def test_invalid_matrix_raises(self, X, y):
        """Test that mismatched lengths or 1D arrays raise DataError."""
        with pytest.raises(DataError):
            validate_feature_matrix(X, y)

    def test_feature_names_validation(self):
        X = np.random.randn(10, 3)
        y = np.array([0, 1] * 5)
        result = validate_feature_matrix(X, y, feature_names=['a', 'b', 'c'])
        assert result[2] == ['a', 'b', 'c']


# --- safe_divide / safe_log ---

class TestSafeMath:
    @pytest.mark.parametrize(
        "a, b, default, expected",
        [
            pytest.param(10, 2, 0.0, 5.0, id="normal"),
            pytest.param(10, 0, 0.0, 0.0, id="div_by_zero"),
            pytest.param(10, 0, -1.0, -1.0, id="div_by_zero_custom"),
        ],
    )
    def test_safe_divide(self, a, b, default, expected):
        assert safe_divide(a, b, default=default) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(np.e, 1.0, id="euler"),
            pytest.param(0, 0.0, id="zero"),
            pytest.param(-1, 0.0, id="negative"),
        ],
    )
    def test_safe_log(self, value, expected):
        result = safe_log(value)
        assert abs(result - expected) < 1e-6


# --- validate_config ---

class TestValidateConfig:
    def test_valid_config(self):
        config = {'a': 1, 'b': 2}
        result = validate_config(config, required_keys=['a', 'b'])
        assert result == config

    @pytest.mark.parametrize(
        "config, required_keys",
        [
            pytest.param({'a': 1}, ['a', 'b'], id="missing_key"),
            pytest.param("not a dict", ['a'], id="not_dict"),
        ],
    )
    def test_invalid_config_raises(self, config, required_keys):
        with pytest.raises(ValidationError):
            validate_config(config, required_keys=required_keys)


# --- create_error_summary ---

class TestCreateErrorSummary:
    @pytest.mark.parametrize(
        "errors, total, type_counts",
        [
            pytest.param([], 0, {}, id="empty"),
            pytest.param(
                [ValueError("a"), TypeError("b"), ValueError("c")],
                3,
                {"ValueError": 2, "TypeError": 1},
                id="multiple",
            ),
        ],
    )
    def test_error_summary(self, errors, total, type_counts):
        summary = create_error_summary(errors)
        assert summary['total_errors'] == total
        for etype, count in type_counts.items():
            assert summary['error_types'][etype] == count


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


# --- ErrorContext ---

class TestErrorContext:
    def test_no_error(self):
        with ErrorContext("test_op"):
            pass

    @pytest.mark.parametrize(
        "reraise, exception, should_propagate",
        [
            pytest.param(False, ValueError("suppressed"), False, id="suppress"),
            pytest.param(True, ValueError("should wrap"), True, id="reraise"),
        ],
    )
    def test_error_propagation(self, reraise, exception, should_propagate):
        """Test ErrorContext suppression vs re-raise behaviour."""
        if should_propagate:
            with pytest.raises(ValueError):
                with ErrorContext("test_op", reraise=reraise):
                    raise exception
        else:
            with ErrorContext("test_op", reraise=reraise):
                raise exception
            # Should not propagate

    def test_keyboard_interrupt_always_propagates(self):
        with pytest.raises(KeyboardInterrupt):
            with ErrorContext("test_op", reraise=False):
                raise KeyboardInterrupt()


# --- check_memory_usage ---

class TestCheckMemoryUsage:
    @pytest.mark.parametrize(
        "data_size_mb, available_gb",
        [
            pytest.param(100, 16.0, id="small_data"),
            pytest.param(14000, 16.0, id="large_data_warns"),
        ],
    )
    def test_does_not_raise(self, data_size_mb, available_gb):
        """Both small and large data should not raise (large warns only)."""
        check_memory_usage(data_size_mb=data_size_mb, available_memory_gb=available_gb)
