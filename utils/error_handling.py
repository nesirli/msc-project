"""
Standardized error handling and validation utilities for the AMR prediction pipeline.
Provides consistent error handling, logging, and validation across all scripts.
"""

import functools
import traceback
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd


class PipelineError(Exception):
    """Custom exception for pipeline-specific errors."""
    pass


class ValidationError(PipelineError):
    """Exception for validation failures."""
    pass


class DataError(PipelineError):
    """Exception for data-related errors."""
    pass


class ModelError(PipelineError):
    """Exception for model-related errors."""
    pass


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up standardized logging for pipeline scripts.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('amr_pipeline')
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set level
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def handle_errors(reraise: bool = True, default_return: Any = None):
    """
    Decorator for standardized error handling in pipeline functions.
    
    Args:
        reraise: Whether to reraise exceptions after logging
        default_return: Default return value if error occurs and reraise=False
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('amr_pipeline')
            
            try:
                logger.info(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__} successfully")
                return result
                
            except KeyboardInterrupt:
                logger.error(f"User interrupted {func.__name__}")
                raise
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                
                if reraise:
                    raise PipelineError(f"Failed in {func.__name__}: {str(e)}") from e
                else:
                    logger.warning(f"Returning default value due to error in {func.__name__}")
                    return default_return
        
        return wrapper
    return decorator


def validate_input_file(file_path: Union[str, Path], required: bool = True, 
                       extensions: Optional[List[str]] = None) -> Path:
    """
    Validate input file exists and has correct extension.
    
    Args:
        file_path: Path to file
        required: Whether file is required to exist
        extensions: List of allowed extensions (e.g., ['.csv', '.json'])
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    path = Path(file_path)
    
    if required and not path.exists():
        raise ValidationError(f"Required input file not found: {path}")
    
    if extensions and path.suffix.lower() not in [ext.lower() for ext in extensions]:
        raise ValidationError(f"File {path} has invalid extension. Expected one of: {extensions}")
    
    if path.exists() and path.stat().st_size == 0:
        warnings.warn(f"Input file is empty: {path}")
    
    return path


def validate_output_dir(dir_path: Union[str, Path], create: bool = True) -> Path:
    """
    Validate and optionally create output directory.
    
    Args:
        dir_path: Path to directory
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Validated Path object
    """
    path = Path(dir_path)
    
    if not path.exists() and create:
        path.mkdir(parents=True, exist_ok=True)
        logging.getLogger('amr_pipeline').info(f"Created output directory: {path}")
    elif not path.exists():
        raise ValidationError(f"Output directory does not exist: {path}")
    
    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    return path


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                      min_rows: int = 1, name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name for error messages
        
    Returns:
        Validated DataFrame
        
    Raises:
        DataError: If validation fails
    """
    logger = logging.getLogger('amr_pipeline')
    
    if df is None:
        raise DataError(f"{name} is None")
    
    if not isinstance(df, pd.DataFrame):
        raise DataError(f"{name} is not a DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise DataError(f"{name} has {len(df)} rows, minimum required: {min_rows}")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataError(f"{name} missing required columns: {missing_cols}")
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        warnings.warn(f"{name} has completely empty columns: {empty_cols}")
    
    logger.info(f"Validated {name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_model_results(results: Dict[str, Any], antibiotic: str) -> Dict[str, Any]:
    """
    Validate model results structure.
    
    Args:
        results: Model results dictionary
        antibiotic: Antibiotic name for context
        
    Returns:
        Validated results
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['cv_results', 'test_results']
    
    for field in required_fields:
        if field not in results:
            raise ValidationError(f"Model results for {antibiotic} missing required field: {field}")
    
    # Validate CV results
    cv_results = results['cv_results']
    if not isinstance(cv_results, list) or len(cv_results) == 0:
        warnings.warn(f"CV results for {antibiotic} is empty or invalid")
    
    # Validate test results
    test_results = results['test_results']
    if not isinstance(test_results, dict):
        raise ValidationError(f"Test results for {antibiotic} must be a dictionary")
    
    required_metrics = ['f1', 'balanced_accuracy']
    for metric in required_metrics:
        if metric not in test_results:
            warnings.warn(f"Test results for {antibiotic} missing metric: {metric}")
    
    return results


def validate_feature_matrix(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> tuple:
    """
    Validate feature matrix and labels.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Optional feature names
        
    Returns:
        Tuple of (X, y, feature_names)
        
    Raises:
        DataError: If validation fails
    """
    logger = logging.getLogger('amr_pipeline')
    
    # Validate X
    if not isinstance(X, np.ndarray):
        raise DataError(f"Feature matrix must be numpy array, got {type(X)}")
    
    if X.ndim != 2:
        raise DataError(f"Feature matrix must be 2D, got shape {X.shape}")
    
    if np.any(np.isnan(X)):
        warnings.warn(f"Feature matrix contains NaN values: {np.sum(np.isnan(X))} total")
    
    if np.any(np.isinf(X)):
        warnings.warn(f"Feature matrix contains infinite values: {np.sum(np.isinf(X))} total")
    
    # Validate y
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if len(y) != X.shape[0]:
        raise DataError(f"Label length {len(y)} doesn't match feature matrix rows {X.shape[0]}")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    if len(unique_classes) < 2:
        warnings.warn(f"Only single class present in labels: {unique_classes}")
    
    # Validate feature names
    if feature_names is not None:
        if len(feature_names) != X.shape[1]:
            raise DataError(f"Feature names length {len(feature_names)} doesn't match matrix columns {X.shape[1]}")
    
    logger.info(f"Validated feature matrix: {X.shape}, labels: {y.shape}")
    return X, y, feature_names


def check_memory_usage(data_size_mb: float, available_memory_gb: float = 16.0):
    """
    Check if data size exceeds available memory.
    
    Args:
        data_size_mb: Size of data in MB
        available_memory_gb: Available memory in GB
        
    Raises:
        Warning if memory usage is high
    """
    available_mb = available_memory_gb * 1024
    usage_pct = (data_size_mb / available_mb) * 100
    
    if usage_pct > 80:
        warnings.warn(f"High memory usage: {usage_pct:.1f}% ({data_size_mb:.1f} MB)")
    elif usage_pct > 50:
        logging.getLogger('amr_pipeline').info(f"Memory usage: {usage_pct:.1f}% ({data_size_mb:.1f} MB)")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def safe_log(x: float, base: float = np.e, default: float = 0.0) -> float:
    """
    Safely compute logarithm, handling edge cases.
    
    Args:
        x: Input value
        base: Logarithm base
        default: Default value for invalid inputs
        
    Returns:
        Logarithm or default value
    """
    try:
        if x <= 0:
            return default
        if base == np.e:
            return np.log(x)
        else:
            return np.log(x) / np.log(base)
    except (TypeError, ValueError):
        return default


class ProgressTracker:
    """Simple progress tracking for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.logger = logging.getLogger('amr_pipeline')
        
    def update(self, steps: int = 1, message: str = None):
        """Update progress."""
        self.current_step += steps
        progress_pct = (self.current_step / self.total_steps) * 100
        
        if message:
            self.logger.info(f"{self.description}: {progress_pct:.1f}% - {message}")
        else:
            self.logger.info(f"{self.description}: {progress_pct:.1f}% ({self.current_step}/{self.total_steps})")
    
    def finish(self, message: str = "Completed"):
        """Mark as finished."""
        self.logger.info(f"{self.description}: 100.0% - {message}")


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        Validated config
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Config must be dictionary, got {type(config)}")
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValidationError(f"Config missing required keys: {missing_keys}")
    
    return config


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in pipeline sections."""
    
    def __init__(self, operation_name: str, reraise: bool = True):
        self.operation_name = operation_name
        self.reraise = reraise
        self.logger = logging.getLogger('amr_pipeline')
    
    def __enter__(self):
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} successfully")
            return False
        
        self.logger.error(f"Error in {self.operation_name}: {exc_val}")
        
        if exc_type == KeyboardInterrupt:
            self.logger.info("Operation interrupted by user")
            return False
        
        if not self.reraise:
            self.logger.warning(f"Suppressing error in {self.operation_name}")
            return True
        
        return False


# Example usage functions
def validate_snakemake_params(snakemake_obj):
    """Validate snakemake object has required attributes."""
    required_attrs = ['input', 'output', 'params']
    
    for attr in required_attrs:
        if not hasattr(snakemake_obj, attr):
            raise ValidationError(f"Snakemake object missing attribute: {attr}")
    
    return snakemake_obj


def create_error_summary(errors: List[Exception]) -> Dict[str, Any]:
    """Create summary of errors for reporting."""
    summary = {
        'total_errors': len(errors),
        'error_types': {},
        'error_messages': []
    }
    
    for error in errors:
        error_type = type(error).__name__
        summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
        summary['error_messages'].append(str(error))
    
    return summary


if __name__ == "__main__":
    # Example usage
    logger = setup_logging('logs/test.log')
    
    @handle_errors()
    def test_function():
        logger.info("This is a test function")
        return "success"
    
    # Test validation functions
    try:
        validate_input_file("nonexistent.csv", required=True)
    except ValidationError as e:
        logger.error(f"Expected validation error: {e}")
    
    # Test context manager
    with ErrorContext("test operation"):
        logger.info("Inside error context")
    
    logger.info("Error handling tests completed")