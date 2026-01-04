"""
Output format validation and standardization utilities.
Ensures consistent file naming and output structure across all pipeline stages.
"""

import json
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
import warnings


class OutputValidator:
    """Validate and standardize pipeline output formats."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.naming_conventions = self.config.get('naming', {})
        self.output_format = self.config.get('output_format', {})
    
    def validate_model_results(self, results_dict: Dict[str, Any], antibiotic: str) -> Dict[str, Any]:
        """
        Validate and standardize model results format.
        
        Args:
            results_dict: Raw model results
            antibiotic: Antibiotic name
            
        Returns:
            Standardized results dictionary
        """
        required_fields = self.output_format.get('model_results', {}).get('required_fields', [])
        required_metrics = self.output_format.get('model_results', {}).get('metrics', [])
        
        # Validate required top-level fields
        missing_fields = [field for field in required_fields if field not in results_dict]
        if missing_fields:
            warnings.warn(f"Missing required fields in {antibiotic} results: {missing_fields}")
        
        # Standardize structure
        standardized = {
            'antibiotic': antibiotic,
            'pipeline_version': '1.0',
            'cv_results': results_dict.get('cv_results', []),
            'test_results': results_dict.get('test_results', {}),
            'cv_summary': results_dict.get('cv_summary', {}),
            'success_criteria': results_dict.get('success_criteria', {}),
            'model_params': results_dict.get('model_params', {}),
            'training_info': {
                'n_train_samples': results_dict.get('n_train_samples', 0),
                'n_test_samples': results_dict.get('n_test_samples', 0),
                'n_features': results_dict.get('n_features', 0),
                'class_distribution': results_dict.get('class_distribution', {}),
                'cross_validation_strategy': results_dict.get('cv_strategy', 'unknown')
            }
        }
        
        # Validate metrics in CV and test results
        self._validate_metrics(standardized['cv_results'], required_metrics, 'CV')
        self._validate_metrics([standardized['test_results']], required_metrics, 'test')
        
        return standardized
    
    def validate_interpretability_results(self, results_dict: Dict[str, Any], antibiotic: str) -> Dict[str, Any]:
        """
        Validate and standardize interpretability analysis format.
        
        Args:
            results_dict: Raw interpretability results
            antibiotic: Antibiotic name
            
        Returns:
            Standardized interpretability dictionary
        """
        required_fields = self.output_format.get('interpretability', {}).get('required_fields', [])
        feature_types = self.output_format.get('interpretability', {}).get('feature_types', [])
        
        # Validate required fields
        missing_fields = [field for field in required_fields if field not in results_dict]
        if missing_fields:
            warnings.warn(f"Missing required fields in {antibiotic} interpretability: {missing_fields}")
        
        # Standardize structure
        standardized = {
            'antibiotic': antibiotic,
            'pipeline_version': '1.0',
            'analysis_timestamp': results_dict.get('analysis_timestamp', 'unknown'),
            'available_models': results_dict.get('available_models', []),
            'model_performance': results_dict.get('model_performance', {}),
            'consensus_features': results_dict.get('consensus_features', {}),
            'statistical_comparison': results_dict.get('statistical_comparison', {}),
            'motif_analysis': results_dict.get('motif_analysis', {}),
            'feature_agreement': results_dict.get('feature_agreement', {}),
            'clinical_relevance': results_dict.get('clinical_relevance', {})
        }
        
        # Validate feature types in consensus features
        if 'by_model_type' in standardized['consensus_features']:
            available_types = list(standardized['consensus_features']['by_model_type'].keys())
            unexpected_types = [t for t in available_types if t not in feature_types + ['tabular_models', 'sequence_models']]
            if unexpected_types:
                warnings.warn(f"Unexpected feature types found: {unexpected_types}")
        
        return standardized
    
    def _validate_metrics(self, results_list: List[Dict], required_metrics: List[str], context: str):
        """Validate that required metrics are present in results."""
        for i, result in enumerate(results_list):
            if not isinstance(result, dict):
                continue
                
            missing_metrics = [metric for metric in required_metrics if metric not in result]
            if missing_metrics:
                warnings.warn(f"Missing metrics in {context} result {i}: {missing_metrics}")
    
    def standardize_filename(self, template_key: str, **kwargs) -> str:
        """
        Generate standardized filename using naming conventions.
        
        Args:
            template_key: Key in naming conventions (e.g., 'model_results')
            **kwargs: Variables for template formatting
            
        Returns:
            Standardized filename
        """
        template = self.naming_conventions.get(template_key, f"{{antibiotic}}_{template_key}")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            warnings.warn(f"Missing variable {e} for template {template_key}")
            return f"{kwargs.get('antibiotic', 'unknown')}_{template_key}"
    
    def validate_output_directory(self, output_dir: str, create_if_missing: bool = True) -> Path:
        """
        Validate and optionally create output directory.
        
        Args:
            output_dir: Output directory path
            create_if_missing: Whether to create directory if it doesn't exist
            
        Returns:
            Path object for the directory
        """
        output_path = Path(output_dir)
        
        if not output_path.exists() and create_if_missing:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_path}")
        elif not output_path.exists():
            warnings.warn(f"Output directory does not exist: {output_path}")
        
        return output_path
    
    def save_standardized_results(self, results_dict: Dict[str, Any], 
                                output_path: str, result_type: str = 'model_results'):
        """
        Save results with standardized format validation.
        
        Args:
            results_dict: Results to save
            output_path: Output file path
            result_type: Type of results ('model_results' or 'interpretability')
        """
        # Extract antibiotic from filename or results
        antibiotic = results_dict.get('antibiotic', Path(output_path).stem.split('_')[0])
        
        # Validate and standardize
        if result_type == 'model_results':
            standardized = self.validate_model_results(results_dict, antibiotic)
        elif result_type == 'interpretability':
            standardized = self.validate_interpretability_results(results_dict, antibiotic)
        else:
            warnings.warn(f"Unknown result type: {result_type}")
            standardized = results_dict
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        self.validate_output_directory(output_dir)
        
        # Save with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(standardized, f, indent=2, default=self._json_serializer)
        
        print(f"Saved standardized {result_type} to: {output_path}")
        return standardized
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)
    
    def validate_feature_matrix(self, df: pd.DataFrame, expected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Validate feature matrix format and column naming.
        
        Args:
            df: Feature matrix DataFrame
            expected_features: Optional list of expected feature names
            
        Returns:
            Validated DataFrame with standardized column names
        """
        # Standardize column names
        df = df.copy()
        
        # Rename common variations
        column_mapping = {
            'sample_id': 'sample_id',
            'Sample_ID': 'sample_id', 
            'SampleID': 'sample_id',
            'ID': 'sample_id',
            'resistance': 'R',
            'Resistance': 'R',
            'resistant': 'R',
            'label': 'R',
            'target': 'R'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Validate required columns
        required_cols = ['sample_id', 'R']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            warnings.warn(f"Missing required columns: {missing_cols}")
        
        # Validate R column (should be 0/1)
        if 'R' in df.columns:
            unique_vals = df['R'].dropna().unique()
            if not all(val in [0, 1] for val in unique_vals):
                warnings.warn(f"R column should contain only 0/1, found: {unique_vals}")
        
        # Check for expected features
        if expected_features:
            feature_cols = [col for col in df.columns if col not in ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']]
            missing_features = [f for f in expected_features if f not in feature_cols]
            if missing_features:
                warnings.warn(f"Missing expected features: {len(missing_features)} out of {len(expected_features)}")
        
        return df
    
    def check_pipeline_consistency(self, results_dir: str, antibiotics: List[str]) -> Dict[str, Any]:
        """
        Check consistency across all pipeline outputs.
        
        Args:
            results_dir: Results directory path
            antibiotics: List of antibiotics to check
            
        Returns:
            Consistency report
        """
        results_path = Path(results_dir)
        consistency_report = {
            'antibiotics_checked': antibiotics,
            'missing_files': [],
            'format_issues': [],
            'metric_consistency': {},
            'overall_status': 'pass'
        }
        
        # Check for missing files
        for antibiotic in antibiotics:
            model_dirs = ['xgboost', 'lightgbm', 'cnn', 'sequence_cnn', 'dnabert']
            
            for model_dir in model_dirs:
                model_file = results_path / 'models' / model_dir / f'{antibiotic}_results.json'
                if not model_file.exists():
                    consistency_report['missing_files'].append(str(model_file))
            
            interp_file = results_path / 'interpretability' / f'{antibiotic}_interpretability.json'
            if not interp_file.exists():
                consistency_report['missing_files'].append(str(interp_file))
        
        # Check format consistency
        for antibiotic in antibiotics:
            try:
                # Load and validate interpretability results
                interp_file = results_path / 'interpretability' / f'{antibiotic}_interpretability.json'
                if interp_file.exists():
                    with open(interp_file, 'r') as f:
                        interp_data = json.load(f)
                    
                    # Validate structure
                    validated = self.validate_interpretability_results(interp_data, antibiotic)
                    
                    # Check for required fields
                    required_fields = ['consensus_features', 'statistical_comparison']
                    for field in required_fields:
                        if field not in validated or not validated[field]:
                            consistency_report['format_issues'].append(f'{antibiotic}: Missing {field}')
            
            except Exception as e:
                consistency_report['format_issues'].append(f'{antibiotic}: {str(e)}')
        
        # Set overall status
        if consistency_report['missing_files'] or consistency_report['format_issues']:
            consistency_report['overall_status'] = 'fail'
        
        return consistency_report


def validate_pipeline_outputs(results_dir: str = "results", 
                            antibiotics: List[str] = ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']):
    """
    Convenience function to validate all pipeline outputs.
    
    Args:
        results_dir: Results directory path
        antibiotics: List of antibiotics to validate
        
    Returns:
        Validation report
    """
    validator = OutputValidator()
    report = validator.check_pipeline_consistency(results_dir, antibiotics)
    
    print("=== PIPELINE OUTPUT VALIDATION REPORT ===")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Antibiotics Checked: {len(report['antibiotics_checked'])}")
    
    if report['missing_files']:
        print(f"\n❌ Missing Files ({len(report['missing_files'])}):")
        for file in report['missing_files'][:10]:  # Show first 10
            print(f"  - {file}")
        if len(report['missing_files']) > 10:
            print(f"  ... and {len(report['missing_files']) - 10} more")
    
    if report['format_issues']:
        print(f"\n⚠️  Format Issues ({len(report['format_issues'])}):")
        for issue in report['format_issues']:
            print(f"  - {issue}")
    
    if not report['missing_files'] and not report['format_issues']:
        print("\n✅ All outputs validated successfully!")
    
    return report


if __name__ == "__main__":
    # Run validation on current results
    report = validate_pipeline_outputs()
    
    # Example usage
    validator = OutputValidator()
    
    # Test filename generation
    model_filename = validator.standardize_filename('model_results', antibiotic='amikacin')
    print(f"Standardized model filename: {model_filename}")
    
    interp_filename = validator.standardize_filename('interpretability', antibiotic='amikacin')
    print(f"Standardized interpretability filename: {interp_filename}")