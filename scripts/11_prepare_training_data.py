#!/usr/bin/env python3
"""
Prepare training data with class weighting for AMR prediction models.
Creates train/test splits with class weight calculation for addressing class imbalance.
Uses temporal splitting and consistent class weighting strategy across all models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pathlib import Path
from utils.class_balancing import get_imbalance_strategy, apply_sampling_strategy

def main():
    # Get antibiotic name from input path
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in snakemake.input.train:
            antibiotic = ab
            break
    
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from input path")
    
    print(f"Creating balanced datasets for {antibiotic}")
    
    # Load batch-corrected data (already includes labels)
    train_df = pd.read_csv(snakemake.input.train)
    test_df = pd.read_csv(snakemake.input.test)
    
    # Load metadata for additional information (if needed)
    train_metadata = pd.read_csv(snakemake.input.train_metadata)
    test_metadata = pd.read_csv(snakemake.input.test_metadata)
    
    print(f"Original training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Check if we have valid labels
    if 'R' not in train_df.columns:
        raise ValueError("No 'R' column found in training data")
    
    # Remove rows with missing labels
    train_clean = train_df.dropna(subset=['R'])
    test_clean = test_df.dropna(subset=['R'])
    
    print(f"Training samples after removing missing labels: {len(train_clean)}")
    print(f"Test samples after removing missing labels: {len(test_clean)}")
    
    # Identify feature and metadata columns
    meta_cols = ['sample_id', 'R', 'Year', 'Location', 'Isolation_source']
    feature_cols = [c for c in train_clean.columns if c not in meta_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Original class distribution: {train_clean['R'].value_counts().to_dict()}")
    
    # Determine optimal imbalance handling strategy
    y_labels = train_clean['R'].values
    strategy = get_imbalance_strategy(y_labels)
    
    if strategy["method"] == "none":
        print(f"WARNING: Single class dataset")
        balanced_train = train_clean
        balance_applied = False
    elif strategy["details"] in ["mild_imbalance", "moderate_imbalance"]:
        # Use class weights only for mild-moderate imbalance
        print(f"Strategy: {strategy['method']} - using class weights only")
        balanced_train = train_clean
        balance_applied = False
    else:
        # Apply sampling for high/extreme imbalance
        print(f"Strategy: {strategy['method']} - applying {strategy['sampling']}")
        
        # Prepare feature matrix for sampling
        X_features = train_clean[feature_cols].values
        X_resampled, y_resampled = apply_sampling_strategy(X_features, y_labels, strategy)
        
        if len(X_resampled) != len(X_features):
            # Sampling was applied, reconstruct DataFrame
            balanced_train = pd.DataFrame(X_resampled, columns=feature_cols)
            balanced_train['sample_id'] = [f"sample_{i}" for i in range(len(balanced_train))]
            balanced_train['R'] = y_resampled
            
            # Add metadata columns if they exist
            for col in ['Year', 'Location', 'Isolation_source']:
                if col in train_clean.columns:
                    # Use majority class value for simplicity
                    most_common = train_clean[col].mode().iloc[0] if len(train_clean[col].mode()) > 0 else 'unknown'
                    balanced_train[col] = most_common
            
            balance_applied = True
            print(f"Sampling applied: {len(train_clean)} -> {len(balanced_train)} samples")
        else:
            # Sampling failed or was skipped
            balanced_train = train_clean
            balance_applied = False
    
    # Ensure output directories exist
    Path(snakemake.output.train).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.test).parent.mkdir(parents=True, exist_ok=True)
    
    # Save balanced datasets
    balanced_train.to_csv(snakemake.output.train, index=False)
    test_clean.to_csv(snakemake.output.test, index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Original vs balanced distribution
    plt.subplot(1, 2, 1)
    original_counts = train_clean['R'].value_counts().sort_index()
    balanced_counts = balanced_train['R'].value_counts().sort_index()
    
    x = range(len(original_counts))
    width = 0.35
    labels = [f'Class {int(idx)}' for idx in original_counts.index]
    
    plt.bar([i - width/2 for i in x], original_counts.values, width, 
            label='Original', alpha=0.7, color='skyblue')
    
    plt.bar([i + width/2 for i in x], balanced_counts.values, width,
            label='Original (Class Weighted)', alpha=0.7, color='lightgreen')
    
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.title(f'Class Distribution - {antibiotic}')
    plt.xticks(x, labels)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    
    # Sample size comparison
    plt.subplot(1, 2, 2)
    sizes = ['Original', 'Balanced']
    counts = [len(train_clean), len(balanced_train)]
    colors = ['skyblue', 'lightgreen']
    
    bars = plt.bar(sizes, counts, color=colors, alpha=0.7)
    plt.ylabel('Total Samples')
    plt.title('Dataset Size')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(snakemake.params.temp_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate class weights for models to use
    class_distribution = train_clean['R'].value_counts().to_dict()
    total_samples = len(train_clean)
    
    # Calculate class weights (inverse frequency weighting)
    class_weights = {}
    for class_label, count in class_distribution.items():
        class_weights[int(class_label)] = total_samples / (len(class_distribution) * count)
    
    # Calculate scale_pos_weight for tree models (ratio of negative to positive)
    if 0 in class_distribution and 1 in class_distribution:
        scale_pos_weight = class_distribution[0] / class_distribution[1]
    else:
        scale_pos_weight = 1.0
    
    # Create summary report
    summary = {
        'antibiotic': antibiotic,
        'original_samples': len(train_clean),
        'balanced_samples': len(balanced_train),
        'test_samples': len(test_clean),
        'original_distribution': class_distribution,
        'balanced_distribution': balanced_train['R'].value_counts().to_dict(),
        'balance_method': strategy.get('method', 'class_weighting'),
        'imbalance_strategy': strategy,
        'balance_applied': balance_applied,
        'features_count': len(feature_cols),
        'class_weights': class_weights,
        'scale_pos_weight': scale_pos_weight
    }
    
    import json
    with open(snakemake.output.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Datasets prepared successfully with class weighting")
    print(f"Balance method: class_weighting")
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    print(f"Class weights: {class_weights}")

if __name__ == "__main__":
    main()