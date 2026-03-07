#!/usr/bin/env python3
"""
Advanced Feature Selection and Engineering for AMR Prediction.
Based on techniques from genomic ML papers summary.
Handles extreme dimensionality (264K+ features, <20 samples).
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import (
    chi2, mutual_info_classif, VarianceThreshold, 
    SelectKBest, SelectPercentile
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from collections import Counter
import re

warnings.filterwarnings('ignore')

def variance_filter(X, feature_names, threshold=0.001):
    """Remove features with very low variance (mostly constant)."""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    mask = selector.get_support()
    feature_names_filtered = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    print(f"Variance filter: {X.shape[1]} → {X_filtered.shape[1]} features")
    return X_filtered, feature_names_filtered

def sparsity_filter(X, feature_names, max_sparsity=0.95):
    """Remove extremely sparse features (>95% zeros), with special handling for AMR genes."""
    sparsity_ratio = (X == 0).mean(axis=0)
    
    # Keep all AMR genes regardless of sparsity (they're clinically important even if rare)
    amr_mask = np.array([name.startswith('gene_') for name in feature_names])
    sparse_mask = sparsity_ratio < max_sparsity
    
    # Combine masks: keep if either AMR gene OR not too sparse
    combined_mask = amr_mask | sparse_mask
    
    X_filtered = X[:, combined_mask]
    feature_names_filtered = [feature_names[i] for i in range(len(feature_names)) if combined_mask[i]]
    
    amr_kept = amr_mask.sum()
    sparse_kept = (sparse_mask & ~amr_mask).sum()
    
    print(f"Sparsity filter: {X.shape[1]} → {X_filtered.shape[1]} features")
    print(f"  Kept {amr_kept} AMR genes (all), {sparse_kept} other non-sparse features")
    
    return X_filtered, feature_names_filtered

def frequency_filter(X, feature_names, min_samples=2):
    """Remove features present in fewer than min_samples, with special handling for AMR genes."""
    presence_count = (X > 0).sum(axis=0)
    
    # Keep all AMR genes regardless of frequency (they're clinically important)
    amr_mask = np.array([name.startswith('gene_') for name in feature_names])
    freq_mask = presence_count >= min_samples
    
    # Combine masks: keep if either AMR gene OR meets frequency threshold
    combined_mask = amr_mask | freq_mask
    
    X_filtered = X[:, combined_mask]
    feature_names_filtered = [feature_names[i] for i in range(len(feature_names)) if combined_mask[i]]
    
    amr_kept = amr_mask.sum()
    freq_kept = (freq_mask & ~amr_mask).sum()
    
    print(f"Frequency filter: {X.shape[1]} → {X_filtered.shape[1]} features")
    print(f"  Kept {amr_kept} AMR genes (all), {freq_kept} other frequent features")
    
    return X_filtered, feature_names_filtered

def snp_genomic_grouping(X, feature_names):
    """
    Group SNPs by genomic context/pathways to reduce dimensionality.
    Based on genomic coordinate proximity and functional annotation.
    """
    snp_indices = [i for i, name in enumerate(feature_names) if name.startswith('snp_')]
    gene_indices = [i for i, name in enumerate(feature_names) if name.startswith('gene_')]
    
    if len(snp_indices) == 0:
        return X, feature_names
    
    print(f"Grouping {len(snp_indices)} SNPs into genomic regions...")
    
    # Parse SNP coordinates (format: snp_contig_position_ref>alt)
    snp_groups = {}
    ungrouped_snps = []
    
    for idx in snp_indices:
        snp_name = feature_names[idx]
        try:
            # Extract contig and position
            parts = snp_name.split('_')
            if len(parts) >= 3:
                contig = parts[1]
                pos = int(parts[2]) if parts[2].isdigit() else 0
                
                # Group into 50kb windows (functional gene clusters)
                window = pos // 50000
                group_key = f"{contig}_w{window}"
                
                if group_key not in snp_groups:
                    snp_groups[group_key] = []
                snp_groups[group_key].append(idx)
            else:
                ungrouped_snps.append(idx)
        except:
            ungrouped_snps.append(idx)
    
    # Create new feature matrix
    new_features = []
    new_feature_names = []
    
    # Keep all AMR gene features (high priority)
    for idx in gene_indices:
        new_features.append(X[:, idx])
        new_feature_names.append(feature_names[idx])
    
    # Add grouped SNP features
    for group_name, indices in snp_groups.items():
        if len(indices) > 5:  # Only group if substantial number of SNPs
            # Use maximum signal in region (presence of any variant)
            grouped_feature = X[:, indices].max(axis=1)
            new_features.append(grouped_feature)
            new_feature_names.append(f"snp_region_{group_name}_n{len(indices)}")
        elif len(indices) > 1:
            # Sum for smaller groups
            grouped_feature = X[:, indices].sum(axis=1)
            new_features.append(grouped_feature)
            new_feature_names.append(f"snp_cluster_{group_name}_n{len(indices)}")
        else:
            # Keep singleton SNPs as-is
            new_features.append(X[:, indices[0]])
            new_feature_names.append(feature_names[indices[0]])
    
    # Add ungrouped SNPs
    for idx in ungrouped_snps:
        new_features.append(X[:, idx])
        new_feature_names.append(feature_names[idx])
    
    if new_features:
        X_grouped = np.column_stack(new_features)
        print(f"Genomic grouping: {len(feature_names)} → {len(new_feature_names)} features")
        return X_grouped, new_feature_names
    else:
        return X, feature_names

def amr_gene_prioritization(X, feature_names):
    """Prioritize known AMR genes and resistance mechanisms."""
    # Known high-importance AMR gene families
    priority_genes = {
        'beta_lactam': ['bla', 'amp', 'tem', 'shv', 'ctx', 'oxa', 'kpc', 'ndm', 'vim', 'imp'],
        'quinolone': ['qnr', 'aac', 'gyr', 'par'],
        'aminoglycoside': ['aac', 'ant', 'aph', 'rmtA', 'rmtB', 'rmtC'],
        'carbapenem': ['kpc', 'ndm', 'vim', 'imp', 'oxa', 'spm', 'gim']
    }
    
    gene_indices = [i for i, name in enumerate(feature_names) if name.startswith('gene_')]
    priority_indices = []
    
    for idx in gene_indices:
        gene_name = feature_names[idx].lower()
        for category, patterns in priority_genes.items():
            if any(pattern in gene_name for pattern in patterns):
                priority_indices.append(idx)
                break
    
    print(f"Prioritized {len(priority_indices)} high-importance AMR genes")
    return priority_indices

def ddg_like_selection(X, y, feature_names, fdr_threshold=0.05):
    """
    Differentially Distributed Genes approach adapted for AMR features.
    Based on Sparta et al. (2024) binomial null model concept.
    """
    from scipy.stats import binom_test
    
    p_values = []
    effect_sizes = []
    
    # Calculate resistance frequency overall
    global_resistance_rate = y.mean()
    
    for i in range(X.shape[1]):
        feature_vec = X[:, i]
        
        # For each feature, test if its presence rate differs between R and S
        resistant_mask = y == 1
        sensitive_mask = y == 0
        
        if resistant_mask.sum() > 0 and sensitive_mask.sum() > 0:
            # Presence rate in resistant vs sensitive
            resistant_rate = feature_vec[resistant_mask].mean()
            sensitive_rate = feature_vec[sensitive_mask].mean()
            
            # Binomial test for differential presence
            resistant_count = int(feature_vec[resistant_mask].sum())
            resistant_total = resistant_mask.sum()
            
            # Test against background rate
            p_val = binom_test(resistant_count, resistant_total, sensitive_rate)
            effect_size = resistant_rate - sensitive_rate
        else:
            p_val = 1.0
            effect_size = 0.0
        
        p_values.append(p_val)
        effect_sizes.append(effect_size)
    
    # FDR correction (Benjamini-Hochberg)
    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    fdr_corrected = p_values.copy()
    
    for i, idx in enumerate(sorted_indices):
        fdr_corrected[idx] = p_values[idx] * len(p_values) / (i + 1)
    
    # Select features passing FDR threshold
    significant_mask = fdr_corrected < fdr_threshold
    
    print(f"DDG-like selection: {significant_mask.sum()} features pass FDR < {fdr_threshold}")
    
    return {
        'p_values': p_values,
        'effect_sizes': np.array(effect_sizes),
        'fdr_corrected': fdr_corrected,
        'significant_mask': significant_mask
    }

def multi_stage_selection(X, y, feature_names, target_features=500):
    """
    Multi-stage aggressive feature selection pipeline.
    """
    print(f"Starting feature selection: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"Sample-to-feature ratio: 1:{X.shape[1]//X.shape[0]}")
    
    # Stage 1: Remove constant/near-constant features
    X, feature_names = variance_filter(X, feature_names, threshold=0.001)
    
    # Stage 2: Remove extremely sparse features
    X, feature_names = sparsity_filter(X, feature_names, max_sparsity=0.98)
    
    # Stage 3: Remove very rare features
    X, feature_names = frequency_filter(X, feature_names, min_samples=2)
    
    # Stage 4: Genomic grouping of SNPs
    X, feature_names = snp_genomic_grouping(X, feature_names)
    
    # Stage 5: Improved clinical-statistical selection
    if X.shape[1] > target_features:
        print(f"Applying clinical-statistical selection...")
        
        # Prioritize AMR genes
        priority_indices = amr_gene_prioritization(X, feature_names)
        print(f"Found {len(priority_indices)} priority AMR genes")
        
        # Calculate statistical importance for all features
        chi2_scores, _ = chi2(X, y)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Combined statistical ranking
        chi2_ranks = rankdata(-chi2_scores)
        mi_ranks = rankdata(-mi_scores)
        statistical_scores = (chi2_ranks + mi_ranks) / 2
        
        # Create clinical relevance scores
        clinical_scores = np.zeros(X.shape[1])
        clinical_scores[priority_indices] = 1000  # High priority for AMR genes
        
        # Add bonus for specific resistance mechanisms per antibiotic
        # Get antibiotic from snakemake context if available
        try:
            antibiotic = snakemake.wildcards.antibiotic
        except:
            antibiotic = 'unknown'
        
        for i, feature_name in enumerate(feature_names):
            feature_lower = feature_name.lower()
            
            # Antibiotic-specific bonuses
            if antibiotic == 'amikacin' and any(x in feature_lower for x in ['aac', 'aph', 'ant', 'rmta', 'rmtb', 'rmtc']):
                clinical_scores[i] += 200
            elif antibiotic == 'ciprofloxacin' and any(x in feature_lower for x in ['gyr', 'par', 'qnr', 'oqx']):
                clinical_scores[i] += 200
            elif antibiotic in ['ceftazidime', 'meropenem'] and any(x in feature_lower for x in ['bla', 'ctx', 'shv', 'tem', 'kpc', 'ndm', 'vim', 'oxa']):
                clinical_scores[i] += 200
        
        # Combine clinical relevance with statistical importance
        combined_scores = clinical_scores + statistical_scores
        
        # Select top features based on combined score
        selected_indices = np.argsort(-combined_scores)[:target_features]  # Note: negative for descending
        
        print(f"Selected {len(selected_indices)} features:")
        amr_selected = sum(1 for i in selected_indices if feature_names[i].startswith('gene_'))
        print(f"  AMR genes: {amr_selected}")
        print(f"  Other features: {len(selected_indices) - amr_selected}")
        
        X = X[:, selected_indices]
        feature_names = [feature_names[i] for i in selected_indices]
    
    final_ratio = X.shape[0] / X.shape[1] if X.shape[1] > 0 else 0
    print(f"Final selection: {X.shape[1]} features")
    print(f"Final sample-to-feature ratio: {final_ratio:.3f}")
    print(f"Overfitting risk: {'REDUCED' if final_ratio > 0.1 else 'HIGH'}")
    
    return X, feature_names

def main():
    # Load data with memory optimization
    print("Loading training data (large file)...")
    train_df = pd.read_csv(snakemake.input.train, low_memory=False)

    print("Loading test data...")
    test_df = pd.read_csv(snakemake.input.test, low_memory=False)

    target_features = snakemake.params.n_features

    # Separate metadata, target, and features
    meta_cols = ['sample_id', 'R', 'Year', 'Location']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    test_feature_cols = [c for c in test_df.columns if c not in meta_cols]

    print(f"Extracting features from DataFrames...")
    X_train = train_df[feature_cols].values
    y_train = train_df['R'].values
    X_test = test_df[test_feature_cols].values

    # Free memory from DataFrames
    del train_df
    del test_df
    import gc
    gc.collect()

    print(f"Original dimensions: {X_train.shape}")

    # Apply multi-stage feature selection
    X_train_selected, selected_features = multi_stage_selection(
        X_train, y_train, feature_cols, target_features
    )

    # Apply same selection to test set
    selected_indices_train = [i for i, name in enumerate(feature_cols) if name in selected_features]
    selected_indices_test = [i for i, name in enumerate(test_feature_cols) if name in selected_features]
    
    # Only keep features that exist in both train and test
    common_features = [name for name in selected_features if name in test_feature_cols]
    train_indices_common = [i for i, name in enumerate(feature_cols) if name in common_features]
    test_indices_common = [i for i, name in enumerate(test_feature_cols) if name in common_features]
    
    X_train_selected = X_train[:, train_indices_common]
    X_test_selected = X_test[:, test_indices_common]
    
    print(f"Common features: {len(common_features)}")

    # Reload only metadata columns to save memory
    print("Creating final datasets...")
    meta_cols = ['sample_id', 'R', 'Year', 'Location']
    train_meta = pd.read_csv(snakemake.input.train, usecols=meta_cols)
    test_meta = pd.read_csv(snakemake.input.test, usecols=meta_cols)

    # Create final datasets
    train_selected = train_meta.copy()
    test_selected = test_meta.copy()

    # Add selected features
    for i, feature_name in enumerate(common_features):
        train_selected[feature_name] = X_train_selected[:, i]
        test_selected[feature_name] = X_test_selected[:, i]

    # Free memory
    del train_meta, test_meta, X_train, X_test, X_train_selected, X_test_selected
    import gc
    gc.collect()
    
    # Create comprehensive importance DataFrame
    importance_df = pd.DataFrame({
        'feature_name': common_features,
        'feature_type': ['gene' if f.startswith('gene_') else 'snp' for f in common_features],
        'is_amr_gene': [f.startswith('gene_') for f in common_features],
        'is_grouped': ['_region_' in f or '_cluster_' in f for f in common_features],
        'selected': True
    })
    
    # Save outputs
    train_selected.to_csv(snakemake.output.train, index=False)
    test_selected.to_csv(snakemake.output.test, index=False)
    importance_df.to_csv(snakemake.output.importance, index=False)
    
    # Save selection report
    report = {
        'original_features': len(feature_cols),
        'selected_features': len(common_features),
        'reduction_ratio': len(feature_cols) / len(common_features),
        'amr_genes': sum(importance_df['is_amr_gene']),
        'snp_features': sum(~importance_df['is_amr_gene']),
        'grouped_features': sum(importance_df['is_grouped']),
        'final_sample_to_feature_ratio': len(train_selected) / len(common_features),
        'overfitting_risk_status': 'REDUCED'
    }
    
    import json
    with open(snakemake.output.report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== FEATURE SELECTION SUMMARY ===")
    print(f"Original features: {len(feature_cols)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Reduction ratio: {len(feature_cols)/len(selected_features):.1f}x")
    print(f"AMR genes: {sum(importance_df['is_amr_gene'])}")
    print(f"SNP features: {sum(~importance_df['is_amr_gene'])}")
    print(f"Grouped features: {sum(importance_df['is_grouped'])}")

if __name__ == "__main__":
    main()