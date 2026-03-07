#!/usr/bin/env python3
"""
Batch Effect Assessment and Correction using PCA visualization and ComBat.
Checks for batch effects from Year, Isolation_source, Location.
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from scipy import stats
from scipy.linalg import solve

warnings.filterwarnings('ignore')

def assess_batch_effects_pca(data, batch_variables=['Year', 'Location', 'Isolation_source']):
    """
    Use PCA to visualize potential batch effects across different variables.
    """
    # Get feature columns
    feature_cols = [c for c in data.columns if c.startswith(('gene_', 'snp_'))]
    meta_cols = ['sample_id', 'R'] + batch_variables
    
    if len(feature_cols) == 0:
        print("No feature columns found")
        return {}
    
    # Prepare data for PCA
    X = data[feature_cols].values
    
    # Handle missing values and infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if X.shape[0] < 3:
        print("Too few samples for PCA analysis")
        return {}
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    n_components = min(10, X_scaled.shape[0]-1, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Add metadata
    for col in meta_cols:
        if col in data.columns:
            pca_df[col] = data[col].values
    
    # Analyze batch effects for each variable
    batch_effects = {}
    
    for batch_var in batch_variables:
        if batch_var not in data.columns:
            continue
            
        print(f"\n=== Analyzing {batch_var} ===")
        
        # Get unique values
        unique_vals = data[batch_var].dropna().unique()
        
        if len(unique_vals) <= 1:
            print(f"Only one value for {batch_var}, skipping")
            continue
            
        if len(unique_vals) > 10:
            print(f"Too many unique values for {batch_var} ({len(unique_vals)}), skipping visualization")
            continue
        
        print(f"Unique values: {unique_vals}")
        
        # Calculate variance explained by this batch variable
        batch_variance = calculate_batch_variance(X_pca[:, :3], data[batch_var])
        
        batch_effects[batch_var] = {
            'unique_values': unique_vals.tolist(),
            'n_unique': len(unique_vals),
            'pc1_variance_explained': batch_variance['pc1'],
            'pc2_variance_explained': batch_variance['pc2'],
            'pc3_variance_explained': batch_variance['pc3'],
            'total_variance_explained': batch_variance['total']
        }
        
        print(f"Variance explained by {batch_var}:")
        print(f"  PC1: {batch_variance['pc1']:.1%}")
        print(f"  PC2: {batch_variance['pc2']:.1%}")
        print(f"  PC3: {batch_variance['pc3']:.1%}")
        print(f"  Total (PC1-3): {batch_variance['total']:.1%}")
    
    return {
        'pca_results': pca_df,
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'batch_effects': batch_effects,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }

def calculate_batch_variance(pca_coords, batch_labels):
    """
    Calculate variance in PCA space explained by batch labels.
    """
    results = {}
    
    for i, pc in enumerate(['pc1', 'pc2', 'pc3']):
        if i >= pca_coords.shape[1]:
            results[pc] = 0.0
            continue
            
        pc_values = pca_coords[:, i]
        
        # Calculate between-batch variance vs within-batch variance
        batch_groups = {}
        for idx, label in enumerate(batch_labels):
            if pd.notna(label):
                if label not in batch_groups:
                    batch_groups[label] = []
                batch_groups[label].append(pc_values[idx])
        
        if len(batch_groups) <= 1:
            results[pc] = 0.0
            continue
        
        # Calculate F-statistic-like measure
        overall_mean = np.mean(pc_values)
        
        # Between-group variance
        between_var = 0
        total_n = 0
        for group, values in batch_groups.items():
            group_mean = np.mean(values)
            group_n = len(values)
            between_var += group_n * (group_mean - overall_mean) ** 2
            total_n += group_n
        between_var /= (len(batch_groups) - 1)
        
        # Within-group variance
        within_var = 0
        for group, values in batch_groups.items():
            group_var = np.var(values) * len(values)
            within_var += group_var
        within_var /= (total_n - len(batch_groups))
        
        # Variance explained (pseudo R-squared)
        total_var = np.var(pc_values)
        if total_var > 0:
            var_explained = between_var / (between_var + within_var)
        else:
            var_explained = 0.0
            
        results[pc] = var_explained
    
    # Total variance explained across PC1-3
    results['total'] = np.mean([results['pc1'], results['pc2'], results['pc3']])
    
    return results

def create_pca_plots(pca_results, output_dir, batch_variables=['Year', 'Location', 'Isolation_source']):
    """
    Create PCA plots colored by batch variables, ensuring external legends are not clipped.
    """
    pca_df = pca_results['pca_results']
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    renderer = plt.gcf().canvas.get_renderer() # Get renderer once
    
    for batch_var in batch_variables:
        if batch_var not in pca_df.columns:
            continue
            
        # Skip if too many unique values or too few samples
        unique_vals = pca_df[batch_var].dropna().unique()
        if len(unique_vals) <= 1 or len(unique_vals) > 10:
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PC1 vs PC2
        for val in unique_vals:
            mask = pca_df[batch_var] == val
            if mask.sum() > 0:
                axes[0].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                              label=str(val), alpha=0.7, s=50)
        
        axes[0].set_xlabel(f'PC1 ({pca_results["pca_variance_ratio"][0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca_results["pca_variance_ratio"][1]:.1%} variance)')
        axes[0].set_title(f'PCA colored by {batch_var}')
        # Store the legend object
        legend_0 = axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # PC1 vs PC3
        legend_1 = None
        if pca_results["pca_variance_ratio"].shape[0] >= 3:
            for val in unique_vals:
                mask = pca_df[batch_var] == val
                if mask.sum() > 0:
                    axes[1].scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC3'], 
                                  label=str(val), alpha=0.7, s=50)
            
            axes[1].set_xlabel(f'PC1 ({pca_results["pca_variance_ratio"][0]:.1%} variance)')
            axes[1].set_ylabel(f'PC3 ({pca_results["pca_variance_ratio"][2]:.1%} variance)')
            axes[1].set_title(f'PCA colored by {batch_var}')
            # Store the legend object
            legend_1 = axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        # Use tight_layout only for internal spacing
        plt.tight_layout()
        
        # Calculate the bounding box to include the external legends
        from matplotlib.transforms import Bbox
        
        bboxes = [fig.get_tightbbox(renderer)]
        
        if legend_0:
            bboxes.append(legend_0.get_window_extent(renderer))
            
        if legend_1:
            bboxes.append(legend_1.get_window_extent(renderer))
        
        final_bbox = Bbox.union(bboxes)
        
        # Save figure using the combined bounding box
        plt.savefig(
            output_dir / f'pca_{batch_var.lower()}.png', 
            dpi=300, 
            bbox_inches=final_bbox.transformed(fig.dpi_scale_trans.inverted())
        )

        plt.close(fig) # Use close(fig) for safety
    
    # Create a summary plot with all batch variables
    create_summary_plot(pca_results, output_dir, batch_variables)

def create_summary_plot(pca_results, output_dir, batch_variables):
    """Create a summary plot for all batch effects."""
    pca_df = pca_results['pca_results']
    output_dir = Path(output_dir)
    
    # Count valid batch variables
    valid_vars = []
    for batch_var in batch_variables:
        if batch_var in pca_df.columns:
            unique_vals = pca_df[batch_var].dropna().unique()
            if 1 < len(unique_vals) <= 10:
                valid_vars.append(batch_var)
    
    if not valid_vars:
        # Create a basic PCA plot if no valid batch variables
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, s=50)
        ax.set_xlabel(f'PC1 ({pca_results["pca_variance_ratio"][0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca_results["pca_variance_ratio"][1]:.1%} variance)')
        ax.set_title('PCA Plot')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'batch_effects_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create subplot grid
    n_vars = len(valid_vars)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
    if n_vars == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes] if n_cols > 1 else [[axes]]
    
    for i, batch_var in enumerate(valid_vars):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]
        
        unique_vals = pca_df[batch_var].dropna().unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_vals)))
        
        for j, val in enumerate(unique_vals):
            mask = pca_df[batch_var] == val
            if mask.sum() > 0:
                ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                          label=str(val), alpha=0.7, s=50, color=colors[j])
        
        ax.set_xlabel(f'PC1 ({pca_results["pca_variance_ratio"][0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca_results["pca_variance_ratio"][1]:.1%} variance)')
        ax.set_title(f'PCA by {batch_var}')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    if n_vars < n_rows * n_cols:
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row][col])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_effects_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def decide_correction_strategy(batch_effects):
    """
    Decide whether batch correction is needed based on PCA analysis.
    """
    correction_needed = False
    problematic_batches = []
    
    # Thresholds based on genomics literature
    HIGH_VARIANCE_THRESHOLD = 0.15  # 15% variance explained is concerning
    MODERATE_VARIANCE_THRESHOLD = 0.05  # 5% variance explained is worth noting
    
    for batch_var, effects in batch_effects.items():
        total_variance = effects['total_variance_explained']
        
        if total_variance > HIGH_VARIANCE_THRESHOLD:
            correction_needed = True
            problematic_batches.append((batch_var, total_variance, "HIGH"))
        elif total_variance > MODERATE_VARIANCE_THRESHOLD:
            problematic_batches.append((batch_var, total_variance, "MODERATE"))
    
    strategy = {
        'correction_needed': correction_needed,
        'problematic_batches': problematic_batches,
        'recommendation': 'combat' if correction_needed else 'none'
    }
    
    if correction_needed:
        print(f"\n🚨 BATCH EFFECTS DETECTED!")
        for batch_var, variance, severity in problematic_batches:
            print(f"  {batch_var}: {variance:.1%} variance ({severity} impact)")
        print(f"Recommendation: Apply ComBat correction")
    else:
        print(f"\n✅ No significant batch effects detected")
        print(f"Recommendation: Proceed without batch correction")
    
    return strategy

def combat_batch_correction(data, batch_labels, parametric=True):
    """
    Python implementation of ComBat batch correction algorithm.
    
    Parameters:
    - data: pandas DataFrame with samples as rows, features as columns
    - batch_labels: array-like with batch assignments for each sample
    - parametric: bool, whether to use parametric adjustments
    
    Returns:
    - corrected_data: DataFrame with batch-corrected values
    """
    print(f"Applying ComBat batch correction...")
    print(f"  - Samples: {data.shape[0]}")
    print(f"  - Features: {data.shape[1]}")
    print(f"  - Batches: {len(np.unique(batch_labels))}")
    
    # Convert to numpy arrays for computation
    X = data.select_dtypes(include=[np.number]).values.T  # Features x Samples
    metadata_cols = data.select_dtypes(exclude=[np.number]).columns
    sample_ids = data.index
    feature_names = data.select_dtypes(include=[np.number]).columns
    
    batch_labels = np.array(batch_labels)
    batches = np.unique(batch_labels)
    n_batch = len(batches)
    n_features, n_samples = X.shape
    
    if n_batch <= 1:
        print("Only one batch found, skipping correction")
        return data
    
    # Create design matrix for batches
    design = np.zeros((n_samples, n_batch))
    for i, batch in enumerate(batches):
        design[batch_labels == batch, i] = 1
    
    # Step 1: Standardize data across features
    print("  Step 1: Standardizing data...")
    grand_mean = np.mean(X, axis=1, keepdims=True)
    var_pooled = np.var(X, axis=1, ddof=1, keepdims=True)
    var_pooled[var_pooled == 0] = np.finfo(float).eps  # Avoid division by zero
    
    # Step 2: Fit linear model to estimate batch effects
    print("  Step 2: Estimating batch effects...")
    batch_means = np.zeros((n_features, n_batch))
    batch_vars = np.zeros((n_features, n_batch))
    
    for i, batch in enumerate(batches):
        batch_mask = batch_labels == batch
        if np.sum(batch_mask) > 1:
            batch_data = X[:, batch_mask]
            batch_means[:, i] = np.mean(batch_data, axis=1)
            batch_vars[:, i] = np.var(batch_data, axis=1, ddof=1)
        else:
            batch_means[:, i] = X[:, batch_mask].flatten()
            batch_vars[:, i] = var_pooled.flatten()
    
    # Step 3: Empirical Bayes adjustment
    print("  Step 3: Empirical Bayes adjustment...")
    
    # Prior parameters for location (additive batch effects)
    alpha_prior = np.zeros((n_features, n_batch))
    beta_prior = np.zeros((n_features, n_batch))
    
    # Prior parameters for scale (multiplicative batch effects) 
    gamma_prior = np.ones((n_features, n_batch))
    delta_prior = np.ones((n_features, n_batch))
    
    if parametric and n_batch > 2:
        # Use method of moments to estimate hyperparameters
        for g in range(n_features):
            # For additive effects (gamma parameters)
            valid_vars = batch_vars[g, :][batch_vars[g, :] > 0]
            if len(valid_vars) > 1:
                mean_var = np.mean(valid_vars)
                var_var = np.var(valid_vars)
                if var_var > 0:
                    # Inverse gamma distribution parameters
                    alpha_g = (2 * var_var + mean_var**2) / var_var
                    beta_g = (mean_var * (var_var + mean_var**2)) / var_var
                    gamma_prior[g, :] = alpha_g
                    delta_prior[g, :] = beta_g
    
    # Step 4: Compute adjusted batch parameters
    print("  Step 4: Computing adjusted parameters...")
    gamma_star = np.copy(batch_vars)
    delta_star = np.copy(batch_means)
    
    for g in range(n_features):
        for i, batch in enumerate(batches):
            batch_mask = batch_labels == batch
            n_batch_samples = np.sum(batch_mask)
            
            if n_batch_samples > 1:
                # Empirical Bayes shrinkage for variance
                if parametric:
                    # Shrink towards prior
                    gamma_star[g, i] = ((n_batch_samples - 1) * batch_vars[g, i] + 
                                      2 * delta_prior[g, i]) / (n_batch_samples + 1 + 2 * gamma_prior[g, i])
                else:
                    gamma_star[g, i] = batch_vars[g, i]
                
                # Shrink batch means towards grand mean
                delta_star[g, i] = ((n_batch_samples * batch_means[g, i] + 
                                   2 * beta_prior[g, i] * grand_mean[g, 0]) / 
                                  (n_batch_samples + 2 * alpha_prior[g, i]))
    
    # Step 5: Apply correction
    print("  Step 5: Applying correction...")
    X_corrected = np.copy(X)
    
    for i, batch in enumerate(batches):
        batch_mask = batch_labels == batch
        if np.sum(batch_mask) > 0:
            batch_data = X[:, batch_mask]
            
            # Apply additive and multiplicative corrections
            for g in range(n_features):
                if gamma_star[g, i] > 0:
                    # Multiplicative correction (variance)
                    variance_correction = np.sqrt(var_pooled[g, 0] / gamma_star[g, i])
                    # Additive correction (mean)
                    mean_correction = delta_star[g, i] - grand_mean[g, 0]
                    
                    # Apply correction
                    X_corrected[g, batch_mask] = (
                        (batch_data[g, :] - delta_star[g, i]) * variance_correction + 
                        grand_mean[g, 0]
                    )
    
    # Convert back to DataFrame
    corrected_df = pd.DataFrame(
        X_corrected.T, 
        index=sample_ids, 
        columns=feature_names
    )
    
    # Add back non-numeric columns
    for col in metadata_cols:
        corrected_df[col] = data[col]
    
    # Reorder columns to match original
    corrected_df = corrected_df[data.columns]
    
    print(f"ComBat correction completed!")
    
    return corrected_df

def apply_combat_correction(data, correction_strategy):
    """
    Apply ComBat correction if needed.
    """
    if not correction_strategy['correction_needed']:
        print("No batch correction needed")
        return data
    
    # Find the primary batch variable with highest variance
    problematic_batches = correction_strategy['problematic_batches']
    if not problematic_batches:
        print("No problematic batches identified")
        return data
    
    # Use the batch variable with highest variance
    primary_batch = problematic_batches[0][0]  # First problematic batch
    
    if primary_batch not in data.columns:
        print(f"Primary batch variable '{primary_batch}' not found in data")
        return data
    
    # Get batch labels
    batch_labels = data[primary_batch].fillna('Unknown')
    
    # Check if we have enough variation
    unique_batches = batch_labels.unique()
    if len(unique_batches) <= 1:
        print("Insufficient batch variation for correction")
        return data
    
    print(f"Applying ComBat correction for batch variable: {primary_batch}")
    print(f"Batch levels: {unique_batches}")
    
    try:
        corrected_data = combat_batch_correction(data, batch_labels, parametric=True)
        print("✅ ComBat correction applied successfully")
        return corrected_data
    
    except Exception as e:
        print(f"⚠️ ComBat correction failed: {e}")
        print("Returning original data")
        return data

def main():
    # Load data
    train_df = pd.read_csv(snakemake.input.train)
    test_df = pd.read_csv(snakemake.input.test)
    
    print(f"=== BATCH EFFECT ASSESSMENT ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create output directory for plots
    plots_dir = Path(snakemake.output.plots).parent
    plots_dir.mkdir(exist_ok=True)
    
    # Assess batch effects on training data
    batch_variables = ['Year', 'Location', 'Isolation_source']
    pca_results = assess_batch_effects_pca(train_df, batch_variables)
    
    if not pca_results:
        print("Could not perform PCA analysis")
        # Save original data
        train_df.to_csv(snakemake.output.train, index=False)
        test_df.to_csv(snakemake.output.test, index=False)
        return
    
    # Create PCA plots
    print(f"\nCreating PCA visualization plots...")
    create_pca_plots(pca_results, plots_dir, batch_variables)
    
    # Decide correction strategy
    correction_strategy = decide_correction_strategy(pca_results['batch_effects'])
    
    # Apply correction if needed
    if correction_strategy['correction_needed']:
        train_corrected = apply_combat_correction(train_df, correction_strategy)
        test_corrected = apply_combat_correction(test_df, correction_strategy)
    else:
        train_corrected = train_df
        test_corrected = test_df
    
    # Save results
    train_corrected.to_csv(snakemake.output.train, index=False)
    test_corrected.to_csv(snakemake.output.test, index=False)
    
    # Copy the summary PNG to the expected output location
    expected_plots = Path(snakemake.output.plots)
    summary_png = plots_dir / 'batch_effects_summary.png'
    
    if summary_png.exists():
        # Copy the summary plot to the expected output location
        import shutil
        shutil.copy2(summary_png, expected_plots)
        print(f"Summary plot copied to: {expected_plots}")
    else:
        # Create a simple placeholder plot if summary wasn't generated
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No batch effect plots generated\nInsufficient data variation', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('Batch Effects Analysis')
        plt.savefig(expected_plots, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Placeholder plot created: {expected_plots}")
    
    # Save comprehensive report
    report = {
        'n_train_samples': len(train_df),
        'n_test_samples': len(test_df),
        'n_features': pca_results['n_features'],
        'pca_variance_explained': pca_results['pca_variance_ratio'].tolist(),
        'batch_effects': pca_results['batch_effects'],
        'correction_strategy': correction_strategy,
        'files_created': {
            'plots_directory': str(plots_dir),
            'summary_plot': str(summary_png),
            'main_output_plot': str(expected_plots),
            'individual_pca_plots': [f'pca_{var.lower()}.png' for var in batch_variables 
                                   if var in pca_results['batch_effects']]
        }
    }
    
    with open(snakemake.output.report, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n=== SUMMARY ===")
    print(f"Batch correction needed: {'Yes' if correction_strategy['correction_needed'] else 'No'}")
    print(f"PCA plots saved to: {plots_dir}")
    print(f"Report saved to: {snakemake.output.report}")

if __name__ == "__main__":
    main()