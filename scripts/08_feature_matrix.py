#!/usr/bin/env python3
"""
Create combined feature matrix (AMR genes + SNPs) with metadata.
Generate separate datasets for each antibiotic.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load inputs
    print("Loading AMR data...")
    amr_df = pd.read_csv(snakemake.input.amr)

    print("Loading SNP data (large file, this may take a while)...")
    # Use low_memory=False and specify dtype to handle large file
    snp_df = pd.read_csv(snakemake.input.snp, low_memory=False, dtype={0: str})

    print("Loading metadata...")
    meta_train = pd.read_csv(snakemake.input.meta_train)
    meta_test = pd.read_csv(snakemake.input.meta_test)

    print(f"Loaded: AMR={amr_df.shape}, SNP={snp_df.shape}")
    
    antibiotics = snakemake.params.antibiotics
    outdir = Path(snakemake.params.outdir)
    
    # Merge AMR and SNP features
    features_df = amr_df.merge(snp_df, on='sample_id', how='outer')
    features_df = features_df.fillna(0)
    
    # Get feature columns
    feature_cols = [c for c in features_df.columns if c != 'sample_id']
    
    # Process each antibiotic
    for antibiotic in antibiotics:
        # Training set
        train_meta = meta_train[['Run', antibiotic, 'Year', 'Location']].copy()
        train_meta = train_meta.rename(columns={'Run': 'sample_id', antibiotic: 'R'})
        train_meta = train_meta.dropna(subset=['R'])
        train_meta['R'] = train_meta['R'].astype(int)
        
        train_df = train_meta.merge(features_df, on='sample_id', how='inner')
        
        # Test set
        test_meta = meta_test[['Run', antibiotic, 'Year', 'Location']].copy()
        test_meta = test_meta.rename(columns={'Run': 'sample_id', antibiotic: 'R'})
        test_meta = test_meta.dropna(subset=['R'])
        test_meta['R'] = test_meta['R'].astype(int)
        
        test_df = test_meta.merge(features_df, on='sample_id', how='inner')
        
        # Save datasets
        train_df.to_csv(outdir / f'{antibiotic}_train.csv', index=False)
        test_df.to_csv(outdir / f'{antibiotic}_test.csv', index=False)
        
        print(f"{antibiotic}: Train={len(train_df)} (R={train_df['R'].sum()}), "
              f"Test={len(test_df)} (R={test_df['R'].sum()})")

if __name__ == "__main__":
    main()