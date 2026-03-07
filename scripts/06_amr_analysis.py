#!/usr/bin/env python3
"""
Combine AMRFinderPlus outputs into a single presence/absence matrix.
"""

import pandas as pd
from pathlib import Path
import os

def main():
    input_files = snakemake.input
    output_file = snakemake.output[0]
    
    all_genes = set()
    sample_genes = {}
    
    # Parse all AMRFinder outputs
    for f in input_files:
        sample_id = Path(f).stem.replace('_amrfinder', '')
        df = pd.read_csv(f, sep='\t')
        
        # Extract gene symbols
        genes = set(df['Element symbol'].dropna().unique())
        all_genes.update(genes)
        sample_genes[sample_id] = genes
    
    # Create presence/absence matrix
    all_genes = sorted(all_genes)
    matrix_data = []
    
    for sample_id, genes in sample_genes.items():
        row = {'sample_id': sample_id}
        for gene in all_genes:
            row[f'gene_{gene}'] = 1 if gene in genes else 0
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.to_csv(output_file, index=False)
    
    print(f"Combined {len(sample_genes)} samples with {len(all_genes)} unique genes")

if __name__ == "__main__":
    main()