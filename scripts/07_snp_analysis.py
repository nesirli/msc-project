#!/usr/bin/env python3
"""
Combine filtered VCF files into SNP presence/absence matrix.
Memory-efficient version using sparse matrix.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix
import gc

def parse_vcf(vcf_file):
    """Parse VCF file and extract SNP positions."""
    snps = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                chrom, pos, _, ref, alt = parts[:5]
                snp_id = f"{chrom}_{pos}_{ref}>{alt}"
                snps.append(snp_id)
    return set(snps)

def main():
    input_files = snakemake.input
    output_file = snakemake.output[0]

    print(f"Processing {len(input_files)} VCF files...")

    # First pass: collect all unique SNPs and count
    all_snps_dict = {}
    sample_snps = {}

    for i, f in enumerate(input_files):
        if i % 100 == 0:
            print(f"Pass 1: Processed {i}/{len(input_files)} files...")

        sample_id = Path(f).stem.replace('.filtered', '')
        snps = parse_vcf(f)
        sample_snps[sample_id] = snps

        # Count SNP occurrences
        for snp in snps:
            all_snps_dict[snp] = all_snps_dict.get(snp, 0) + 1

    # Filter SNPs: only keep those present in at least 2 samples (to reduce noise)
    min_samples = 2
    filtered_snps = {snp for snp, count in all_snps_dict.items() if count >= min_samples}
    print(f"Total unique SNPs: {len(all_snps_dict)}")
    print(f"Filtered SNPs (>={min_samples} samples): {len(filtered_snps)}")

    # Create SNP index
    snp_to_idx = {snp: idx for idx, snp in enumerate(sorted(filtered_snps))}

    # Second pass: create sparse matrix
    n_samples = len(sample_snps)
    n_snps = len(filtered_snps)
    print(f"Creating matrix: {n_samples} samples × {n_snps} SNPs")

    # Use sparse matrix for memory efficiency
    matrix = lil_matrix((n_samples, n_snps), dtype=np.int8)
    sample_ids = []

    for sample_idx, (sample_id, snps) in enumerate(sample_snps.items()):
        if sample_idx % 100 == 0:
            print(f"Pass 2: Processed {sample_idx}/{n_samples} samples...")

        sample_ids.append(sample_id)
        for snp in snps:
            if snp in snp_to_idx:
                matrix[sample_idx, snp_to_idx[snp]] = 1

    # Convert to dense and save (in chunks if needed)
    print("Converting to DataFrame and saving...")

    # Save in chunks to avoid memory issues
    chunk_size = 10000  # Process 10k SNPs at a time

    # Start with sample IDs
    result_df = pd.DataFrame({'sample_id': sample_ids})

    for start_idx in range(0, n_snps, chunk_size):
        end_idx = min(start_idx + chunk_size, n_snps)
        print(f"Processing SNP chunk {start_idx}-{end_idx}/{n_snps}...")

        # Get SNPs for this chunk
        chunk_snps = sorted(filtered_snps)[start_idx:end_idx]

        # Extract dense data for this chunk
        chunk_data = matrix[:, start_idx:end_idx].toarray()

        # Create column names
        chunk_cols = {f'snp_{snp}': chunk_data[:, i]
                     for i, snp in enumerate(chunk_snps)}
        chunk_df = pd.DataFrame(chunk_cols)

        # Concatenate
        result_df = pd.concat([result_df, chunk_df], axis=1)

        # Force garbage collection
        del chunk_data, chunk_df
        gc.collect()

    # Save final result
    result_df.to_csv(output_file, index=False)

    print(f"✓ Combined {len(sample_snps)} samples with {len(filtered_snps)} SNPs")
    print(f"  (filtered from {len(all_snps_dict)} total unique SNPs)")

if __name__ == "__main__":
    main()