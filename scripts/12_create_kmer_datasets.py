#!/usr/bin/env python3
"""
K-mer Dataset Creation for 1D-CNN Training.
Efficiently creates k-mer datasets from ONLY the balanced samples.
Processes FASTQ files only for samples selected during balancing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from collections import Counter
from itertools import product
import gzip
import random
from tqdm import tqdm
import json

def count_kmers(sequence, k):
    """Count k-mers in a sequence."""
    kmers = Counter()
    seq = str(sequence).upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:  # Skip k-mers with ambiguous bases
            kmers[kmer] += 1
    return kmers

def get_sample_kmers_efficiently(sample_id, processed_dir, k, n_reads):
    """Get k-mer counts for a sample without storing full vectors."""
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"
    
    kmer_counter = Counter()
    total_reads_processed = 0
    
    for path in [r1_path, r2_path]:
        if path.exists() and total_reads_processed < n_reads:
            try:
                with gzip.open(path, 'rt') as f:
                    for record in SeqIO.parse(f, 'fastq'):
                        if total_reads_processed >= n_reads:
                            break
                        seq = str(record.seq).upper()
                        if len(seq) >= 50 and 'N' not in seq:
                            # Count k-mers in this read
                            for i in range(len(seq) - k + 1):
                                kmer = seq[i:i+k]
                                if 'N' not in kmer:
                                    kmer_counter[kmer] += 1
                            total_reads_processed += 1
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
    
    return kmer_counter if kmer_counter else None

def sample_reads_from_fastq(fastq_path, n_reads=10000, min_length=50):
    """Sample reads from a FASTQ file."""
    reads = []
    try:
        with gzip.open(fastq_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fastq'):
                seq = str(record.seq).upper()
                if len(seq) >= min_length and 'N' not in seq:
                    reads.append(seq)
                if len(reads) >= n_reads * 2:  # Collect extra to sample from
                    break
    except Exception as e:
        print(f"Warning: Could not read {fastq_path}: {e}")
        return []
    
    if len(reads) > n_reads:
        return random.sample(reads, n_reads)
    return reads

def create_kmer_features_for_sample(sample_id, processed_dir, k=11, n_reads=10000):
    """Create k-mer features for a single sample."""
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"
    
    all_reads = []
    for path in [r1_path, r2_path]:
        if path.exists():
            reads = sample_reads_from_fastq(path, n_reads//2)
            all_reads.extend(reads)
    
    if not all_reads:
        print(f"Warning: No reads found for sample {sample_id}")
        return None
    
    # Count k-mers across all reads for this sample
    total_kmers = Counter()
    for read in all_reads:
        kmers = count_kmers(read, k)
        total_kmers.update(kmers)
    
    return total_kmers

def main():
    # Get antibiotic name from input path
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in snakemake.input.balanced_train:
            antibiotic = ab
            break
    
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from input path")
    
    print(f"Creating k-mer datasets for {antibiotic}")
    
    # Load balanced tree model datasets (to know which samples to process)
    balanced_train_df = pd.read_csv(snakemake.input.balanced_train)
    balanced_test_df = pd.read_csv(snakemake.input.balanced_test)
    
    # Get the real samples (exclude SMOTE synthetic samples)
    real_train_samples = balanced_train_df[~balanced_train_df['sample_id'].str.contains('SMOTE_synthetic', na=False)]
    real_test_samples = balanced_test_df  # Test set has no synthetic samples
    
    print(f"Balanced training samples: {len(balanced_train_df)} (real: {len(real_train_samples)})")
    print(f"Test samples: {len(real_test_samples)}")
    print(f"Training class distribution: {real_train_samples['R'].value_counts().to_dict()}")
    
    # Parameters - use smaller k-mer size for memory efficiency
    processed_dir = Path(snakemake.params.processed_dir)
    k = snakemake.params.get('kmer_size', 9)  # Reduced from 11 to 9 for memory
    n_reads = snakemake.params.get('n_reads_per_sample', 5000)  # Reduced reads for faster processing
    
    print(f"K-mer size: {k}")
    print(f"Reads per sample: {n_reads}")
    print(f"Processed directory: {processed_dir}")
    
    # Set random seed
    random.seed(42)
    
    # First pass: collect all k-mers from all samples to identify the vocabulary
    print("\nFirst pass: collecting k-mer vocabulary...")
    all_sample_kmers = []
    sample_labels = []
    sample_ids = []
    
    # Process all samples to collect k-mers efficiently
    all_samples = list(real_train_samples.iterrows()) + list(real_test_samples.iterrows())
    for _, row in tqdm(all_samples, desc="Collecting k-mers"):
        sample_id = row['sample_id']
        label = int(row['R'])
        
        kmer_counts = get_sample_kmers_efficiently(sample_id, processed_dir, k, n_reads)
        
        if kmer_counts is not None:
            all_sample_kmers.append(kmer_counts)
            sample_labels.append(label)
            sample_ids.append(sample_id)
    
    print(f"Successfully processed {len(all_sample_kmers)} samples")
    
    # Find most frequent k-mers across all samples (memory-efficient vocabulary)
    print("Creating memory-efficient k-mer vocabulary...")
    vocabulary_counter = Counter()
    for kmer_counts in all_sample_kmers:
        vocabulary_counter.update(kmer_counts.keys())
    
    # Select top k-mers to keep feature space manageable - be more aggressive for memory
    max_features = min(10000, len(vocabulary_counter))  # Limit to 10K features for memory efficiency
    top_kmers = [kmer for kmer, count in vocabulary_counter.most_common(max_features)]
    total_observed_kmers = len(vocabulary_counter)
    print(f"Selected top {len(top_kmers)} k-mers (from {total_observed_kmers} total)")
    
    # Clear vocabulary counter to free memory
    del vocabulary_counter
    
    # Create feature vectors for all samples
    print("Creating feature vectors...")
    all_X = []
    for kmer_counts in tqdm(all_sample_kmers, desc="Converting to features"):
        total_count = sum(kmer_counts.values())
        if total_count > 0:
            feature_vector = [kmer_counts.get(kmer, 0) / total_count for kmer in top_kmers]
            all_X.append(feature_vector)
        else:
            all_X.append([0.0] * len(top_kmers))
    
    # Split back into train/test
    n_train = len(real_train_samples)
    train_X = all_X[:n_train]
    train_y = sample_labels[:n_train]
    train_sample_ids = sample_ids[:n_train]
    
    test_X = all_X[n_train:]
    test_y = sample_labels[n_train:]
    test_sample_ids = sample_ids[n_train:]
    
    # Convert to numpy arrays
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)
    
    print(f"\nExtracted k-mer features:")
    print(f"Training: X={train_X.shape}, y={train_y.shape}")
    print(f"Test: X={test_X.shape}, y={test_y.shape}")
    print(f"Real sample class distribution: {np.bincount(train_y)}")
    
    # Use original data without synthetic sampling (consistent with new strategy)
    print(f"Using original k-mer data without synthetic balancing")
    print(f"Class weighting will be handled in model training")
    
    train_X_balanced = train_X
    train_y_balanced = train_y
    
    print(f"Final k-mer dataset: X={train_X_balanced.shape}, y={train_y_balanced.shape}")
    print(f"Class distribution: {np.bincount(train_y_balanced)}")
    
    # Ensure output directories exist
    Path(snakemake.output.train).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.test).parent.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    np.savez_compressed(snakemake.output.train,
                       X=train_X_balanced,
                       y=train_y_balanced,
                       sample_ids=train_sample_ids[:len(train_X_balanced)] if len(train_X_balanced) <= len(train_sample_ids) else train_sample_ids,
                       feature_names=top_kmers)
    
    np.savez_compressed(snakemake.output.test,
                       X=test_X,
                       y=test_y,
                       sample_ids=test_sample_ids,
                       feature_names=top_kmers)
    
    # Save summary
    summary = {
        'antibiotic': antibiotic,
        'k': k,
        'n_reads_per_sample': n_reads,
        'selected_kmers': len(top_kmers),
        'total_observed_kmers': total_observed_kmers,
        'real_train_samples': len(train_X),
        'balanced_train_samples': len(train_X_balanced),
        'test_samples': len(test_X),
        'real_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))},
        'balanced_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_y_balanced, return_counts=True))},
        'test_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'note': 'Created from balanced tree model selection, then balanced again for deep learning'
    }
    
    with open(snakemake.output.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nK-mer datasets created successfully for {antibiotic}")
    print(f"Efficiently processed only {len(real_train_samples) + len(real_test_samples)} selected samples")

if __name__ == "__main__":
    main()