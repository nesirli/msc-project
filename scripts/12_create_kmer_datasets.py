#!/usr/bin/env python3
"""
K-mer Dataset Creation for 1D-CNN Training.
Efficiently creates k-mer datasets from ONLY the balanced samples.
Uses a true two-pass streaming approach to avoid OOM:
  Pass 1: build vocabulary counter, discard each sample's Counter immediately
  Pass 2: re-process each sample, create feature vector on the fly
Only one sample's Counter is in RAM at any time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from collections import Counter
import gzip
import random
from tqdm import tqdm
import json


def get_sample_kmers(sample_id, processed_dir, k, n_reads):
    """
    Stream k-mer counts for one sample.
    Returns a Counter or None. Caller must del it after use.
    """
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"

    kmer_counter = Counter()
    total_reads = 0

    for path in [r1_path, r2_path]:
        if not path.exists() or total_reads >= n_reads:
            continue
        try:
            with gzip.open(path, 'rt') as f:
                for record in SeqIO.parse(f, 'fastq'):
                    if total_reads >= n_reads:
                        break
                    seq = str(record.seq).upper()
                    if len(seq) >= 50 and 'N' not in seq:
                        for i in range(len(seq) - k + 1):
                            kmer = seq[i:i + k]
                            if 'N' not in kmer:
                                kmer_counter[kmer] += 1
                        total_reads += 1
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")

    return kmer_counter if kmer_counter else None


def main():
    # Determine antibiotic from input path
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in snakemake.input.balanced_train:
            antibiotic = ab
            break
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from input path")

    print(f"Creating k-mer datasets for {antibiotic}")

    # Load sample lists
    balanced_train_df = pd.read_csv(snakemake.input.balanced_train)
    balanced_test_df = pd.read_csv(snakemake.input.balanced_test)

    real_train_samples = balanced_train_df[
        ~balanced_train_df['sample_id'].str.contains('SMOTE_synthetic', na=False)
    ]
    real_test_samples = balanced_test_df

    print(f"Balanced training samples: {len(balanced_train_df)} (real: {len(real_train_samples)})")
    print(f"Test samples: {len(real_test_samples)}")
    print(f"Training class distribution: {real_train_samples['R'].value_counts().to_dict()}")

    # Parameters
    processed_dir = Path(snakemake.params.processed_dir)
    k = snakemake.params.get('kmer_size', 9)
    n_reads = snakemake.params.get('n_reads_per_sample', 5000)
    max_features = 10000

    print(f"K-mer size: {k}")
    print(f"Reads per sample: {n_reads}")
    print(f"Processed directory: {processed_dir}")

    random.seed(42)

    all_rows = list(real_train_samples.iterrows()) + list(real_test_samples.iterrows())

    # Track which samples are valid (have reads) — needed to align labels
    valid_sample_ids = []
    valid_labels = []

    # -----------------------------------------------------------------------
    # PASS 1: build vocabulary — one Counter at a time, discard immediately
    # -----------------------------------------------------------------------
    print("\nPass 1: building vocabulary (streaming, one sample at a time)...")
    vocabulary_counter = Counter()

    for _, row in tqdm(all_rows, desc="Vocabulary pass"):
        sample_id = row['sample_id']
        label = int(row['R'])

        kmer_counts = get_sample_kmers(sample_id, processed_dir, k, n_reads)
        if kmer_counts is None:
            continue

        # Update global vocabulary with just the keys (presence, not counts)
        vocabulary_counter.update(kmer_counts.keys())
        valid_sample_ids.append(sample_id)
        valid_labels.append(label)

        del kmer_counts  # free immediately — this is the key fix

    print(f"Successfully scanned {len(valid_sample_ids)} samples")
    print(f"Building vocabulary: selecting top {max_features} from {len(vocabulary_counter)} total k-mers...")

    top_kmers = [kmer for kmer, _ in vocabulary_counter.most_common(max_features)]
    total_observed_kmers = len(vocabulary_counter)
    kmer_index = {kmer: i for i, kmer in enumerate(top_kmers)}  # O(1) lookup
    n_features = len(top_kmers)

    del vocabulary_counter  # free before pass 2

    print(f"Selected {n_features} k-mers")

    # -----------------------------------------------------------------------
    # PASS 2: create feature vectors — one sample at a time, stream to matrix
    # -----------------------------------------------------------------------
    print("\nPass 2: creating feature matrix (streaming)...")
    n_samples = len(valid_sample_ids)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, sample_id in enumerate(tqdm(valid_sample_ids, desc="Feature pass")):
        kmer_counts = get_sample_kmers(sample_id, processed_dir, k, n_reads)
        if kmer_counts is None:
            continue  # row stays zero

        total_count = sum(kmer_counts.values())
        if total_count > 0:
            for kmer, count in kmer_counts.items():
                idx = kmer_index.get(kmer)
                if idx is not None:
                    X[i, idx] = count / total_count

        del kmer_counts  # free immediately

    y = np.array(valid_labels, dtype=np.int32)

    # Split train / test
    n_train = sum(
        1 for sid in valid_sample_ids
        if sid in set(real_train_samples['sample_id'])
    )
    # Preserve original order: train rows first, then test rows
    train_mask = [sid in set(real_train_samples['sample_id']) for sid in valid_sample_ids]
    test_mask  = [not m for m in train_mask]

    train_X = X[train_mask]
    train_y = y[train_mask]
    train_ids = [sid for sid, m in zip(valid_sample_ids, train_mask) if m]

    test_X = X[test_mask]
    test_y = y[test_mask]
    test_ids = [sid for sid, m in zip(valid_sample_ids, test_mask) if m]

    del X  # free full matrix now that we have splits

    print(f"\nExtracted k-mer features:")
    print(f"Training: X={train_X.shape}, y={train_y.shape}")
    print(f"Test: X={test_X.shape}, y={test_y.shape}")
    print(f"Class distribution (train): {np.bincount(train_y)}")

    # Save outputs
    Path(snakemake.output.train).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.test).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        snakemake.output.train,
        X=train_X, y=train_y,
        sample_ids=train_ids,
        feature_names=top_kmers
    )
    np.savez_compressed(
        snakemake.output.test,
        X=test_X, y=test_y,
        sample_ids=test_ids,
        feature_names=top_kmers
    )

    summary = {
        'antibiotic': antibiotic,
        'k': k,
        'n_reads_per_sample': n_reads,
        'selected_kmers': n_features,
        'total_observed_kmers': total_observed_kmers,
        'train_samples': int(train_X.shape[0]),
        'test_samples': int(test_X.shape[0]),
        'train_class_distribution': {
            int(cls): int(cnt)
            for cls, cnt in zip(*np.unique(train_y, return_counts=True))
        },
        'test_class_distribution': {
            int(cls): int(cnt)
            for cls, cnt in zip(*np.unique(test_y, return_counts=True))
        },
        'note': 'Two-pass streaming: vocabulary built in pass 1, features in pass 2. No simultaneous Counter storage.'
    }

    with open(snakemake.output.summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nK-mer datasets created successfully for {antibiotic}")


if __name__ == "__main__":
    main()