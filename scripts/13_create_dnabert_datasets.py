#!/usr/bin/env python3
"""
DNABERT Dataset Creation for Transformer Training.
Efficiently creates DNABERT datasets from ONLY the balanced samples.
Processes FASTQ files only for samples selected during balancing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import pickle
import json
import gzip
import random
from tqdm import tqdm

# K-mer based DNA tokenization following DNABERT-1 approach
def create_kmer_vocab(k=6):
    """Create k-mer vocabulary following DNABERT approach."""
    from itertools import product
    
    # Special tokens
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1, 
        '[CLS]': 2,
        '[SEP]': 3,
        '[MASK]': 4
    }
    
    # Generate all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    for i, kmer in enumerate(kmers):
        vocab[kmer] = i + 5
    
    return vocab, len(vocab)

def tokenize_dna_sequence(seq, vocab, k=6, max_len=512):
    """
    Tokenize DNA sequence using overlapping k-mers following DNABERT approach.
    """
    seq = str(seq).upper().replace('N', '')  # Remove ambiguous bases
    
    if len(seq) < k:
        # Sequence too short for k-mers
        tokens = [vocab['[CLS]'], vocab['[SEP]']]
        attention_mask = [1, 1]
    else:
        # Create overlapping k-mers
        kmers = []
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                kmers.append(kmer)
        
        # Convert k-mers to tokens
        tokens = [vocab['[CLS]']]  # Start token
        for kmer in kmers[:max_len-2]:  # Leave room for CLS and SEP
            tokens.append(vocab.get(kmer, vocab['[UNK]']))
        tokens.append(vocab['[SEP]'])  # End token
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
    
    # Pad to max length
    while len(tokens) < max_len:
        tokens.append(vocab['[PAD]'])
        attention_mask.append(0)
    
    return tokens[:max_len], attention_mask[:max_len]

def sample_reads_from_fastq(fastq_path, n_reads=1000, min_length=100, max_length=512):
    """Sample reads from a FASTQ file."""
    reads = []
    try:
        with gzip.open(fastq_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fastq'):
                seq = str(record.seq).upper()
                if min_length <= len(seq) <= max_length and 'N' not in seq:
                    reads.append(seq)
                if len(reads) >= n_reads * 2:  # Collect extra to sample from
                    break
    except Exception as e:
        print(f"Warning: Could not read {fastq_path}: {e}")
        return []
    
    if len(reads) > n_reads:
        return random.sample(reads, n_reads)
    return reads

def create_sequence_features_for_sample(sample_id, processed_dir, vocab, k=6, n_reads=1000, max_seq_len=512, sequences_per_sample=3):
    """Create sequence features for a single sample using k-mer tokenization."""
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"
    
    all_reads = []
    for path in [r1_path, r2_path]:
        if path.exists():
            reads = sample_reads_from_fastq(path, n_reads//2, max_length=max_seq_len*k)  # Longer reads for k-mer tokenization
            all_reads.extend(reads)
    
    if not all_reads:
        print(f"Warning: No reads found for sample {sample_id}")
        return None, None
    
    # Select multiple sequences per sample for better training data
    num_sequences = min(sequences_per_sample, len(all_reads))
    selected_reads = random.sample(all_reads, num_sequences) if len(all_reads) > num_sequences else all_reads
    
    # Tokenize sequences using k-mers
    tokenized_sequences = []
    attention_masks = []
    
    for read in selected_reads:
        tokens, mask = tokenize_dna_sequence(read, vocab, k, max_seq_len)
        tokenized_sequences.append(tokens)
        attention_masks.append(mask)
    
    return np.array(tokenized_sequences, dtype=np.int32), np.array(attention_masks, dtype=np.int32)

def main():
    # Get antibiotic name from input path
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in snakemake.input.balanced_train:
            antibiotic = ab
            break
    
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from input path")
    
    print(f"Creating DNABERT datasets for {antibiotic}")
    
    # Load balanced tree model datasets (to know which samples to process)
    balanced_train_df = pd.read_csv(snakemake.input.balanced_train)
    balanced_test_df = pd.read_csv(snakemake.input.balanced_test)
    
    # Get the real samples (exclude SMOTE synthetic samples)
    real_train_samples = balanced_train_df[~balanced_train_df['sample_id'].str.contains('SMOTE_synthetic', na=False)]
    real_test_samples = balanced_test_df  # Test set has no synthetic samples
    
    print(f"Balanced training samples: {len(balanced_train_df)} (real: {len(real_train_samples)})")
    print(f"Test samples: {len(real_test_samples)}")
    print(f"Training class distribution: {real_train_samples['R'].value_counts().to_dict()}")
    
    # Load processed metadata files to get SRR IDs directly 
    project_root = Path(snakemake.scriptdir).parent
    train_metadata_path = project_root / "results/features/metadata_train_processed.csv"
    test_metadata_path = project_root / "results/features/metadata_test_processed.csv"
    
    if not train_metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {train_metadata_path}")
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {test_metadata_path}")
    
    train_metadata = pd.read_csv(train_metadata_path)
    test_metadata = pd.read_csv(test_metadata_path)
    
    print(f"Loaded metadata: {len(train_metadata)} train samples, {len(test_metadata)} test samples")
    
    # Note: sample_ids in tree_models files are already SRR IDs, no mapping needed
    # Keeping this section for documentation purposes only
    print(f"Using direct SRR IDs from tree_models files (no mapping needed)")
    
    # Parameters
    processed_dir = Path(snakemake.params.processed_dir)
    max_seq_len = snakemake.params.get('max_seq_len', 512)
    n_reads = snakemake.params.get('n_reads_per_sample', 1000)
    k = snakemake.params.get('kmer_size', 6)  # K-mer size for DNABERT
    sequences_per_sample = snakemake.params.get('sequences_per_sample', 3)
    
    print(f"Max sequence length: {max_seq_len}")
    print(f"Reads per sample: {n_reads}")
    print(f"K-mer size: {k}")
    print(f"Sequences per sample: {sequences_per_sample}")
    print(f"Processed directory: {processed_dir}")
    
    # Create k-mer vocabulary
    print("Creating k-mer vocabulary...")
    vocab, vocab_size = create_kmer_vocab(k)
    print(f"Vocabulary size: {vocab_size} (k={k})")
    
    # Set random seed
    random.seed(42)
    
    # Process training samples (only real samples, not synthetic)
    print("\nProcessing real training samples for sequence extraction...")
    train_X = []
    train_attention_masks = []
    train_y = []
    train_sample_ids = []
    
    for _, row in tqdm(real_train_samples.iterrows(), total=len(real_train_samples)):
        sample_id = row['sample_id']
        label = int(row['R'])
        
        # sample_id is already the SRR ID from tree_models file
        srr_id = sample_id
        
        sequence_data, attention_masks = create_sequence_features_for_sample(
            srr_id, processed_dir, vocab, k, n_reads, max_seq_len, sequences_per_sample
        )
        
        if sequence_data is not None and len(sequence_data) > 0:
            # Add all sequences from this sample (multiple sequences per sample)
            for seq, mask in zip(sequence_data, attention_masks):
                train_X.append(seq)
                train_attention_masks.append(mask)
                train_y.append(label)
                train_sample_ids.append(sample_id)
    
    # Process test samples
    print("\nProcessing test samples...")
    test_X = []
    test_attention_masks = []
    test_y = []
    test_sample_ids = []
    
    for _, row in tqdm(real_test_samples.iterrows(), total=len(real_test_samples)):
        sample_id = row['sample_id']
        label = int(row['R'])
        
        sequence_data, attention_masks = create_sequence_features_for_sample(
            sample_id, processed_dir, vocab, k, n_reads, max_seq_len, sequences_per_sample
        )
        
        if sequence_data is not None and len(sequence_data) > 0:
            # Add all sequences from this sample (multiple sequences per sample)
            for seq, mask in zip(sequence_data, attention_masks):
                test_X.append(seq)
                test_attention_masks.append(mask)
                test_y.append(label)
                test_sample_ids.append(sample_id)
    
    # Convert to numpy arrays
    train_X = np.array(train_X, dtype=np.int32)
    train_attention_masks = np.array(train_attention_masks, dtype=np.int32)
    train_y = np.array(train_y, dtype=np.int32)
    test_X = np.array(test_X, dtype=np.int32)
    test_attention_masks = np.array(test_attention_masks, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)
    
    print(f"\nExtracted sequence features:")
    print(f"Training: X={train_X.shape}, attention_masks={train_attention_masks.shape}, y={train_y.shape}")
    print(f"Test: X={test_X.shape}, attention_masks={test_attention_masks.shape}, y={test_y.shape}")
    print(f"Real sample class distribution: {np.bincount(train_y)}")
    
    # Use original data without synthetic sampling (consistent with new strategy)
    print(f"Using original sequence data without synthetic balancing")
    print(f"Class weighting will be handled in DNABERT training")
    
    train_X_balanced = train_X
    train_attention_masks_balanced = train_attention_masks
    train_y_balanced = train_y
    
    print(f"Final DNABERT dataset: X={train_X_balanced.shape}, y={train_y_balanced.shape}")
    print(f"Class distribution: {np.bincount(train_y_balanced)}")
    
    # Ensure output directories exist
    Path(snakemake.output.train).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.test).parent.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    np.savez_compressed(snakemake.output.train,
                       X=train_X_balanced,
                       attention_mask=train_attention_masks_balanced,
                       y=train_y_balanced,
                       sample_ids=train_sample_ids[:len(train_X_balanced)] if len(train_X_balanced) <= len(train_sample_ids) else train_sample_ids)
    
    np.savez_compressed(snakemake.output.test,
                       X=test_X,
                       attention_mask=test_attention_masks,
                       y=test_y,
                       sample_ids=test_sample_ids)
    
    # Save tokenizer (vocab and reverse vocab)
    tokenizer_data = {
        'vocab': vocab,
        'id_to_token': {v: k for k, v in vocab.items()},
        'k': k,
        'vocab_size': vocab_size
    }
    with open(snakemake.output.tokenizer, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    
    # Save summary
    summary = {
        'antibiotic': antibiotic,
        'max_seq_len': max_seq_len,
        'n_reads_per_sample': n_reads,
        'kmer_size': k,
        'sequences_per_sample': sequences_per_sample,
        'vocab_size': vocab_size,
        'tokenization': 'kmer_overlapping',
        'special_tokens': ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        'real_train_samples': len(train_X),
        'balanced_train_samples': len(train_X_balanced),
        'test_samples': len(test_X),
        'real_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))},
        'balanced_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_y_balanced, return_counts=True))},
        'test_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'note': 'Created with DNABERT-1 style k-mer tokenization, multiple sequences per sample, attention masks'
    }
    
    with open(snakemake.output.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDNABERT datasets created successfully for {antibiotic}")
    print(f"Efficiently processed only {len(real_train_samples) + len(real_test_samples)} selected samples")

if __name__ == "__main__":
    main()