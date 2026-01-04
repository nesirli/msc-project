#!/usr/bin/env python3
"""
Raw Sequence CNN Training for AMR Prediction.
PyTorch implementation using one-hot encoded ACGT sequences.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix
)
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import warnings
import gzip
import random
import sys
import os
from collections import Counter

warnings.filterwarnings('ignore')

try:
    from Bio import SeqIO
except ImportError:
    print("Warning: BioPython not available. Install with: conda install -c bioconda biopython")
    sys.exit(1)

# Configure PyTorch parallelism based on Snakemake threads
def configure_pytorch_threads():
    """Configure PyTorch to use all available threads from Snakemake."""
    try:
        num_threads = snakemake.threads
        print(f"Configuring PyTorch to use {num_threads} threads")

        # Set PyTorch intraop parallelism (within operations)
        torch.set_num_threads(num_threads)

        # Set environment variables for various backends
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

        # Calculate num_workers for DataLoader (leave some threads for computation)
        # Use approximately 1/4 of threads for data loading, rest for computation
        num_workers = max(1, num_threads // 4)

        return num_workers
    except (NameError, AttributeError):
        # Fallback if not running under Snakemake
        print("Warning: Not running under Snakemake, using default thread settings")
        return 2

class GeographicTemporalKFold(BaseCrossValidator):
    """
    K-Fold cross-validation that respects geographic and temporal structure.
    Ensures strains from the same location-year combination are not split across training/validation.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups parameter (location-year combinations) is required for geographic-temporal CV")
        
        groups = np.array(groups)
        y = np.array(y) if y is not None else None
        
        # Get unique groups (location-year combinations)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(f"Number of location-year groups ({n_groups}) < n_splits ({self.n_splits})")
        
        # Set random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        # Shuffle groups if requested
        if self.shuffle:
            rng.shuffle(unique_groups)
        
        # Simple round-robin assignment for small datasets
        group_fold_assignments = {}
        for i, group in enumerate(unique_groups):
            group_fold_assignments[group] = i % self.n_splits
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            test_groups = [group for group, assigned_fold in group_fold_assignments.items() 
                          if assigned_fold == fold]
            train_groups = [group for group in unique_groups if group not in test_groups]
            
            # Get indices
            test_idx = np.concatenate([np.where(groups == group)[0] for group in test_groups])
            train_idx = np.concatenate([np.where(groups == group)[0] for group in train_groups])
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class SequenceDataset(Dataset):
    """PyTorch dataset for one-hot encoded DNA sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class SequenceCNN(nn.Module):
    """CNN for raw DNA sequence classification."""
    
    def __init__(self, seq_length, num_classes=2, dropout=0.5):
        super(SequenceCNN, self).__init__()
        
        self.seq_length = seq_length
        
        # Convolutional layers - designed for sequence patterns
        self.conv1 = nn.Conv1d(4, 32, kernel_size=15, padding=7)  # Motif detection
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)  # Pattern combinations
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)  # Higher-order patterns
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)  # Complex patterns
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers with heavy regularization
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # For interpretability
        self.feature_maps = None
        
    def forward(self, x):
        # Input: (batch_size, 4, seq_length) - already one-hot encoded
        
        # Convolutional layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        
        # Store feature maps for interpretability
        self.feature_maps = x.detach()
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def one_hot_encode_sequence(sequence, max_length):
    """Convert DNA sequence to one-hot encoding."""
    # Mapping: A=0, C=1, G=2, T=3
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Initialize with zeros
    encoded = np.zeros((4, max_length))
    
    # Truncate or pad sequence
    sequence = sequence[:max_length]
    
    for i, base in enumerate(sequence):
        if base in base_to_index:
            encoded[base_to_index[base], i] = 1
    
    return encoded

def extract_sequences_from_sample(sample_id, processed_dir, n_reads=1000, min_length=150):
    """Extract high-quality sequences from FASTQ files."""
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"
    
    sequences = []
    total_reads_processed = 0
    
    # Check file existence with detailed logging
    if not r1_path.exists() and not r2_path.exists():
        print(f"Warning: No FASTQ files found for sample {sample_id} in {processed_dir}")
        print(f"  Checked: {r1_path}")
        print(f"  Checked: {r2_path}")
        # Check if any files exist for this sample ID
        possible_files = list(processed_dir.glob(f"{sample_id}*"))
        if possible_files:
            print(f"  Found other files: {[f.name for f in possible_files[:5]]}")
        return sequences
    
    for path in [r1_path, r2_path]:
        if path.exists() and total_reads_processed < n_reads:
            try:
                with gzip.open(path, 'rt') as f:
                    record_count = 0
                    for record in SeqIO.parse(f, 'fastq'):
                        if total_reads_processed >= n_reads:
                            break
                        
                        record_count += 1
                        seq = str(record.seq).upper()
                        
                        # Quality filtering
                        if (len(seq) >= min_length and 
                            'N' not in seq and 
                            len(set(seq) & {'A', 'C', 'G', 'T'}) >= 3):  # At least 3 different bases
                            
                            sequences.append(seq)
                            total_reads_processed += 1
                        
                        # Break early if we have enough sequences
                        if len(sequences) >= n_reads:
                            break
                    
                    # Debug for test samples that don't produce sequences
                    if sample_id.startswith('SRR24673') and len(sequences) == 0:
                        print(f"Debug {sample_id}: {path.name} - {record_count} total records, 0 passed quality filter")
                            
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    return sequences

def prepare_sequence_dataset(train_samples, test_samples, processed_dir, 
                           n_reads_per_sample=1000, max_seq_length=200):
    """Prepare one-hot encoded sequence dataset."""
    print("Extracting sequences from samples...")
    
    all_sequences = []
    all_labels = []
    sample_ids = []
    
    # Process training samples
    for _, row in tqdm(train_samples.iterrows(), desc="Processing train samples", total=len(train_samples)):
        sample_id = row['sample_id']
        label = int(row['R'])
        
        sequences = extract_sequences_from_sample(sample_id, processed_dir, n_reads_per_sample)
        
        # Take a subset of sequences per sample for memory efficiency
        if sequences:
            # Randomly sample sequences if we have more than needed
            selected_sequences = sequences[:min(30, len(sequences))]  # Max 30 sequences per sample (reduced from 50 for faster I/O)

            for seq_idx, seq in enumerate(selected_sequences):
                all_sequences.append(seq)
                all_labels.append(label)
                sample_ids.append(f"{sample_id}_{seq_idx + 1}")
    
    # Process test samples
    for _, row in tqdm(test_samples.iterrows(), desc="Processing test samples", total=len(test_samples)):
        sample_id = row['sample_id']
        label = int(row['R'])
        
        sequences = extract_sequences_from_sample(sample_id, processed_dir, n_reads_per_sample)
        
        # Debug for test samples
        if sample_id.startswith('SRR24673'):
            print(f"Debug: Test sample {sample_id} extracted {len(sequences)} sequences")
        
        if sequences:
            selected_sequences = sequences[:min(30, len(sequences))]  # Max 30 sequences per sample (reduced from 50 for faster I/O)

            for seq_idx, seq in enumerate(selected_sequences):
                all_sequences.append(seq)
                all_labels.append(label)
                sample_ids.append(f"{sample_id}_{seq_idx + 1}")
    
    print(f"Extracted {len(all_sequences)} sequences total")
    
    if len(all_sequences) == 0:
        raise ValueError(f"No sequences extracted. Check FASTQ files in {processed_dir}")
    
    print(f"Class distribution: {Counter(all_labels)}")
    
    # Debug: Show sample of sequence IDs to verify test sequences are included
    print(f"Sample sequence IDs: {sample_ids[:5]} ... {sample_ids[-5:]}")
    train_seq_count = sum(1 for sid in sample_ids if not sid.startswith('SRR24673'))
    test_seq_count = sum(1 for sid in sample_ids if sid.startswith('SRR24673'))
    print(f"In sample_ids: {train_seq_count} training sequences, {test_seq_count} test sequences")
    
    # Determine actual maximum length for efficiency
    seq_lengths = [len(seq) for seq in all_sequences]
    actual_max_length = min(max_seq_length, int(np.percentile(seq_lengths, 95)))
    print(f"Using sequence length: {actual_max_length} (95th percentile: {int(np.percentile(seq_lengths, 95))})")
    
    # One-hot encode all sequences
    print("One-hot encoding sequences...")
    encoded_sequences = []
    for seq in tqdm(all_sequences, desc="Encoding"):
        encoded = one_hot_encode_sequence(seq, actual_max_length)
        encoded_sequences.append(encoded)
    
    encoded_sequences = np.array(encoded_sequences)
    all_labels = np.array(all_labels)
    
    print(f"Final dataset shape: {encoded_sequences.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return encoded_sequences, all_labels, sample_ids, actual_max_length


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def cross_validation(X, y, sample_ids, location_year_groups=None, cv_folds=5, random_state=42, num_workers=2, **model_params):
    """Perform cross-validation with sequence data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"DataLoader workers: {num_workers}")
    
    # For sequence data, we need to split by original sample IDs to avoid data leakage
    # Extract original sample IDs
    original_samples = {}
    for i, sample_id in enumerate(sample_ids):
        original_id = sample_id.split('_')[0]  # Get original sample ID
        if original_id not in original_samples:
            original_samples[original_id] = []
        original_samples[original_id].append(i)
    
    # Create sample-level labels for stratification
    sample_labels = []
    sample_names = []
    for orig_id, indices in original_samples.items():
        # Use majority label for this sample
        labels_for_sample = y[indices]
        majority_label = Counter(labels_for_sample).most_common(1)[0][0]
        sample_labels.append(majority_label)
        sample_names.append(orig_id)
    
    sample_labels = np.array(sample_labels)
    
    # Check if we have enough samples for cross-validation
    if len(sample_labels) == 0:
        raise ValueError("No samples found for cross-validation. Check sequence extraction.")
    
    if len(sample_labels) < cv_folds:
        raise ValueError(f"Not enough samples ({len(sample_labels)}) for {cv_folds}-fold cross-validation")
    
    # Map sample names to SNP clusters if available
    sample_location_year = None
    if location_year_groups is not None:
        sample_cluster_map = {}
        for i, sample_id in enumerate(sample_ids):
            original_id = sample_id.split('_')[0]
            if original_id not in sample_cluster_map:
                sample_cluster_map[original_id] = location_year_groups[i]
        
        sample_location_year = np.array([sample_cluster_map.get(name, 'unknown') 
                                       for name in sample_names])
        print(f"Sample-level location-year groups: {len(np.unique(sample_location_year))} unique")
    
    # Use phylogenetic CV if SNP cluster info available
    if sample_location_year is not None:
        # Check if we have enough unique groups for geographic-temporal CV
        unique_groups = len(np.unique(sample_location_year))
        if unique_groups >= cv_folds:
            try:
                cv_splitter = GeographicTemporalKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                print("Using phylogenetic-aware cross-validation")
            except ValueError as e:
                print(f"Warning: Cannot use phylogenetic CV ({e}), falling back to stratified CV")
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                sample_location_year = None
        else:
            print(f"Warning: Only {unique_groups} location-year groups available for {cv_folds}-fold CV, falling back to stratified CV")
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            sample_location_year = None
    else:
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        print("Using stratified cross-validation")
    
    cv_results = []
    fold_models = []
    
    for fold, (train_samples_idx, val_samples_idx) in enumerate(cv_splitter.split(sample_names, sample_labels, groups=sample_location_year)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        
        # Get sequence indices for training and validation
        train_idx = []
        val_idx = []
        
        for sample_idx in train_samples_idx:
            orig_id = sample_names[sample_idx]
            train_idx.extend(original_samples[orig_id])
        
        for sample_idx in val_samples_idx:
            orig_id = sample_names[sample_idx]
            val_idx.extend(original_samples[orig_id])
        
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        print(f"Fold {fold + 1}: Train={len(X_train_fold)}, Val={len(X_val_fold)}")
        print(f"Train classes: {Counter(y_train_fold)}, Val classes: {Counter(y_val_fold)}")
        
        # Log cluster information if available
        if sample_location_year is not None:
            train_groups = [sample_location_year[i] for i in train_samples_idx]
            val_groups = [sample_location_year[i] for i in val_samples_idx]
            print(f"  Train sample clusters: {len(np.unique(train_groups))}, "
                  f"Val sample clusters: {len(np.unique(val_groups))}")
        
        # Create datasets
        train_dataset = SequenceDataset(X_train_fold, y_train_fold)
        val_dataset = SequenceDataset(X_val_fold, y_val_fold)
        
        # Create dataloaders (no weighted sampling - use class weights in loss instead)
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params.get('batch_size', 16),  # Smaller batch size for sequences
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_params.get('batch_size', 16),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model
        model = SequenceCNN(
            seq_length=X.shape[2],
            dropout=model_params.get('dropout', 0.5)
        ).to(device)
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train_fold, minlength=2)  # Ensure we have weights for both classes
        total_samples = len(y_train_fold)
        
        # Handle case where a class is missing (count=0)
        class_weights = []
        for count in class_counts:
            if count > 0:
                class_weights.append(total_samples / (len(class_counts) * count))
            else:
                class_weights.append(0.0)  # No weight for missing class
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        print(f"  Fold {fold + 1} class distribution: {class_counts}")
        print(f"  Calculated class weights: {class_weights.cpu().numpy()}")
        
        # Loss and optimizer with class weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_params.get('learning_rate', 0.0001),  # Lower learning rate
            weight_decay=model_params.get('weight_decay', 1e-3)  # Higher weight decay
        )
        
        # Training loop
        epochs = model_params.get('epochs', 50)  # Fewer epochs
        best_val_auc = 0
        best_model_state = model.state_dict().copy()
        patience = model_params.get('patience', 15)  # Increased from 10 for more stable training
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_preds, val_probs, val_labels = evaluate(model, val_loader, device)

            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict().copy())
        
        # Final evaluation
        val_preds, val_probs, val_labels = evaluate(model, val_loader, device)
        
        fold_results = {
            'fold': fold,
            'f1': f1_score(val_labels, val_preds, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(val_labels, val_preds),
            'auc': roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(val_labels, val_preds).tolist()
        }
        
        cv_results.append(fold_results)
        
        print(f"Fold {fold + 1} results:")
        print(f"  F1: {fold_results['f1']:.4f}")
        print(f"  Balanced Accuracy: {fold_results['balanced_accuracy']:.4f}")
        print(f"  AUC: {fold_results['auc']:.4f}")
    
    return cv_results, fold_models

def main():
    # Configure PyTorch threading for optimal CPU utilization
    num_workers = configure_pytorch_threads()

    # Check if running under Snakemake or standalone
    if 'snakemake' not in globals():
        print("Error: This script must be run from Snakemake pipeline.")
        print("Use: snakemake --use-conda --cores 8 results/models/sequence_cnn/{antibiotic}_results.json")
        sys.exit(1)

    try:
        # Load sample information from tree model datasets
        train_df = pd.read_csv(snakemake.input.train)
        test_df = pd.read_csv(snakemake.input.test)
        
        # Get antibiotic name from output path
        antibiotic = None
        for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
            if ab in str(snakemake.output.model):
                antibiotic = ab
                break
        
        if antibiotic is None:
            raise ValueError("Could not determine antibiotic from output path")
    except Exception as e:
        print(f"Error accessing Snakemake inputs/outputs: {e}")
        sys.exit(1)
    
    print(f"=== RAW SEQUENCE CNN TRAINING - {antibiotic.upper()} ===")
    
    # Get the real samples (exclude SMOTE synthetic samples)
    real_train_samples = train_df[~train_df['sample_id'].str.contains('SMOTE_synthetic', na=False)]
    real_test_samples = test_df
    
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
    
    # Use SRR IDs directly from the processed metadata files
    # Training: get SRR IDs for the samples that have sequence data
    train_srr_ids = train_metadata['Run'].tolist()
    test_srr_ids = test_metadata['Run'].tolist()
    
    print(f"Train SRR IDs: {len(train_srr_ids)} samples")
    print(f"Test SRR IDs: {len(test_srr_ids)} samples") 
    print(f"Sample train IDs: {train_srr_ids[:5]}")
    print(f"Sample test IDs: {test_srr_ids[:5]}")
    
    # Check for overlap - there should be NO overlap
    overlap = set(train_srr_ids).intersection(set(test_srr_ids))
    if overlap:
        print(f"WARNING: Found {len(overlap)} samples in both train and test sets!")
        raise ValueError(f"Data leakage detected: {list(overlap)[:5]}")
    
    # Sample IDs in tree model files are already SRR IDs - no mapping needed!
    # Map labels directly from tree model data
    train_srr_to_label = {}
    for _, row in real_train_samples.iterrows():
        train_srr_to_label[row['sample_id']] = int(row['R'])

    test_srr_to_label = {}
    for _, row in real_test_samples.iterrows():
        test_srr_to_label[row['sample_id']] = int(row['R'])
    
    # Create dataframes with proper labels
    print(f"Building training labels from {len(train_srr_to_label)} mapped samples")
    print(f"Building test labels from {len(test_srr_to_label)} mapped samples")

    train_labels = []
    train_matched = 0
    for srr_id in train_srr_ids:
        if srr_id in train_srr_to_label:
            train_labels.append(train_srr_to_label[srr_id])
            train_matched += 1
        else:
            train_labels.append(0)  # Default to susceptible if not found

    test_labels = []
    test_matched = 0
    for srr_id in test_srr_ids:
        if srr_id in test_srr_to_label:
            test_labels.append(test_srr_to_label[srr_id])
            test_matched += 1
        else:
            test_labels.append(0)

    print(f"Matched {train_matched}/{len(train_srr_ids)} training labels")
    print(f"Matched {test_matched}/{len(test_srr_ids)} test labels")

    train_for_sequences = pd.DataFrame({
        'sample_id': train_srr_ids,
        'R': train_labels
    })

    test_for_sequences = pd.DataFrame({
        'sample_id': test_srr_ids,
        'R': test_labels
    })

    print(f"Prepared {len(train_for_sequences)} training samples")
    print(f"Prepared {len(test_for_sequences)} test samples")
    print(f"Train labels: {train_for_sequences['R'].value_counts().to_dict()}")
    print(f"Test labels: {test_for_sequences['R'].value_counts().to_dict()}")
    
    # Check that we have both classes in training data
    if len(train_for_sequences['R'].unique()) < 2:
        print(f"WARNING: Training data only has {len(train_for_sequences['R'].unique())} class(es)")
    if len(test_for_sequences['R'].unique()) < 2:
        print(f"WARNING: Test data only has {len(test_for_sequences['R'].unique())} class(es)")
    
    # Prepare sequence dataset using SRR IDs
    try:
        processed_dir = Path(snakemake.params.processed_dir)
    except AttributeError:
        # Fallback to default path
        processed_dir = project_root / "data/processed"
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Get parameters with safe defaults
    n_reads_per_sample = getattr(snakemake.params, 'n_reads_per_sample', 1000)
    max_seq_length = getattr(snakemake.params, 'max_seq_length', 200)
    
    X, y, sample_ids, seq_length = prepare_sequence_dataset(
        train_for_sequences, test_for_sequences, processed_dir,
        n_reads_per_sample=n_reads_per_sample,
        max_seq_length=max_seq_length
    )
    
    # Split into train/test based on SRR IDs from metadata files
    train_sample_names = set(train_srr_ids)
    test_sample_names = set(test_srr_ids)
    
    print(f"Train SRR set: {len(train_sample_names)} samples")
    print(f"Test SRR set: {len(test_sample_names)} samples")
    print(f"Available sequence sample IDs (first 10): {sample_ids[:10]}")
    
    train_indices = []
    test_indices = []
    
    for i, sample_id in enumerate(sample_ids):
        original_id = sample_id.split('_')[0]  # Extract SRR ID from sequence sample ID
        
        # Debug first few samples
        if i < 5:
            is_train = original_id in train_sample_names
            is_test = original_id in test_sample_names
            print(f"Debug split: {sample_id} -> {original_id}, train={is_train}, test={is_test}")
        
        if original_id in train_sample_names:
            train_indices.append(i)
        elif original_id in test_sample_names:
            test_indices.append(i)
    
    print(f"Found {len(train_indices)} training sequences, {len(test_indices)} test sequences")
    
    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    
    # Check if we have any training samples
    if len(train_indices) == 0:
        raise ValueError(f"No training sequences found. Train samples: {len(real_train_samples)}, "
                        f"available sample IDs: {sample_ids[:10] if sample_ids else 'None'}")
    
    if len(test_indices) == 0:
        raise ValueError(f"No test sequences found. Test samples: {len(real_test_samples)}, "
                        f"available sample IDs: {sample_ids[:10] if sample_ids else 'None'}")
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    train_sample_ids = [sample_ids[i] for i in train_indices]
    
    print(f"Training sequences: {X_train.shape[0]} from {len(real_train_samples)} samples")
    print(f"Test sequences: {X_test.shape[0]} from {len(real_test_samples)} samples")
    print(f"Sequence length: {seq_length}")
    print(f"Training class distribution: {Counter(y_train)}")
    print(f"Test class distribution: {Counter(y_test)}")
    
    # Model parameters with safe defaults
    model_params = {
        'epochs': getattr(snakemake.params, 'epochs', 50),
        'batch_size': getattr(snakemake.params, 'batch_size', 16),
        'learning_rate': getattr(snakemake.params, 'learning_rate', 0.0001),
        'dropout': getattr(snakemake.params, 'dropout', 0.5),
        'weight_decay': getattr(snakemake.params, 'weight_decay', 1e-3),
        'patience': getattr(snakemake.params, 'patience', 15),  # Increased from 10 for more stable training
        'num_workers': num_workers  # Add num_workers for DataLoader parallelism
    }

    cv_folds = getattr(snakemake.params, 'cv_folds', 5)
    random_state = getattr(snakemake.params, 'random_state', 42)
    
    # Create location-year groups for geographic-temporal CV
    location_year_train = None
    try:
        # Get antibiotic name to find corresponding metadata
        antibiotic = None
        for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
            if ab in str(snakemake.input.train):
                antibiotic = ab
                break
        
        if antibiotic:
            # Load training metadata to get location and year
            metadata_path = str(snakemake.input.train).replace(f'sequence_datasets/{antibiotic}_train_sequence.npz', 
                                                             f'tree_models/{antibiotic}_train_final.csv')
            train_metadata = pd.read_csv(metadata_path)
            
            # Create sample_id to location-year mapping
            train_metadata['Location_Year'] = (train_metadata['Location'].fillna('unknown').astype(str) + '_' + 
                                              train_metadata['Year'].fillna('unknown').astype(str))
            sample_to_group = dict(zip(train_metadata['sample_id'], 
                                     train_metadata['Location_Year']))
            
            # Map sequence sample IDs to location-year groups (using original sample ID)
            location_year_train = np.array([sample_to_group.get(sid.split('_')[0], 'unknown') 
                                          for sid in train_sample_ids])
            
            print(f"Loaded location-year info for {len(location_year_train)} sequences")
            print(f"Unique location-year groups: {len(np.unique(location_year_train))}")
            print(f"Location-year distribution: {Counter(location_year_train).most_common(10)}")
    except Exception as e:
        print(f"Warning: Could not load location-year information: {e}")
        location_year_train = None
    
    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_results, fold_models = cross_validation(
        X_train, y_train, train_sample_ids, location_year_train, cv_folds, random_state, **model_params
    )
    
    # Train final model on all training data
    print(f"\nTraining final model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    
    # Create dataloaders (no weighted sampling - use class weights in loss instead)
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_params['batch_size'],
        shuffle=True,
        num_workers=model_params.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_params['batch_size'],
        shuffle=False,
        num_workers=model_params.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    final_model = SequenceCNN(seq_length=seq_length, dropout=model_params['dropout']).to(device)
    
    # Calculate class weights for final model
    class_counts = np.bincount(y_train, minlength=2)  # Ensure we have weights for both classes
    total_samples = len(y_train)
    
    # Handle case where a class is missing (count=0)
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(total_samples / (len(class_counts) * count))
        else:
            class_weights.append(0.0)  # No weight for missing class
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Final model class distribution: {class_counts}")
    print(f"Final model class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(final_model.parameters(), 
                          lr=model_params['learning_rate'], 
                          weight_decay=model_params['weight_decay'])
    
    # Training loop
    for epoch in range(model_params['epochs']):
        train_loss, train_acc = train_epoch(final_model, train_loader, criterion, optimizer, device)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    
    # Test evaluation
    test_preds, test_probs, test_labels = evaluate(final_model, test_loader, device)
    
    test_results = {
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(test_labels, test_preds),
        'auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5,
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
        'classification_report': classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
    }
    
    # Save results
    results = {
        'cv_results': cv_results,
        'test_results': test_results,
        'cv_mean_f1': np.mean([r['f1'] for r in cv_results]),
        'cv_std_f1': np.std([r['f1'] for r in cv_results]),
        'cv_mean_auc': np.mean([r['auc'] for r in cv_results]),
        'cv_std_auc': np.std([r['auc'] for r in cv_results]),
        'model_params': model_params,
        'sequence_length': seq_length,
        'n_train_sequences': X_train.shape[0],
        'n_test_sequences': X_test.shape[0],
        'architecture': 'Raw sequence CNN with one-hot encoding'
    }
    
    # Ensure output directory exists
    output_dir = Path(snakemake.output.results).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(snakemake.output.results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {snakemake.output.results}")
    except Exception as e:
        print(f"Error saving results: {e}")
        raise
    
    # Save model
    try:
        model_dir = Path(snakemake.output.model).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), snakemake.output.model)
        print(f"Model saved to: {snakemake.output.model}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise
    
    # Create simple feature importance plot (conv layer weights)
    try:
        plt.figure(figsize=(12, 8))
        
        # CV results plot
        plt.subplot(2, 2, 1)
        folds = [r['fold'] + 1 for r in cv_results]
        f1_scores = [r['f1'] for r in cv_results]
        plt.bar(folds, f1_scores)
        plt.title('Cross-Validation F1 Scores')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.ylim([0, 1])
        
        # AUC scores
        auc_scores = [r['auc'] for r in cv_results]
        plt.subplot(2, 2, 2)
        plt.bar(folds, auc_scores)
        plt.title('Cross-Validation AUC Scores')
        plt.xlabel('Fold')
        plt.ylabel('AUC')
        plt.ylim([0, 1])
        
        # Test confusion matrix
        plt.subplot(2, 2, 3)
        cm = np.array(test_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Architecture summary
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'Architecture: Raw Sequence CNN', fontsize=12, weight='bold')
        plt.text(0.1, 0.7, f'Sequence Length: {seq_length}', fontsize=10)
        plt.text(0.1, 0.6, f'Input: One-hot ACGT (4 channels)', fontsize=10)
        plt.text(0.1, 0.5, f'Conv Layers: 4 (32→64→128→256)', fontsize=10)
        plt.text(0.1, 0.4, f'Kernel Sizes: 15→11→7→5', fontsize=10)
        plt.text(0.1, 0.3, f'Training Seqs: {X_train.shape[0]}', fontsize=10)
        plt.text(0.1, 0.2, f'Test F1: {test_results["f1"]:.3f}', fontsize=10)
        plt.text(0.1, 0.1, f'CV F1: {results["cv_mean_f1"]:.3f} ± {results["cv_std_f1"]:.3f}', fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Ensure plot directory exists
        plot_dir = Path(snakemake.output.plots).parent
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(snakemake.output.plots, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to: {snakemake.output.plots}")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        # Create an empty file to satisfy Snakemake output requirements
        Path(snakemake.output.plots).touch()
    
    print(f"\n=== RAW SEQUENCE CNN RESULTS SUMMARY ===")
    print(f"CV F1: {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"CV AUC: {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")

if __name__ == "__main__":
    main()