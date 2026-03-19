#!/usr/bin/env python3
"""
Raw Sequence CNN Training for AMR Prediction.
PyTorch implementation using one-hot encoded ACGT sequences.

Fixes applied:
- sys.path.insert for utils imports when run via Snakemake temp script
- train_final_model uses 10% early-stopping holdout to prevent overfitting
- num_workers=0 on MPS (macOS) for DataLoader stability
- FASTQ existence check with clear error message if files were deleted by cleanup rule
- cv_folds/random_state pulled from smk params (fixed in smk file)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gzip
import random
from collections import Counter

warnings.filterwarnings('ignore')

try:
    from Bio import SeqIO
except ImportError:
    print("Error: BioPython not available. Install with: conda install -c bioconda biopython")
    sys.exit(1)

from utils.dl_training import configure_pytorch_threads, get_device
from utils.cross_validation import GeographicTemporalKFold, get_cross_validator


# =============================================================================
# Dataset
# =============================================================================

class SequenceDataset(Dataset):
    """PyTorch dataset for one-hot encoded DNA sequences."""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================================================
# Model
# =============================================================================

class SequenceCNN(nn.Module):
    """CNN for raw DNA sequence classification using one-hot encoding."""

    def __init__(self, seq_length, num_classes=2, dropout=0.5):
        super(SequenceCNN, self).__init__()
        self.seq_length = seq_length

        self.conv1 = nn.Conv1d(4, 32,  kernel_size=15, padding=7)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4   = nn.BatchNorm1d(256)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.feature_maps = None

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))

        self.feature_maps = x.detach()

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# =============================================================================
# Sequence processing
# =============================================================================

def one_hot_encode_sequence(sequence, max_length):
    """Convert DNA sequence to one-hot encoding. Shape: (4, max_length)"""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, max_length), dtype=np.float32)
    for i, base in enumerate(sequence[:max_length]):
        if base in base_to_idx:
            encoded[base_to_idx[base], i] = 1.0
    return encoded


def extract_sequences_from_sample(sample_id, processed_dir,
                                   n_reads=1000, min_length=150):
    """Extract quality-filtered sequences from FASTQ files."""
    r1_path = processed_dir / f"{sample_id}_1.fastq.gz"
    r2_path = processed_dir / f"{sample_id}_2.fastq.gz"

    sequences = []
    total_processed = 0

    for path in [r1_path, r2_path]:
        if not path.exists() or total_processed >= n_reads:
            continue
        try:
            with gzip.open(path, 'rt') as f:
                for record in SeqIO.parse(f, 'fastq'):
                    if total_processed >= n_reads:
                        break
                    seq = str(record.seq).upper()
                    if (len(seq) >= min_length
                            and 'N' not in seq
                            and len(set(seq) & {'A', 'C', 'G', 'T'}) >= 3):
                        sequences.append(seq)
                        total_processed += 1
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")

    return sequences


def prepare_sequence_dataset(train_samples, test_samples, processed_dir,
                              n_reads_per_sample=1000, max_seq_length=200):
    """Prepare one-hot encoded sequence dataset from FASTQ files."""
    print("Extracting sequences from samples...")

    all_sequences = []
    all_labels    = []
    sample_ids    = []

    # Check at least a few files exist before committing to full extraction
    n_found = sum(
        1 for sid in list(train_samples['sample_id'])[:5]
        if (processed_dir / f"{sid}_1.fastq.gz").exists()
    )
    if n_found == 0:
        raise FileNotFoundError(
            f"No processed FASTQ files found in {processed_dir}.\n"
            f"Script 17 reads from data/processed/ which may have been deleted "
            f"by the terminal cleanup rule. Either:\n"
            f"  1. Keep processed files until step 17 completes (move cleanup rule)\n"
            f"  2. Re-run step 03 to regenerate processed files\n"
            f"  3. Use assemblies (data/assemblies/) as input instead of raw reads"
        )

    for df, desc in [(train_samples, "train"), (test_samples, "test")]:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {desc} samples",
                           total=len(df)):
            sample_id = row['sample_id']
            label     = int(row['R'])

            seqs = extract_sequences_from_sample(
                sample_id, processed_dir, n_reads_per_sample
            )

            if seqs:
                selected = seqs[:min(30, len(seqs))]
                for seq_idx, seq in enumerate(selected):
                    all_sequences.append(seq)
                    all_labels.append(label)
                    sample_ids.append(f"{sample_id}_{seq_idx + 1}")

    print(f"Extracted {len(all_sequences)} sequences total")

    if len(all_sequences) == 0:
        raise ValueError(
            f"No sequences extracted. Check FASTQ files in {processed_dir}"
        )

    print(f"Class distribution: {Counter(all_labels)}")

    seq_lengths    = [len(s) for s in all_sequences]
    actual_max_len = min(max_seq_length, int(np.percentile(seq_lengths, 95)))
    print(f"Using sequence length: {actual_max_len}")

    print("One-hot encoding sequences...")
    encoded = np.array([
        one_hot_encode_sequence(s, actual_max_len)
        for s in tqdm(all_sequences, desc="Encoding")
    ])

    return encoded, np.array(all_labels), sample_ids, actual_max_len


# =============================================================================
# Training helpers
# =============================================================================

def make_dataloader(X, y, batch_size, shuffle, num_workers):
    dataset = SequenceDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )


def compute_class_weights(y, device):
    class_counts = np.bincount(y, minlength=2)
    total = len(y)
    weights = torch.FloatTensor([
        total / (len(class_counts) * c) if c > 0 else 0.0
        for c in class_counts
    ]).to(device)
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs   = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def run_training_with_early_stopping(model, train_loader, val_loader,
                                     criterion, optimizer, device,
                                     epochs, patience):
    """Train with early stopping on validation AUC."""
    best_val_auc     = 0.0
    best_state       = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_preds, val_probs, val_labels = evaluate(model, val_loader, device)

        val_auc = (roc_auc_score(val_labels, val_probs)
                   if len(np.unique(val_labels)) > 1 else 0.5)
        val_f1  = f1_score(val_labels, val_preds, zero_division=0)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={train_loss:.4f} | "
                  f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best auc={best_val_auc:.4f} at epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_auc


# =============================================================================
# Cross-validation (sample-level split to prevent sequence leakage)
# =============================================================================

def cross_validation(X, y, sample_ids, location_year_groups=None,
                     cv_folds=5, random_state=42, num_workers=0,
                     **model_params):
    """
    Cross-validation split at sample level (not sequence level) to prevent
    data leakage from multiple sequences per sample.
    """
    device     = get_device()
    batch_size = model_params.get('batch_size', 16)
    epochs     = model_params.get('epochs', 50)
    patience   = model_params.get('patience', 15)
    dropout    = model_params.get('dropout', 0.5)
    lr         = model_params.get('learning_rate', 0.0001)
    wd         = model_params.get('weight_decay', 1e-3)

    print(f"DataLoader workers: {num_workers}")

    # Build sample-level index (group sequences from same sample)
    original_samples = {}
    for i, sid in enumerate(sample_ids):
        orig_id = sid.split('_')[0]
        original_samples.setdefault(orig_id, []).append(i)

    sample_names  = list(original_samples.keys())
    sample_labels = np.array([
        Counter(y[original_samples[n]]).most_common(1)[0][0]
        for n in sample_names
    ])

    if len(sample_labels) < cv_folds:
        raise ValueError(
            f"Only {len(sample_labels)} samples for {cv_folds}-fold CV"
        )

    # Map location-year to sample level
    sample_loc_year = None
    if location_year_groups is not None:
        sample_cluster = {}
        for i, sid in enumerate(sample_ids):
            orig = sid.split('_')[0]
            if orig not in sample_cluster:
                sample_cluster[orig] = location_year_groups[i]
        sample_loc_year = np.array([
            sample_cluster.get(n, 'unknown') for n in sample_names
        ])

    if sample_loc_year is not None and len(np.unique(sample_loc_year)) >= cv_folds:
        try:
            cv_splitter = GeographicTemporalKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            print("Using geographic-temporal cross-validation")
        except ValueError as e:
            print(f"Warning: {e} — falling back to stratified CV")
            cv_splitter     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            sample_loc_year = None
    else:
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        print("Using stratified cross-validation")

    cv_results  = []
    fold_models = []

    for fold, (train_s_idx, val_s_idx) in enumerate(
        cv_splitter.split(sample_names, sample_labels, groups=sample_loc_year)
    ):
        print(f"\nFold {fold + 1}/{cv_folds}")

        train_samples_fold = [sample_names[i] for i in train_s_idx]
        val_samples_fold   = [sample_names[i] for i in val_s_idx]

        train_seq_idx = [
            idx for name in train_samples_fold
            for idx in original_samples[name]
        ]
        val_seq_idx = [
            idx for name in val_samples_fold
            for idx in original_samples[name]
        ]

        X_tr, y_tr = X[train_seq_idx], y[train_seq_idx]
        X_val, y_val = X[val_seq_idx], y[val_seq_idx]

        print(f"  Train seqs={len(X_tr)} | Val seqs={len(X_val)}")
        print(f"  Train dist: {Counter(y_tr)} | Val dist: {Counter(y_val)}")

        if len(np.unique(y_tr)) < 2:
            print(f"  Fold {fold+1} single class — skipping")
            cv_results.append({
                'fold': fold, 'f1': 0.0,
                'balanced_accuracy': 0.5, 'auc': 0.5,
                'confusion_matrix': [[0,0],[0,0]]
            })
            fold_models.append({})
            continue

        train_loader = make_dataloader(X_tr,  y_tr,  batch_size, True,  num_workers)
        val_loader   = make_dataloader(X_val, y_val, batch_size, False, num_workers)

        model = SequenceCNN(seq_length=X.shape[2], dropout=dropout).to(device)

        class_weights = compute_class_weights(y_tr, device)
        print(f"  Class weights: {class_weights.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        model, best_epoch, best_auc = run_training_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs, patience
        )

        fold_models.append({k: v.cpu() for k, v in model.state_dict().items()})

        val_preds, val_probs, val_labels = evaluate(model, val_loader, device)

        fold_results = {
            'fold':              fold,
            'best_epoch':        best_epoch,
            'f1':                float(f1_score(val_labels, val_preds, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(val_labels, val_preds)),
            'auc':               float(roc_auc_score(val_labels, val_probs)
                                       if len(np.unique(val_labels)) > 1 else 0.5),
            'confusion_matrix':  confusion_matrix(val_labels, val_preds).tolist()
        }

        cv_results.append(fold_results)
        print(f"  → F1={fold_results['f1']:.4f} | "
              f"Bal.Acc={fold_results['balanced_accuracy']:.4f} | "
              f"AUC={fold_results['auc']:.4f} | "
              f"Best epoch={best_epoch}")

    return cv_results, fold_models


# =============================================================================
# Final model — 10% early-stopping holdout prevents overfitting
# =============================================================================

def train_final_model(X_train, y_train, X_test, y_test,
                      seq_length, num_workers=0, **model_params):
    """
    Train final model with a 10% early-stopping holdout.
    Prevents overfitting seen when training for full epochs with no stopping.
    """
    device     = get_device()
    batch_size = model_params.get('batch_size', 16)
    epochs     = model_params.get('epochs', 50)
    patience   = model_params.get('patience', 15)
    dropout    = model_params.get('dropout', 0.5)
    lr         = model_params.get('learning_rate', 0.0001)
    wd         = model_params.get('weight_decay', 1e-3)

    X_ft, X_es, y_ft, y_es = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\nFinal model — fine-tune: {len(X_ft)} | "
          f"ES holdout: {len(X_es)} | Test: {len(X_test)}")
    print(f"Fine-tune class dist: {np.bincount(y_ft)}")

    train_loader = make_dataloader(X_ft,   y_ft,   batch_size, True,  num_workers)
    es_loader    = make_dataloader(X_es,   y_es,   batch_size, False, num_workers)
    test_loader  = make_dataloader(X_test, y_test, batch_size, False, num_workers)

    model = SequenceCNN(seq_length=seq_length, dropout=dropout).to(device)

    class_weights = compute_class_weights(y_ft, device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model, best_epoch, best_es_auc = run_training_with_early_stopping(
        model, train_loader, es_loader, criterion, optimizer,
        device, epochs, patience
    )
    print(f"Final model stopped at epoch {best_epoch} "
          f"(ES holdout AUC={best_es_auc:.4f})")

    test_preds, test_probs, test_labels = evaluate(model, test_loader, device)

    test_results = {
        'f1':                float(f1_score(test_labels, test_preds, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(test_labels, test_preds)),
        'auc':               float(roc_auc_score(test_labels, test_probs)
                                   if len(np.unique(test_labels)) > 1 else 0.5),
        'confusion_matrix':  confusion_matrix(test_labels, test_preds).tolist(),
        'classification_report': classification_report(
            test_labels, test_preds, output_dict=True, zero_division=0
        ),
        'best_epoch':     best_epoch,
        'es_holdout_auc': float(best_es_auc)
    }

    return model, test_results, test_preds, test_probs, test_labels


# =============================================================================
# Main
# =============================================================================

def main():
    num_workers = configure_pytorch_threads()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        num_workers = 0
        print("MPS device detected: forcing num_workers=0")

    # ------------------------------------------------------------------
    # Load sample lists from tree model CSVs
    # ------------------------------------------------------------------
    train_df = pd.read_csv(snakemake.input.train)
    test_df  = pd.read_csv(snakemake.input.test)

    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in str(snakemake.output.model):
            antibiotic = ab
            break
    if antibiotic is None:
        raise ValueError("Could not determine antibiotic from output path")

    print(f"=== RAW SEQUENCE CNN: {antibiotic.upper()} ===")

    real_train = train_df[~train_df['sample_id'].str.contains('SMOTE_synthetic', na=False)]
    real_test  = test_df

    # ------------------------------------------------------------------
    # Load SRR IDs from metadata
    # ------------------------------------------------------------------
    project_root = Path(snakemake.scriptdir).parent
    train_meta   = pd.read_csv(project_root / "results/features/metadata_train_processed.csv")
    test_meta    = pd.read_csv(project_root / "results/features/metadata_test_processed.csv")

    train_srr_ids = train_meta['Run'].tolist()
    test_srr_ids  = test_meta['Run'].tolist()

    overlap = set(train_srr_ids) & set(test_srr_ids)
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} samples in both train and test")

    train_label_map = {row['sample_id']: int(row['R'])
                       for _, row in real_train.iterrows()}
    test_label_map  = {row['sample_id']: int(row['R'])
                       for _, row in real_test.iterrows()}

    train_df_seq = pd.DataFrame({
        'sample_id': train_srr_ids,
        'R': [train_label_map.get(s, 0) for s in train_srr_ids]
    })
    test_df_seq = pd.DataFrame({
        'sample_id': test_srr_ids,
        'R': [test_label_map.get(s, 0) for s in test_srr_ids]
    })

    # ------------------------------------------------------------------
    # Extract sequences from FASTQ files
    # ------------------------------------------------------------------
    processed_dir    = Path(snakemake.params.processed_dir)
    n_reads          = getattr(snakemake.params, 'n_reads_per_sample', 1000)
    max_seq_length   = getattr(snakemake.params, 'max_seq_length', 200)

    X, y, sample_ids, seq_length = prepare_sequence_dataset(
        train_df_seq, test_df_seq, processed_dir,
        n_reads_per_sample=n_reads,
        max_seq_length=max_seq_length
    )

    # Split into train / test by SRR ID
    train_set  = set(train_srr_ids)
    test_set   = set(test_srr_ids)

    train_idx  = [i for i, sid in enumerate(sample_ids) if sid.split('_')[0] in train_set]
    test_idx   = [i for i, sid in enumerate(sample_ids) if sid.split('_')[0] in test_set]

    if not train_idx:
        raise ValueError("No training sequences found after split")
    if not test_idx:
        raise ValueError("No test sequences found after split")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test  = X[test_idx]
    y_test  = y[test_idx]
    train_sample_ids = [sample_ids[i] for i in train_idx]

    print(f"Training sequences: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Sequence length:    {seq_length}")
    print(f"Train class dist:   {Counter(y_train)}")
    print(f"Test class dist:    {Counter(y_test)}")

    # ------------------------------------------------------------------
    # Geographic-temporal groups
    # ------------------------------------------------------------------
    location_year_train = None
    try:
        meta_path = str(snakemake.input.train).replace(
            f'tree_models/{antibiotic}_train_final.csv',
            f'tree_models/{antibiotic}_train_final.csv'
        )
        meta = pd.read_csv(meta_path)
        meta['loc_year'] = (
            meta['Location'].fillna('unknown').astype(str) + '_' +
            meta['Year'].fillna('unknown').astype(str)
        )
        sid_to_group = dict(zip(meta['sample_id'], meta['loc_year']))
        location_year_train = np.array([
            sid_to_group.get(sid.split('_')[0], 'unknown')
            for sid in train_sample_ids
        ])
        print(f"Location-year groups: {len(np.unique(location_year_train))} unique")
    except Exception as e:
        print(f"Warning: Could not load location-year info: {e}")

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    model_params = {
        'epochs':        getattr(snakemake.params, 'epochs', 50),
        'batch_size':    getattr(snakemake.params, 'batch_size', 16),
        'learning_rate': getattr(snakemake.params, 'learning_rate', 0.0001),
        'dropout':       getattr(snakemake.params, 'dropout', 0.5),
        'weight_decay':  getattr(snakemake.params, 'weight_decay', 1e-3),
        'patience':      getattr(snakemake.params, 'patience', 15),
        'num_workers':   num_workers
    }

    cv_folds     = snakemake.params.get('cv_folds', 5)
    random_state = snakemake.params.get('random_state', 42)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    print(f"\nRunning {cv_folds}-fold cross-validation...")
    cv_results, fold_models = cross_validation(
        X_train, y_train, train_sample_ids,
        location_year_train, cv_folds, random_state,
        **model_params
    )

    # ------------------------------------------------------------------
    # Final model with early-stopping holdout
    # ------------------------------------------------------------------
    print("\nTraining final model (with early-stopping holdout)...")
    final_model, test_results, test_preds, test_probs, test_labels = train_final_model(
        X_train, y_train, X_test, y_test,
        seq_length, num_workers, **model_params
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    results = {
        'cv_results':        cv_results,
        'test_results':      test_results,
        'cv_mean_f1':        float(np.mean([r['f1']  for r in cv_results])),
        'cv_std_f1':         float(np.std( [r['f1']  for r in cv_results])),
        'cv_mean_auc':       float(np.mean([r['auc'] for r in cv_results])),
        'cv_std_auc':        float(np.std( [r['auc'] for r in cv_results])),
        'model_params':      model_params,
        'sequence_length':   seq_length,
        'n_train_sequences': int(X_train.shape[0]),
        'n_test_sequences':  int(X_test.shape[0]),
        'architecture':      'Raw sequence CNN with one-hot encoding',
        'test_predictions': {
            'y_true':  test_labels.tolist(),
            'y_pred':  test_preds.tolist(),
            'y_proba': test_probs.tolist()
        }
    }

    Path(snakemake.output.results).parent.mkdir(parents=True, exist_ok=True)
    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {snakemake.output.results}")

    Path(snakemake.output.model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), snakemake.output.model)
    print(f"Model saved to: {snakemake.output.model}")

    # Plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        folds = [r['fold'] + 1 for r in cv_results]

        f1_scores = [r['f1'] for r in cv_results]
        axes[0, 0].bar(folds, f1_scores, color='steelblue')
        axes[0, 0].axhline(np.mean(f1_scores), color='red', linestyle='--',
                            label=f'Mean={np.mean(f1_scores):.3f}')
        axes[0, 0].set_title('CV F1 Scores')
        axes[0, 0].set_xlabel('Fold'); axes[0, 0].set_ylabel('F1')
        axes[0, 0].set_ylim(0, 1); axes[0, 0].legend()

        auc_scores = [r['auc'] for r in cv_results]
        axes[0, 1].bar(folds, auc_scores, color='darkorange')
        axes[0, 1].axhline(np.mean(auc_scores), color='red', linestyle='--',
                            label=f'Mean={np.mean(auc_scores):.3f}')
        axes[0, 1].set_title('CV AUC Scores')
        axes[0, 1].set_xlabel('Fold'); axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_ylim(0, 1); axes[0, 1].legend()

        cm = np.array(test_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues',
                    xticklabels=['Susceptible', 'Resistant'],
                    yticklabels=['Susceptible', 'Resistant'])
        axes[1, 0].set_title(
            f"Test Confusion Matrix\n"
            f"F1={test_results['f1']:.3f} | AUC={test_results['auc']:.3f}"
        )

        axes[1, 1].axis('off')
        info_lines = [
            f'Architecture: Sequence CNN (one-hot ACGT)',
            f'Sequence length: {seq_length}',
            f'Conv layers: 4 (32→64→128→256)',
            f'Training sequences: {X_train.shape[0]}',
            f'Test F1: {test_results["f1"]:.3f}',
            f'CV F1: {results["cv_mean_f1"]:.3f} ± {results["cv_std_f1"]:.3f}',
            f'Best epoch: {test_results.get("best_epoch", "N/A")}',
        ]
        for i, line in enumerate(info_lines):
            axes[1, 1].text(0.05, 0.9 - i * 0.12, line, fontsize=10,
                            transform=axes[1, 1].transAxes)

        plt.tight_layout()
        Path(snakemake.output.plots).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(snakemake.output.plots, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to: {snakemake.output.plots}")

    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        Path(snakemake.output.plots).touch()

    print(f"\n=== RESULTS SUMMARY ===")
    print(f"CV F1:    {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"CV AUC:   {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    print(f"Test F1:  {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")
    print(f"Best epoch: {test_results.get('best_epoch', 'N/A')}")


if __name__ == "__main__":
    main()
