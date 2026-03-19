#!/usr/bin/env python3
"""
DNABERT-2 Fine-tuning for AMR Prediction.

Official loading pattern (from DNABERT-2 docs):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    inputs = tokenizer(dna, return_tensors='pt')["input_ids"]
    hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

Fixes applied:
- SimpleDNATokenizer at module level (DataLoader worker pickling)
- AutoModel.from_pretrained with only trust_remote_code=True (no config= arg)
- get_attention_weights handles DNABERT-2 tuple output (no .attentions attribute)
- train_final_model uses 10% early-stopping holdout to prevent overfitting
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
from transformers import AutoTokenizer, AutoModel
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

from utils.dl_training import configure_pytorch_threads, get_device
from utils.cross_validation import GeographicTemporalKFold


# =============================================================================
# Fallback tokenizer — MUST be at module level for DataLoader worker pickling
# =============================================================================

class SimpleDNATokenizer:
    """
    Character-level DNA tokenizer. Fallback when official DNABERT-2 tokenizer
    cannot be loaded. Must be at module level so multiprocessing DataLoader
    workers can pickle it.
    """
    def __init__(self):
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            'A': 4, 'T': 5, 'G': 6, 'C': 7, 'N': 8
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0

    def __call__(self, sequence, max_length=512, truncation=True,
                 padding='max_length', return_tensors=None):
        tokens = ['[CLS]'] + list(sequence.upper()[:max_length - 2]) + ['[SEP]']
        input_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.vocab['[PAD]'])
            attention_mask.append(0)
        result = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result

    def convert_ids_to_tokens(self, ids):
        return [self.inverse_vocab.get(int(i), '[UNK]') for i in ids]


# =============================================================================
# Dataset
# =============================================================================

class DNASequenceDataset(Dataset):
    """
    PyTorch dataset for DNA sequences.
    Compatible with both the official DNABERT-2 tokenizer and SimpleDNATokenizer.
    Generates attention_mask if the tokenizer does not return one.
    """
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sequence,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        input_ids = encoding['input_ids'].squeeze(0)

        if 'attention_mask' in encoding:
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
            attention_mask = (input_ids != pad_id).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }


# =============================================================================
# Model
# =============================================================================

class DNABERT2Classifier(nn.Module):
    """
    DNABERT-2 fine-tuning classifier for AMR resistance prediction.

    Correct loading (official pattern — no config= argument):
        AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    Falls back to a small randomly-initialized BERT if loading fails.
    Classification head: mean-pooled hidden states → dropout → linear.
    """

    MODEL_NAME = "zhihan1996/DNABERT-2-117M"

    def __init__(self, num_classes=2, classifier_dropout=0.1):
        super().__init__()
        self.is_pretrained = False

        try:
            print(f"  Loading pretrained encoder: {self.MODEL_NAME}")
            self.encoder = AutoModel.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            self.is_pretrained = True
            print(f"  Successfully loaded pretrained {self.MODEL_NAME}")
            print(f"  Hidden size: {self.encoder.config.hidden_size} | "
                  f"Layers: {self.encoder.config.num_hidden_layers}")

        except Exception as e:
            print(f"  WARNING: Could not load pretrained model: {e}")
            print("  Falling back to randomly-initialized small BERT.")
            print("  NOTE: Results reflect training from scratch, NOT transfer learning.")
            from transformers import BertModel, BertConfig
            config = BertConfig(
                vocab_size=9,
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=1024,
                max_position_embeddings=512,
                hidden_dropout_prob=classifier_dropout,
                attention_probs_dropout_prob=classifier_dropout,
                type_vocab_size=2,
                layer_norm_eps=1e-12
            )
            self.encoder = BertModel(config)
            print(f"  From-scratch BERT: vocab={config.vocab_size}, "
                  f"hidden={config.hidden_size}")

        self.hidden_size = self.encoder.config.hidden_size
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass following the official DNABERT-2 embedding pattern:
            hidden_states = model(input_ids)[0]  # [batch, seq_len, hidden]
        We use attention-masked mean pooling to exclude padding positions.
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [batch, seq_len, hidden_size]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled = torch.sum(hidden_states * mask, dim=1) / \
                     torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def get_attention_weights(self, input_ids, attention_mask=None):
        """
        Extract averaged attention weights for interpretability.

        DNABERT-2 returns a plain tuple, not a ModelOutput, so .attentions
        does not exist. With output_attentions=True the tuple structure is:
            (last_hidden_state, pooler_output, *attentions_per_layer)
        Attention layers start at index 2.
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

            # Handle both ModelOutput (standard HF) and plain tuple (DNABERT-2)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions
            elif isinstance(outputs, tuple) and len(outputs) > 2:
                # DNABERT-2 custom tuple: (hidden, pooler, attn_layer_0, attn_layer_1, ...)
                attentions = outputs[2:]
                # Filter to only tensor elements (attention weight arrays)
                attentions = [a for a in attentions if isinstance(a, torch.Tensor)]
            else:
                # Fallback: return uniform attention over non-padded positions
                seq_len = input_ids.shape[1]
                uniform = torch.ones(
                    input_ids.shape[0], 1, seq_len,
                    device=input_ids.device
                )
                if attention_mask is not None:
                    uniform = uniform * attention_mask.unsqueeze(1).float()
                return uniform

            if not attentions:
                seq_len = input_ids.shape[1]
                uniform = torch.ones(
                    input_ids.shape[0], 1, seq_len,
                    device=input_ids.device
                )
                if attention_mask is not None:
                    uniform = uniform * attention_mask.unsqueeze(1).float()
                return uniform

            # Average over all layers [num_layers, batch, heads, seq, seq]
            # then average over heads → [batch, seq, seq]
            avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1)
            # Average over query positions → [batch, seq]
            attention_scores = avg_attention.mean(dim=1)
            if attention_mask is not None:
                attention_scores = attention_scores * attention_mask.float()

            return attention_scores.unsqueeze(1)


# =============================================================================
# Training helpers
# =============================================================================

def make_dataloader(sequences, labels, tokenizer, batch_size, shuffle, num_workers):
    dataset = DNASequenceDataset(sequences, labels, tokenizer, max_length=512)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )


def compute_class_weights(labels, device):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = torch.FloatTensor([
        total / (len(class_counts) * c) for c in class_counts
    ]).to(device)
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def extract_attention_patterns(model, dataloader, device, tokenizer, max_samples=100):
    model.eval()
    attention_data = []
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            attn_weights = model.get_attention_weights(input_ids, attention_mask)

            for i in range(input_ids.size(0)):
                if sample_count >= max_samples:
                    break
                sample_ids  = input_ids[i].cpu().numpy()
                sample_attn = attn_weights[i].cpu().numpy()
                sample_mask = attention_mask[i].cpu().numpy()
                label       = labels[i].item()

                valid_ids = sample_ids[sample_mask.astype(bool)]
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    tokens = tokenizer.convert_ids_to_tokens(valid_ids)
                else:
                    tokens = [str(id_) for id_ in valid_ids]

                attention_data.append({
                    'tokens':           tokens,
                    'attention_scores': sample_attn[0][:len(tokens)].tolist(),
                    'label':            label,
                    'sample_index':     sample_count
                })
                sample_count += 1

    return attention_data


def run_training_with_early_stopping(model, train_loader, val_loader,
                                     criterion, optimizer, device,
                                     epochs, patience):
    """
    Train with early stopping based on validation AUC.
    Returns the best model state and the epoch it was achieved.
    """
    best_val_auc     = 0.0
    best_state       = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_preds, val_probs, val_true = evaluate(model, val_loader, device)

        val_f1  = f1_score(val_true, val_preds, zero_division=0)
        val_auc = (roc_auc_score(val_true, val_probs)
                   if len(np.unique(val_true)) > 1 else 0.5)

        print(f"  Epoch {epoch+1}: loss={train_loss:.4f} | acc={train_acc:.4f} | "
              f"val_f1={val_f1:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best auc={best_val_auc:.4f} at epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_auc


# =============================================================================
# Cross-validation
# =============================================================================

def cross_validation(sequences, labels, tokenizer,
                     location_year_groups=None, cv_folds=5, random_state=42,
                     **model_params):
    device      = get_device()
    batch_size  = model_params.get('batch_size', 16)
    num_workers = model_params.get('num_workers', 0)
    dropout     = model_params.get('dropout', 0.1)
    epochs      = model_params.get('epochs', 7)
    patience    = model_params.get('patience', 5)

    if location_year_groups is not None:
        try:
            cv_splitter = GeographicTemporalKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            print("Using geographic-temporal-aware cross-validation")
        except ValueError as e:
            print(f"Warning: {e} — falling back to stratified CV")
            cv_splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            location_year_groups = None
    else:
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        print("Using stratified cross-validation")

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv_splitter.split(sequences, labels, groups=location_year_groups)
    ):
        print(f"\nFold {fold + 1}/{cv_folds}")
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs   = [sequences[i] for i in val_idx]
        train_lbls = labels[train_idx]
        val_lbls   = labels[val_idx]

        print(f"  Train={len(train_seqs)} | Val={len(val_seqs)}")
        print(f"  Train dist: {Counter(train_lbls)} | Val dist: {Counter(val_lbls)}")

        train_loader = make_dataloader(
            train_seqs, train_lbls, tokenizer, batch_size, True, num_workers
        )
        val_loader = make_dataloader(
            val_seqs, val_lbls, tokenizer, batch_size, False, num_workers
        )

        print(f"Setting up DNABERT-2 model...")
        model = DNABERT2Classifier(
            num_classes=2, classifier_dropout=dropout
        ).to(device)

        class_weights = compute_class_weights(train_lbls, device)
        print(f"  Class weights: {class_weights.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_params.get('learning_rate', 2e-5),
            weight_decay=model_params.get('weight_decay', 0.01)
        )

        model, best_epoch, best_auc = run_training_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs, patience
        )

        val_preds, val_probs, val_true = evaluate(model, val_loader, device)

        fold_result = {
            'fold':              fold,
            'best_epoch':        best_epoch,
            'best_val_auc':      float(best_auc),
            'f1':                float(f1_score(val_true, val_preds, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(val_true, val_preds)),
            'auc':               float(roc_auc_score(val_true, val_probs)
                                       if len(np.unique(val_true)) > 1 else 0.5),
            'confusion_matrix':  confusion_matrix(val_true, val_preds).tolist()
        }
        cv_results.append(fold_result)
        print(f"  → F1={fold_result['f1']:.4f} | "
              f"Bal.Acc={fold_result['balanced_accuracy']:.4f} | "
              f"AUC={fold_result['auc']:.4f} | "
              f"Best epoch={best_epoch}")

    return cv_results


# =============================================================================
# Final model — uses 10% early-stopping holdout to prevent overfitting
# =============================================================================

def train_final_model(train_sequences, train_labels, test_sequences, test_labels,
                      tokenizer, **model_params):
    """
    Train final model on training data with a 10% early-stopping holdout.
    The holdout is used only for stopping — not for evaluation.
    Test set is evaluated separately and reported as final performance.

    This prevents the severe overfitting seen when training for 20 epochs
    on 100% of training data with no stopping signal.
    """
    device      = get_device()
    batch_size  = model_params.get('batch_size', 16)
    num_workers = model_params.get('num_workers', 0)
    dropout     = model_params.get('dropout', 0.1)
    epochs      = model_params.get('epochs', 7)
    patience    = model_params.get('patience', 5)

    # Split 10% of training data for early stopping only
    (train_seqs_ft, es_seqs,
     train_lbls_ft, es_lbls) = train_test_split(
        train_sequences, train_labels,
        test_size=0.1, random_state=42, stratify=train_labels
    )

    print(f"\nFinal model training:")
    print(f"  Fine-tune set: {len(train_seqs_ft)} | "
          f"Early-stop holdout: {len(es_seqs)} | "
          f"Test: {len(test_sequences)}")
    print(f"  Fine-tune class dist: {np.bincount(train_lbls_ft)}")

    train_loader = make_dataloader(
        train_seqs_ft, train_lbls_ft, tokenizer, batch_size, True, num_workers
    )
    es_loader = make_dataloader(
        es_seqs, es_lbls, tokenizer, batch_size, False, num_workers
    )
    test_loader = make_dataloader(
        test_sequences, test_labels, tokenizer, batch_size, False, num_workers
    )

    print(f"Setting up DNABERT-2 model (final)...")
    model = DNABERT2Classifier(
        num_classes=2, classifier_dropout=dropout
    ).to(device)

    class_weights = compute_class_weights(train_lbls_ft, device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_params.get('learning_rate', 2e-5),
        weight_decay=model_params.get('weight_decay', 0.01)
    )

    model, best_epoch, best_es_auc = run_training_with_early_stopping(
        model, train_loader, es_loader, criterion, optimizer,
        device, epochs, patience
    )

    print(f"  Final model stopped at epoch {best_epoch} "
          f"(early-stop holdout AUC={best_es_auc:.4f})")

    # Evaluate on actual test set
    test_preds, test_probs, test_true = evaluate(model, test_loader, device)

    test_results = {
        'f1':                float(f1_score(test_true, test_preds, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(test_true, test_preds)),
        'auc':               float(roc_auc_score(test_true, test_probs)
                                   if len(np.unique(test_true)) > 1 else 0.5),
        'confusion_matrix':  confusion_matrix(test_true, test_preds).tolist(),
        'classification_report': classification_report(
            test_true, test_preds, output_dict=True, zero_division=0
        ),
        'best_epoch':        best_epoch,
        'early_stop_auc':    float(best_es_auc),
        '_y_true':  test_true.tolist(),
        '_y_pred':  test_preds.tolist(),
        '_y_proba': test_probs.tolist()
    }

    print("Extracting attention patterns...")
    attention_data = extract_attention_patterns(
        model, test_loader, device, tokenizer, max_samples=100
    )

    return model, test_results, attention_data


# =============================================================================
# Sequence reconstruction from k-mer tokens
# =============================================================================

def reconstruct_sequences_from_tokens(input_ids, attention_mask, old_tokenizer):
    """Reconstruct raw DNA strings from the step-13 k-mer token ID arrays."""
    sequences   = []
    id_to_token = old_tokenizer['id_to_token']
    k           = old_tokenizer['k']

    for ids, mask in zip(input_ids, attention_mask):
        valid_ids = ids[mask.astype(bool)]
        tokens    = [id_to_token[tid] for tid in valid_ids if tid in id_to_token]
        tokens    = [t for t in tokens
                     if t not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']]

        if not tokens:
            sequences.append("A" * 100)
            continue

        sequence = tokens[0]
        for token in tokens[1:]:
            if (len(token) == k
                    and len(sequence) >= k - 1
                    and sequence[-(k - 1):] == token[:k - 1]):
                sequence += token[-1]
            else:
                sequence += token

        if len(sequence) < 50:
            sequence += "A" * (50 - len(sequence))
        sequences.append(sequence)

    return sequences


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(cv_results, test_results, attention_data, plots_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    folds = [r['fold'] + 1 for r in cv_results]

    # CV F1
    f1_scores = [r['f1'] for r in cv_results]
    axes[0, 0].bar(folds, f1_scores, color='steelblue')
    axes[0, 0].axhline(np.mean(f1_scores), color='red', linestyle='--',
                        label=f'Mean={np.mean(f1_scores):.3f}')
    axes[0, 0].set_title('CV F1 Scores')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('F1')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    # CV AUC
    auc_scores = [r['auc'] for r in cv_results]
    axes[0, 1].bar(folds, auc_scores, color='darkorange')
    axes[0, 1].axhline(np.mean(auc_scores), color='red', linestyle='--',
                        label=f'Mean={np.mean(auc_scores):.3f}')
    axes[0, 1].set_title('CV AUC Scores')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    # Test confusion matrix
    cm = np.array(test_results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues',
                xticklabels=['Susceptible', 'Resistant'],
                yticklabels=['Susceptible', 'Resistant'])
    axes[1, 0].set_title(
        f"Test Confusion Matrix\n"
        f"F1={test_results['f1']:.3f} | AUC={test_results['auc']:.3f}"
    )
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # Attention patterns
    if attention_data:
        res_attn, sen_attn = [], []
        for s in attention_data:
            scores = list(s['attention_scores'])[:100]
            (res_attn if s['label'] == 1 else sen_attn).append(scores)
        if res_attn and sen_attn:
            L = max(
                max(len(s) for s in res_attn),
                max(len(s) for s in sen_attn)
            )
            def pad(seqs):
                return [s + [0] * (L - len(s)) for s in seqs]
            axes[1, 1].plot(np.mean(pad(res_attn), axis=0),
                            label='Resistant', color='firebrick', alpha=0.8)
            axes[1, 1].plot(np.mean(pad(sen_attn), axis=0),
                            label='Susceptible', color='steelblue', alpha=0.8)
            axes[1, 1].set_title('Average Attention by Class')
            axes[1, 1].set_xlabel('Token Position')
            axes[1, 1].set_ylabel('Attention Score')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient attention data',
                            ha='center', va='center',
                            transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'No attention data extracted',
                        ha='center', va='center',
                        transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(plots_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plots_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # MPS on macOS: force num_workers=0 for DataLoader stability
    num_workers = configure_pytorch_threads()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        num_workers = 0
        print("MPS device detected: forcing num_workers=0 for DataLoader stability")

    # ------------------------------------------------------------------
    # Load k-mer encoded data from step 13
    # ------------------------------------------------------------------
    train_data = np.load(snakemake.input.train, allow_pickle=True)
    test_data  = np.load(snakemake.input.test,  allow_pickle=True)

    with open(snakemake.input.tokenizer, 'rb') as f:
        old_tokenizer = pickle.load(f)

    train_dict = dict(train_data)
    test_dict  = dict(test_data)

    input_ids      = train_dict.get('X', train_dict.get('input_ids'))
    labels         = train_dict.get('y', train_dict.get('labels'))
    attention_mask = train_dict.get('attention_mask')

    test_input_ids      = test_dict.get('X', test_dict.get('input_ids'))
    test_labels         = test_dict.get('y', test_dict.get('labels'))
    test_attention_mask = test_dict.get('attention_mask')

    # ------------------------------------------------------------------
    # Reconstruct raw DNA sequences from k-mer token IDs
    # ------------------------------------------------------------------
    print("Reconstructing DNA sequences from k-mer tokens...")
    print(f"Train: {input_ids.shape} | Test: {test_input_ids.shape}")
    train_sequences = reconstruct_sequences_from_tokens(
        input_ids, attention_mask, old_tokenizer
    )
    test_sequences = reconstruct_sequences_from_tokens(
        test_input_ids, test_attention_mask, old_tokenizer
    )
    print(f"Reconstructed {len(train_sequences)} train / "
          f"{len(test_sequences)} test sequences")

    if len(train_sequences) == 0:
        raise ValueError(
            "No training sequences reconstructed — check input npz files."
        )

    # ------------------------------------------------------------------
    # Load DNABERT-2 tokenizer (official pattern), fallback to SimpleDNA
    # ------------------------------------------------------------------
    print("\nLoading DNABERT-2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True
        )
        print(f"Loaded official DNABERT-2 tokenizer "
              f"(vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"WARNING: Could not load official tokenizer: {e}")
        print("Falling back to SimpleDNATokenizer")
        tokenizer = SimpleDNATokenizer()

    # ------------------------------------------------------------------
    # Determine antibiotic
    # ------------------------------------------------------------------
    antibiotic = None
    for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
        if ab in str(snakemake.input.train):
            antibiotic = ab
            break

    print(f"\n=== DNABERT-2 FINE-TUNING: {antibiotic} ===")
    print(f"Train: {len(train_sequences)} seqs | Test: {len(test_sequences)} seqs")
    print(f"Example: {train_sequences[0][:80]}...")
    print(f"Train class dist: {np.bincount(labels)}")
    print(f"Test class dist:  {np.bincount(test_labels)}")

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    model_params = {
        'epochs':        snakemake.params.get('epochs', 7),
        'batch_size':    snakemake.params.get('batch_size', 16),
        'learning_rate': snakemake.params.get('learning_rate', 2e-5),
        'dropout':       snakemake.params.get('dropout', 0.1),
        'weight_decay':  snakemake.params.get('weight_decay', 0.01),
        'patience':      snakemake.params.get('patience', 5),
        'num_workers':   num_workers
    }

    cv_folds     = snakemake.params.get('cv_folds', 5)
    random_state = snakemake.params.get('random_state', 42)

    # ------------------------------------------------------------------
    # Geographic-temporal groups for CV
    # ------------------------------------------------------------------
    location_year_train = None
    try:
        if antibiotic:
            meta_path = str(snakemake.input.train).replace(
                f'deep_models/{antibiotic}_dnabert_train_final.npz',
                f'tree_models/{antibiotic}_train_final.csv'
            )
            meta = pd.read_csv(meta_path)
            meta['loc_year'] = (
                meta['Location'].fillna('unknown').astype(str) + '_' +
                meta['Year'].fillna('unknown').astype(str)
            )
            sid_to_group = dict(zip(meta['sample_id'], meta['loc_year']))
            if 'sample_ids' in train_dict:
                sids = train_dict['sample_ids']
                location_year_train = np.array([
                    sid_to_group.get(s, 'unknown') for s in sids
                ])
                print(f"Location-year groups: "
                      f"{len(np.unique(location_year_train))} unique")
    except Exception as e:
        print(f"Warning: Could not load location-year info: {e}")

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    print(f"\nRunning {cv_folds}-fold cross-validation...")
    cv_results = cross_validation(
        train_sequences, labels, tokenizer,
        location_year_train, cv_folds, random_state,
        **model_params
    )

    # ------------------------------------------------------------------
    # Final model with early-stopping holdout
    # ------------------------------------------------------------------
    print("\nTraining final model (with early-stopping holdout)...")
    final_model, test_results, attention_data = train_final_model(
        train_sequences, labels,
        test_sequences,  test_labels,
        tokenizer, **model_params
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    cv_mean_f1  = float(np.mean([r['f1']  for r in cv_results]))
    cv_std_f1   = float(np.std( [r['f1']  for r in cv_results]))
    cv_mean_auc = float(np.mean([r['auc'] for r in cv_results]))
    cv_std_auc  = float(np.std( [r['auc'] for r in cv_results]))

    results = {
        'antibiotic':              antibiotic,
        'cv_results':              cv_results,
        'test_results':            {
            k: v for k, v in test_results.items() if not k.startswith('_')
        },
        'cv_mean_f1':              cv_mean_f1,
        'cv_std_f1':               cv_std_f1,
        'cv_mean_auc':             cv_mean_auc,
        'cv_std_auc':              cv_std_auc,
        'model_type':              ('DNABERT-2-117M'
                                    if final_model.is_pretrained
                                    else 'BERT-from-scratch'),
        'pretrained_weights_used': final_model.is_pretrained,
        'model_params':            model_params,
        'test_predictions': {
            'y_true':  test_results['_y_true'],
            'y_pred':  test_results['_y_pred'],
            'y_proba': test_results['_y_proba']
        }
    }

    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(final_model.state_dict(), snakemake.output.model)

    with open(snakemake.output.attention, 'wb') as f:
        pickle.dump(attention_data, f)

    create_visualizations(
        cv_results, test_results, attention_data, snakemake.output.plots
    )

    print(f"\n=== SUMMARY: {antibiotic} ===")
    print(f"Pretrained weights used: {final_model.is_pretrained}")
    print(f"CV F1:    {cv_mean_f1:.3f} ± {cv_std_f1:.3f}")
    print(f"CV AUC:   {cv_mean_auc:.3f} ± {cv_std_auc:.3f}")
    print(f"Test F1:  {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")
    print(f"Final model best epoch: {test_results['best_epoch']}")


if __name__ == "__main__":
    main()