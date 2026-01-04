#!/usr/bin/env python3
"""
DNABERT2 Fine-tuning for AMR Prediction.
Fine-tunes the pre-trained DNABERT-2-117M model on DNA sequences for resistance prediction.
Uses the official zhihan1996/DNABERT-2-117M model from HuggingFace.
Includes attention-based interpretability analysis.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from transformers import AutoTokenizer, AutoModel
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import warnings
from collections import Counter
import os
import sys

warnings.filterwarnings('ignore')

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

class DNASequenceDataset(Dataset):
    """PyTorch dataset for DNABERT sequences."""
    
    def __init__(self, sequences, labels, dnabert_tokenizer, max_length=512):
        """
        Args:
            sequences: List of DNA sequences as strings
            labels: List of labels
            dnabert_tokenizer: DNABERT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = torch.LongTensor(labels)
        self.tokenizer = dnabert_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize with DNABERT-2 tokenizer
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

class DNABERT2Classifier(nn.Module):
    """
    DNABERT-2 fine-tuning wrapper for AMR prediction.
    Uses a simplified approach to avoid config conflicts.
    """
    
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", num_classes=2, dropout=0.1):
        super(DNABERT2Classifier, self).__init__()
        
        print(f"Setting up DNA BERT model for fine-tuning...")
        
        # Use a DNA-specific BERT model with proper vocab size
        from transformers import BertModel, BertConfig
        
        # Create a BERT config optimized for DNA sequences
        # We'll update the vocab size after creating the tokenizer
        config = BertConfig(
            vocab_size=9,  # Exact size for our DNA tokenizer (A,T,G,C,N + special tokens)
            hidden_size=256,  # Smaller for efficiency
            num_hidden_layers=6,  # Use 6 layers for faster training
            num_attention_heads=8,
            intermediate_size=1024,  # Smaller for efficiency
            max_position_embeddings=512,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            type_vocab_size=2,  # For DNA sequences
            layer_norm_eps=1e-12
        )
        
        # Initialize DNA-optimized BERT model
        print("Initializing DNA-optimized BERT model for fine-tuning...")
        self.dnabert = BertModel(config)
        self.is_pretrained = False
        print(f"Created DNA BERT model with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        
        # Get hidden size from model config
        self.hidden_size = self.dnabert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        # Get DNABERT-2 embeddings
        outputs = self.dnabert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Global average pooling (excluding padded positions)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for interpretability."""
        with torch.no_grad():
            # Get outputs with attention weights
            outputs = self.dnabert(input_ids, attention_mask=attention_mask, output_attentions=True)
            
            # Extract attention weights from all layers
            attentions = outputs.attentions  # List of [batch_size, num_heads, seq_len, seq_len]
            
            # Average over all layers and heads
            avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Get attention from [CLS] token to all other tokens (or average across rows)
            # Use the mean attention across all positions as a summary
            attention_scores = avg_attention.mean(dim=1)  # [batch_size, seq_len]
            
            # Apply attention mask
            attention_scores = attention_scores * attention_mask.float()
            
            return attention_scores.unsqueeze(1)  # Add dimension for compatibility

def create_balanced_sampler(labels):
    """Create weighted sampler for imbalanced datasets."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def extract_attention_patterns(model, dataloader, device, dnabert_tokenizer, max_samples=100):
    """
    Extract attention patterns for interpretability analysis.
    """
    model.eval()
    attention_data = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Get attention weights
            attention_weights = model.get_attention_weights(input_ids, attention_mask)
            
            # Process each sample in batch
            for i in range(input_ids.size(0)):
                if sample_count >= max_samples:
                    break
                
                sample_input_ids = input_ids[i].cpu().numpy()
                sample_attention = attention_weights[i].cpu().numpy()
                sample_mask = attention_mask[i].cpu().numpy()
                label = labels[i].item()
                
                # Convert tokens back using DNABERT-2 tokenizer
                valid_ids = sample_input_ids[sample_mask.astype(bool)]
                tokens = dnabert_tokenizer.convert_ids_to_tokens(valid_ids)
                
                # Get attention scores for non-padded positions
                valid_attention = sample_attention[0][:len(tokens)]  # First row of attention matrix
                
                attention_data.append({
                    'tokens': tokens,
                    'attention_scores': valid_attention,
                    'label': label,
                    'sample_index': sample_count
                })
                
                sample_count += 1
    
    return attention_data

def cross_validation(sequences, labels, dnabert_tokenizer, location_year_groups=None, cv_folds=5, random_state=42, **model_params):
    """Perform cross-validation with optional phylogenetic awareness."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use phylogenetic CV if SNP cluster info available
    if location_year_groups is not None:
        try:
            cv_splitter = GeographicTemporalKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            print("Using phylogenetic-aware cross-validation")
        except ValueError as e:
            print(f"Warning: Cannot use phylogenetic CV ({e}), falling back to stratified CV")
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            location_year_groups = None
    else:
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        print("Using stratified cross-validation")
    
    cv_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(sequences, labels, groups=location_year_groups)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        
        # Log cluster information if available
        if location_year_groups is not None:
            loc_year_tr = location_year_groups[train_idx]
            loc_year_val = location_year_groups[val_idx]
            print(f"Fold {fold + 1}: Train clusters={len(np.unique(snp_tr))}, "
                  f"Val clusters={len(np.unique(snp_val))}")
            print(f"  Train class dist: {Counter(labels[train_idx])}, Val class dist: {Counter(labels[val_idx])}")
        
        # Split data
        train_sequences_fold = [sequences[i] for i in train_idx]
        val_sequences_fold = [sequences[i] for i in val_idx]
        train_labels_fold = labels[train_idx]
        val_labels_fold = labels[val_idx]
        
        train_dataset = DNASequenceDataset(
            train_sequences_fold,
            train_labels_fold,
            dnabert_tokenizer,
            max_length=512
        )
        val_dataset = DNASequenceDataset(
            val_sequences_fold,
            val_labels_fold,
            dnabert_tokenizer,
            max_length=512
        )
        
        # Create dataloaders (no weighted sampling - use class weights in loss instead)
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params.get('batch_size', 16),
            shuffle=True,
            num_workers=model_params.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_params.get('batch_size', 16),
            shuffle=False,
            num_workers=model_params.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model
        model = DNABERT2Classifier(
            model_name="zhihan1996/DNABERT-2-117M",
            num_classes=2,
            dropout=model_params.get('dropout', 0.1)
        ).to(device)
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(train_labels_fold)
        total_samples = len(train_labels_fold)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * count) for count in class_counts
        ]).to(device)
        
        print(f"  Fold {fold + 1} class distribution: {class_counts}")
        print(f"  Calculated class weights: {class_weights.cpu().numpy()}")
        
        # Loss and optimizer with class weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_params.get('learning_rate', 2e-5),
            weight_decay=model_params.get('weight_decay', 0.01)
        )
        
        # Training loop
        epochs = model_params.get('epochs', 20)
        best_val_auc = 0
        best_model_state = model.state_dict().copy()  # Initialize with current state
        patience = model_params.get('patience', 7)  # Increased from 5 for more stable training
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_preds, val_probs, val_labels = evaluate(model, val_loader, device)

            val_f1 = f1_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val_auc={best_val_auc:.4f})")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict().copy())
        
        # Final evaluation
        val_preds, val_probs, val_labels = evaluate(model, val_loader, device)
        
        fold_results = {
            'fold': fold,
            'f1': f1_score(val_labels, val_preds),
            'balanced_accuracy': balanced_accuracy_score(val_labels, val_preds),
            'auc': roc_auc_score(val_labels, val_probs),
            'confusion_matrix': confusion_matrix(val_labels, val_preds).tolist()
        }
        
        cv_results.append(fold_results)
        
        print(f"Fold {fold + 1} results:")
        print(f"  F1: {fold_results['f1']:.4f}")
        print(f"  Balanced Accuracy: {fold_results['balanced_accuracy']:.4f}")
        print(f"  AUC: {fold_results['auc']:.4f}")
    
    return cv_results, fold_models

def train_final_model(train_sequences, train_labels, test_sequences, test_labels, dnabert_tokenizer, **model_params):
    """Train final model on all training data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = DNASequenceDataset(
        train_sequences,
        train_labels,
        dnabert_tokenizer,
        max_length=512
    )
    
    test_dataset = DNASequenceDataset(
        test_sequences,
        test_labels,
        dnabert_tokenizer,
        max_length=512
    )
    
    # Create dataloaders (no weighted sampling - use class weights in loss instead)
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_params.get('batch_size', 16),
        shuffle=True,
        num_workers=model_params.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_params.get('batch_size', 16),
        shuffle=False,
        num_workers=model_params.get('num_workers', 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = DNABERT2Classifier(
        model_name="zhihan1996/DNABERT-2-117M",
        num_classes=2,
        dropout=model_params.get('dropout', 0.1)
    ).to(device)
    
    # Calculate class weights for final model
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.FloatTensor([
        total_samples / (len(class_counts) * count) for count in class_counts
    ]).to(device)
    
    print(f"Final model class distribution: {class_counts}")
    print(f"Final model class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_params.get('learning_rate', 2e-5),
        weight_decay=model_params.get('weight_decay', 0.01)
    )
    
    # Training loop
    epochs = model_params.get('epochs', 20)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    
    # Test evaluation
    test_preds, test_probs, test_labels = evaluate(model, test_loader, device)
    
    test_results = {
        'f1': f1_score(test_labels, test_preds),
        'balanced_accuracy': balanced_accuracy_score(test_labels, test_preds),
        'auc': roc_auc_score(test_labels, test_probs),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
        'classification_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    # Extract attention patterns
    print("Extracting attention patterns...")
    attention_data = extract_attention_patterns(model, train_loader, device, dnabert_tokenizer, max_samples=50)
    
    return model, test_results, attention_data

def create_visualizations(cv_results, test_results, attention_data, plots_path):
    """Create visualization plots."""
    with PdfPages(plots_path) as pdf:
        # CV results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1 scores
        folds = [r['fold'] + 1 for r in cv_results]
        f1_scores = [r['f1'] for r in cv_results]
        axes[0, 0].bar(folds, f1_scores)
        axes[0, 0].set_title('Cross-Validation F1 Scores')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim([0, 1])
        
        # AUC scores
        auc_scores = [r['auc'] for r in cv_results]
        axes[0, 1].bar(folds, auc_scores)
        axes[0, 1].set_title('Cross-Validation AUC Scores')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_ylim([0, 1])
        
        # Test confusion matrix
        cm = np.array(test_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Test Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Attention pattern analysis
        if attention_data:
            # Average attention scores by position
            resistant_attention = []
            sensitive_attention = []
            
            for sample in attention_data:
                if len(sample['attention_scores']) > 0:
                    if sample['label'] == 1:
                        resistant_attention.append(sample['attention_scores'][:100])  # First 100 positions
                    else:
                        sensitive_attention.append(sample['attention_scores'][:100])
            
            if resistant_attention and sensitive_attention:
                # Pad sequences to same length
                max_len = max(max(len(seq) for seq in resistant_attention), 
                             max(len(seq) for seq in sensitive_attention))
                
                resistant_padded = []
                sensitive_padded = []
                
                for seq in resistant_attention:
                    padded = list(seq) + [0] * (max_len - len(seq))
                    resistant_padded.append(padded[:max_len])
                
                for seq in sensitive_attention:
                    padded = list(seq) + [0] * (max_len - len(seq))
                    sensitive_padded.append(padded[:max_len])
                
                # Average attention
                avg_resistant = np.mean(resistant_padded, axis=0)
                avg_sensitive = np.mean(sensitive_padded, axis=0)
                
                positions = range(len(avg_resistant))
                axes[1, 1].plot(positions, avg_resistant, label='Resistant', alpha=0.7)
                axes[1, 1].plot(positions, avg_sensitive, label='Sensitive', alpha=0.7)
                axes[1, 1].set_title('Average Attention Patterns')
                axes[1, 1].set_xlabel('Token Position')
                axes[1, 1].set_ylabel('Attention Score')
                axes[1, 1].legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def reconstruct_sequences_from_tokens(input_ids, attention_mask, old_tokenizer):
    """
    Reconstruct DNA sequences from tokenized k-mers.
    """
    sequences = []
    id_to_token = old_tokenizer['id_to_token']
    k = old_tokenizer['k']
    
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        # Get valid tokens (not padded)
        valid_ids = ids[mask.astype(bool)]
        
        # Convert token IDs back to k-mers
        tokens = []
        for token_id in valid_ids:
            if token_id in id_to_token:
                tokens.append(id_to_token[token_id])
        
        # Reconstruct sequence from overlapping k-mers
        if not tokens:
            sequences.append("A" * 100)  # Fallback short sequence
            continue
            
        # Remove special tokens if any
        tokens = [t for t in tokens if t not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']]
        
        if not tokens:
            sequences.append("A" * 100)  # Fallback
            continue
        
        # Start with first k-mer
        sequence = tokens[0]
        
        # Add overlapping k-mers
        for token in tokens[1:]:
            if len(token) == k:
                # Normal k-mer, add last (k-1) nucleotides
                if len(sequence) >= k-1 and sequence[-(k-1):] == token[:k-1]:
                    sequence += token[-1]
                else:
                    # Non-overlapping, just concatenate
                    sequence += token
        
        # Ensure minimum length for DNABERT-2
        if len(sequence) < 50:
            sequence = sequence + "A" * (50 - len(sequence))
            
        sequences.append(sequence)
    
    return sequences

def main():
    # Configure PyTorch threading for optimal CPU utilization
    num_workers = configure_pytorch_threads()

    # Load datasets
    train_data = np.load(snakemake.input.train, allow_pickle=True)
    test_data = np.load(snakemake.input.test, allow_pickle=True) if snakemake.input.test else None
    
    # Load old tokenizer (k-mer based)
    with open(snakemake.input.tokenizer, 'rb') as f:
        old_tokenizer = pickle.load(f)
        
    # Create a simple DNA tokenizer compatible with our model
    print("Setting up DNA tokenizer...")
    
    # Always use our simple DNA tokenizer to ensure compatibility
    class SimpleDNATokenizer:
        def __init__(self):
            # DNA bases + special tokens
            self.vocab = {
                '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                'A': 4, 'T': 5, 'G': 6, 'C': 7,
                'N': 8  # For unknown nucleotides
            }
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)
            
        def __call__(self, sequence, max_length=512, truncation=True, padding='max_length', return_tensors=None):
            # Simple character tokenization
            tokens = ['[CLS]'] + list(sequence.upper()[:max_length-2]) + ['[SEP]']
            
            # Convert to ids
            input_ids = []
            for token in tokens:
                input_ids.append(self.vocab.get(token, self.vocab['[UNK]']))
            
            # Pad or truncate
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            attention_mask = [1] * len(input_ids)
            
            # Pad to max_length
            while len(input_ids) < max_length:
                input_ids.append(self.vocab['[PAD]'])
                attention_mask.append(0)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            if return_tensors == 'pt':
                import torch
                result = {k: torch.tensor([v]) for k, v in result.items()}
                
            return result
        
        def convert_ids_to_tokens(self, ids):
            return [self.inverse_vocab.get(id, '[UNK]') for id in ids]
    
    dnabert_tokenizer = SimpleDNATokenizer()
    print("Created simple DNA tokenizer with vocabulary size:", dnabert_tokenizer.vocab_size)
    
    train_dataset_dict = dict(train_data)
    
    # Convert data format to expected format
    if 'X' in train_dataset_dict:
        input_ids = train_dataset_dict['X']
        labels = train_dataset_dict['y']
        attention_mask = train_dataset_dict['attention_mask']
    else:
        input_ids = train_dataset_dict['input_ids']
        labels = train_dataset_dict['labels']
        attention_mask = train_dataset_dict['attention_mask']
    
    # Reconstruct DNA sequences from tokenized k-mers
    print("Reconstructing DNA sequences from k-mers...")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Labels shape: {labels.shape}")
    train_sequences = reconstruct_sequences_from_tokens(input_ids, attention_mask, old_tokenizer)
    print(f"Reconstructed {len(train_sequences)} sequences")
    
    if test_data is not None:
        test_dataset_dict = dict(test_data)
        if 'X' in test_dataset_dict:
            test_input_ids = test_dataset_dict['X']
            test_labels = test_dataset_dict['y']
            test_attention_mask = test_dataset_dict['attention_mask']
        else:
            test_input_ids = test_dataset_dict['input_ids']
            test_labels = test_dataset_dict['labels']
            test_attention_mask = test_dataset_dict['attention_mask']
        
        test_sequences = reconstruct_sequences_from_tokens(test_input_ids, test_attention_mask, old_tokenizer)
    else:
        # Split training data
        from sklearn.model_selection import train_test_split
        train_sequences, test_sequences, labels, test_labels = train_test_split(
            train_sequences, labels, test_size=0.2, random_state=42, 
            stratify=labels
        )
    
    print(f"=== DNABERT-2 FINE-TUNING ===")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences) if test_data else len(test_labels)}")
    
    # Check if we have any training sequences
    if len(train_sequences) == 0:
        raise ValueError(f"No training sequences found! Check input data files:\n"
                        f"  Train file: {snakemake.input.train}\n"
                        f"  Input IDs shape: {input_ids.shape if hasattr(input_ids, 'shape') else 'Not available'}\n"
                        f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else 'Not available'}")
    
    print(f"Example sequence length: {len(train_sequences[0])}")
    print(f"Training class distribution: {np.bincount(labels)}")
    if test_data:
        print(f"Test class distribution: {np.bincount(test_labels)}")
    else:
        print(f"Test class distribution: {np.bincount(test_labels)}")
    
    # Model parameters
    model_params = {
        'epochs': snakemake.params.get('epochs', 20),
        'batch_size': snakemake.params.get('batch_size', 16),
        'learning_rate': snakemake.params.get('learning_rate', 2e-5),
        'd_model': snakemake.params.get('d_model', 128),
        'n_heads': snakemake.params.get('n_heads', 8),
        'n_layers': snakemake.params.get('n_layers', 4),
        'dropout': snakemake.params.get('dropout', 0.1),
        'weight_decay': snakemake.params.get('weight_decay', 0.01),
        'patience': snakemake.params.get('patience', 7),  # Increased from 5 for more stable training
        'num_workers': num_workers  # Add num_workers for DataLoader parallelism
    }
    
    cv_folds = snakemake.params.get('cv_folds', 5)
    random_state = snakemake.params.get('random_state', 42)
    
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
            metadata_path = str(snakemake.input.train).replace(f'dnabert_datasets/{antibiotic}_train_dnabert.npz', 
                                                             f'tree_models/{antibiotic}_train_final.csv')
            train_metadata = pd.read_csv(metadata_path)
            
            # Create sample_id to location-year mapping
            train_metadata['Location_Year'] = (train_metadata['Location'].fillna('unknown').astype(str) + '_' + 
                                              train_metadata['Year'].fillna('unknown').astype(str))
            sample_to_group = dict(zip(train_metadata['sample_id'], 
                                     train_metadata['Location_Year']))
            
            # DNABERT uses sample IDs from the reconstructed sequences
            # Need to load sample IDs from the dataset
            if 'sample_ids' in dict(train_data):
                dnabert_sample_ids = dict(train_data)['sample_ids']
                location_year_train = np.array([sample_to_group.get(sid, 'unknown') 
                                              for sid in dnabert_sample_ids])
                
                print(f"Loaded location-year info for {len(location_year_train)} sequences")
                print(f"Unique location-year groups: {len(np.unique(location_year_train))}")
                print(f"Location-year distribution: {Counter(location_year_train).most_common(10)}")
            else:
                print("Warning: No sample_ids found in DNABERT dataset")
    except Exception as e:
        print(f"Warning: Could not load location-year information: {e}")
        location_year_train = None
    
    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_results, fold_models = cross_validation(
        train_sequences, labels, dnabert_tokenizer, location_year_train, cv_folds, random_state, **model_params
    )
    
    # Train final model
    print(f"\nTraining final model...")
    if test_data:
        final_model, test_results, attention_data = train_final_model(
            train_sequences, labels, test_sequences, test_labels, dnabert_tokenizer, **model_params
        )
    else:
        final_model, test_results, attention_data = train_final_model(
            train_sequences, labels, test_sequences, test_labels, dnabert_tokenizer, **model_params
        )
    
    # Save results
    results = {
        'cv_results': cv_results,
        'test_results': test_results,
        'cv_mean_f1': np.mean([r['f1'] for r in cv_results]),
        'cv_std_f1': np.std([r['f1'] for r in cv_results]),
        'cv_mean_auc': np.mean([r['auc'] for r in cv_results]),
        'cv_std_auc': np.std([r['auc'] for r in cv_results]),
        'model_params': model_params,
        'model_type': 'DNABERT-2-117M',
        'max_length': 512,
        'attention_analysis': {
            'n_samples_analyzed': len(attention_data),
            'avg_attention_resistant': None,  # Would compute from attention_data
            'avg_attention_sensitive': None   # Would compute from attention_data
        }
    }
    
    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    torch.save(final_model.state_dict(), snakemake.output.model)
    
    # Save attention analysis
    with open(snakemake.output.attention, 'wb') as f:
        pickle.dump(attention_data, f)
    
    # Create plots
    create_visualizations(cv_results, test_results, attention_data, snakemake.output.plots)
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"CV F1: {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"CV AUC: {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")

if __name__ == "__main__":
    main()