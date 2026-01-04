#!/usr/bin/env python3
"""
1D-CNN Training for AMR Prediction using K-mer Features.
PyTorch implementation with cross-validation and interpretability.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
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

class KmerDataset(Dataset):
    """PyTorch dataset for k-mer features."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    """1D CNN for k-mer classification."""
    
    def __init__(self, input_size, num_classes=2, dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.input_size = input_size
        
        # Conv layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # FC layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # For interpretability
        self.feature_maps = None
        
    def forward(self, x):
        # Add channel dimension: (batch_size, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Conv layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Store feature maps for interpretability
        self.feature_maps = x.detach()
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # FC layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


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

def extract_feature_importance(model, dataloader, device):
    """
    Extract feature importance using gradients.
    """
    model.eval()
    model.zero_grad()
    
    all_gradients = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_x.requires_grad_(True)
        
        outputs = model(batch_x)
        
        # Get gradients with respect to input for positive class
        positive_class_outputs = outputs[:, 1]
        gradients = torch.autograd.grad(
            outputs=positive_class_outputs.sum(),
            inputs=batch_x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Take absolute mean across batch
        gradients = torch.abs(gradients).mean(dim=0).squeeze()
        all_gradients.append(gradients.cpu().numpy())
        
        batch_x.requires_grad_(False)
    
    # Average gradients across all batches
    feature_importance = np.mean(all_gradients, axis=0)
    
    return feature_importance

def cross_validation(X, y, location_year_groups=None, cv_folds=5, random_state=42, num_workers=2, use_geographic_cv=False, **model_params):
    """Perform cross-validation with optional phylogenetic awareness."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"DataLoader workers: {num_workers}")

    # Use phylogenetic CV only if explicitly enabled and groups are available
    if use_geographic_cv and location_year_groups is not None:
        try:
            cv_splitter = GeographicTemporalKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            print("Using geographic-temporal cross-validation")
        except ValueError as e:
            print(f"Warning: Cannot use geographic-temporal CV ({e}), falling back to stratified CV")
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            location_year_groups = None
    else:
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if use_geographic_cv:
            print("Geographic-temporal CV disabled or groups not available, using stratified cross-validation")
        else:
            print("Using stratified cross-validation (geographic-temporal CV disabled for stability)")
    
    cv_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups=location_year_groups)):
        print(f"\nFold {fold + 1}/{cv_folds}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Standardize features (zero mean, unit variance)
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        print(f"  Features standardized: mean={X_train_fold.mean():.4f}, std={X_train_fold.std():.4f}")
        
        # Log cluster information if available
        if location_year_groups is not None:
            loc_year_tr = location_year_groups[train_idx]
            loc_year_val = location_year_groups[val_idx]
            print(f"Fold {fold + 1}: Train location-years={len(np.unique(loc_year_tr))}, "
                  f"Val location-years={len(np.unique(loc_year_val))}")
            print(f"  Train class dist: {Counter(y_train_fold)}, Val class dist: {Counter(y_val_fold)}")
        
        # Create datasets
        train_dataset = KmerDataset(X_train_fold, y_train_fold)
        val_dataset = KmerDataset(X_val_fold, y_val_fold)
        
        # Create dataloaders (no weighted sampling - use class weights in loss instead)
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params.get('batch_size', 32),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_params.get('batch_size', 32),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model
        model = CNN1D(
            input_size=X.shape[1],
            dropout=model_params.get('dropout', 0.3)
        ).to(device)
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train_fold)
        total_samples = len(y_train_fold)
        
        # Check for single-class fold
        if len(class_counts) == 1:
            print(f"  Fold {fold + 1} has single class: {class_counts}")
            print(f"  Skipping fold due to single-class distribution")
            # Create dummy results for this fold
            fold_results = {
                'fold': fold,
                'f1': 0.0,
                'balanced_accuracy': 0.5,
                'auc': 0.5,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
            cv_results.append(fold_results)
            fold_models.append({})  # Empty model state
            continue
        
        # Ensure we have weights for both classes (0 and 1)
        if len(class_counts) < 2:
            class_counts = np.array([class_counts[0] if len(class_counts) > 0 else 1, 1])
        
        class_weights = torch.FloatTensor([
            total_samples / (2 * count) if count > 0 else 1.0 for count in class_counts[:2]
        ]).to(device)
        
        print(f"  Fold {fold + 1} class distribution: {class_counts}")
        print(f"  Calculated class weights: {class_weights.cpu().numpy()}")
        
        # Loss and optimizer with class weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_params.get('learning_rate', 0.001),
            weight_decay=model_params.get('weight_decay', 1e-4)
        )
        
        # Training loop
        epochs = model_params.get('epochs', 100)
        best_val_auc = 0
        best_model_state = model.state_dict().copy()  # Initialize with current model
        patience = model_params.get('patience', 15)  # Increased from 7 for more stable training
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_preds, val_probs, val_labels = evaluate(model, val_loader, device)

            # Calculate metrics - use AUC for early stopping (more stable than F1)
            if len(set(val_labels)) == 1:
                val_auc = 0.5
                val_f1 = 0.0
            else:
                val_auc = roc_auc_score(val_labels, val_probs)
                val_f1 = f1_score(val_labels, val_preds)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}, val_f1={val_f1:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict().copy())
        
        # Final evaluation
        val_preds, val_probs, val_labels = evaluate(model, val_loader, device)
        
        # Handle single-class validation set
        if len(set(val_labels)) == 1:
            print(f"  Warning: Fold {fold + 1} validation set has single class")
            fold_results = {
                'fold': fold,
                'f1': 0.0,
                'balanced_accuracy': 0.5,
                'auc': 0.5,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
        else:
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

def train_final_model(X_train, y_train, X_test, y_test, num_workers=2, **model_params):
    """Train final model on all training data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"Features standardized for final model: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    # Create datasets
    train_dataset = KmerDataset(X_train, y_train)
    test_dataset = KmerDataset(X_test, y_test)

    # Create dataloaders (no weighted sampling - use class weights in loss instead)
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_params.get('batch_size', 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_params.get('batch_size', 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = CNN1D(
        input_size=X_train.shape[1],
        dropout=model_params.get('dropout', 0.3)
    ).to(device)
    
    # Calculate class weights for final model
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Ensure we have weights for both classes (0 and 1)
    if len(class_counts) < 2:
        class_counts = np.array([class_counts[0] if len(class_counts) > 0 else 1, 1])
    
    class_weights = torch.FloatTensor([
        total_samples / (2 * count) if count > 0 else 1.0 for count in class_counts[:2]
    ]).to(device)
    
    print(f"Final model class distribution: {class_counts}")
    print(f"Final model class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_params.get('learning_rate', 0.001),
        weight_decay=model_params.get('weight_decay', 1e-4)
    )
    
    # Training loop
    epochs = model_params.get('epochs', 100)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    
    # Test evaluation
    test_preds, test_probs, test_labels = evaluate(model, test_loader, device)
    
    # Handle empty test set
    if len(test_labels) == 0:
        print("Warning: Empty test set, using dummy test results")
        test_results = {
            'f1': 0.0,
            'balanced_accuracy': 0.5,
            'auc': 0.5,
            'confusion_matrix': [[0, 0], [0, 0]],
            'classification_report': {}
        }
    else:
        test_results = {
            'f1': f1_score(test_labels, test_preds) if len(set(test_labels)) > 1 else 0.0,
            'balanced_accuracy': balanced_accuracy_score(test_labels, test_preds),
            'auc': roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
            'classification_report': classification_report(test_labels, test_preds, output_dict=True) if len(set(test_labels)) > 1 else {}
        }
    
    # Extract feature importance
    print("Extracting feature importance...")
    feature_importance = extract_feature_importance(model, train_loader, device)
    
    return model, test_results, feature_importance

def create_visualizations(cv_results, test_results, feature_importance, kmer_names, plots_path):
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
        
        # Feature importance (top 20)
        if len(feature_importance) > 0 and len(kmer_names) > 0:
            top_indices = np.argsort(feature_importance)[-20:]
            top_importance = feature_importance[top_indices]
            top_kmers = [kmer_names[i] if i < len(kmer_names) else f'kmer_{i}' for i in top_indices]
            
            axes[1, 1].barh(range(len(top_importance)), top_importance)
            axes[1, 1].set_yticks(range(len(top_importance)))
            axes[1, 1].set_yticklabels(top_kmers, fontsize=8)
            axes[1, 1].set_title('Top 20 Important K-mers')
            axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    # Configure PyTorch threading for optimal CPU utilization
    num_workers = configure_pytorch_threads()

    # Load k-mer dataset
    kmer_data = np.load(snakemake.input.train, allow_pickle=True)
    kmer_test_data = np.load(snakemake.input.test, allow_pickle=True) if snakemake.input.test else None
    
    X_train = kmer_data['X']
    y_train = kmer_data['y']
    train_sample_ids = kmer_data['sample_ids'] if 'sample_ids' in kmer_data else None
    kmer_names = kmer_data['kmer_names'].tolist() if 'kmer_names' in kmer_data else []
    
    # Create location-year groups from metadata
    location_year_train = None
    if train_sample_ids is not None:
        try:
            # Get antibiotic name to find corresponding metadata
            antibiotic = None
            for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
                if ab in str(snakemake.input.train):
                    antibiotic = ab
                    break
            
            if antibiotic:
                # Load training metadata to get location and year
                metadata_path = str(snakemake.input.train).replace(f'kmer_datasets/{antibiotic}_train_kmer.npz', 
                                                                 f'tree_models/{antibiotic}_train_final.csv')
                train_metadata = pd.read_csv(metadata_path)
                
                # Create sample_id to location-year mapping
                train_metadata['Location_Year'] = (train_metadata['Location'].fillna('unknown').astype(str) + '_' + 
                                                  train_metadata['Year'].fillna('unknown').astype(str))
                sample_to_group = dict(zip(train_metadata['sample_id'], 
                                         train_metadata['Location_Year']))
                
                # Map k-mer sample IDs to location-year groups
                location_year_train = np.array([sample_to_group.get(sid, 'unknown') 
                                              for sid in train_sample_ids])
                
                print(f"Loaded location-year info for {len(location_year_train)} samples")
                print(f"Unique location-year groups: {len(np.unique(location_year_train))}")
                print(f"Location-year distribution: {Counter(location_year_train).most_common(10)}")
        except Exception as e:
            print(f"Warning: Could not load location-year information: {e}")
            location_year_train = None
    
    if kmer_test_data is not None:
        X_test = kmer_test_data['X']
        y_test = kmer_test_data['y']
    else:
        # Split training data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    print(f"=== 1D-CNN TRAINING ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"K-mer features: {X_train.shape[1]}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Model parameters
    model_params = {
        'epochs': snakemake.params.get('epochs', 100),
        'batch_size': snakemake.params.get('batch_size', 32),
        'learning_rate': snakemake.params.get('learning_rate', 0.001),
        'dropout': snakemake.params.get('dropout', 0.3),
        'weight_decay': snakemake.params.get('weight_decay', 1e-4),
        'patience': snakemake.params.get('patience', 7)  # Reduced from 10 for faster training
    }
    
    cv_folds = snakemake.params.get('cv_folds', 5)
    random_state = snakemake.params.get('random_state', 42)
    
    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    # Disable geographic-temporal CV by default for stability (set use_geographic_cv=True to enable)
    use_geographic_cv = False
    cv_results, fold_models = cross_validation(
        X_train, y_train, location_year_train, cv_folds, random_state, num_workers, use_geographic_cv, **model_params
    )

    # Train final model
    print(f"\nTraining final model...")
    final_model, test_results, feature_importance = train_final_model(
        X_train, y_train, X_test, y_test, num_workers, **model_params
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
        'n_features': X_train.shape[1],
        'feature_importance_top20': {
            'indices': np.argsort(feature_importance)[-20:].tolist(),
            'scores': feature_importance[np.argsort(feature_importance)[-20:]].tolist(),
            'kmers': [kmer_names[i] if i < len(kmer_names) else f'kmer_{i}' 
                     for i in np.argsort(feature_importance)[-20:]]
        }
    }
    
    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    torch.save(final_model.state_dict(), snakemake.output.model)
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'kmer': kmer_names[:len(feature_importance)] if len(kmer_names) >= len(feature_importance) else [f'kmer_{i}' for i in range(len(feature_importance))],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(snakemake.output.importance, index=False)
    
    # Create plots
    create_visualizations(cv_results, test_results, feature_importance, kmer_names, snakemake.output.plots)
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"CV F1: {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"CV AUC: {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")

if __name__ == "__main__":
    main()