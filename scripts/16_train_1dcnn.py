#!/usr/bin/env python3
"""
1D-CNN Training for AMR Prediction using K-mer Features.
PyTorch implementation with cross-validation and interpretability.

Fixes applied:
- sys.path.insert for utils imports when run via Snakemake temp script
- kmer_names key corrected to 'feature_names' (matches step 12 output)
- create_visualizations uses PNG (matplotlib) not PdfPages
- train_final_model uses 10% early-stopping holdout to prevent overfitting
- num_workers=0 on MPS (macOS) for DataLoader stability
- matplotlib backend set to Agg
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import json
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
from utils.cross_validation import GeographicTemporalKFold, get_cross_validator


# =============================================================================
# Dataset
# =============================================================================

class KmerDataset(Dataset):
    """PyTorch dataset for k-mer features."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# Model
# =============================================================================

class CNN1D(nn.Module):
    """1D CNN for k-mer frequency classification."""

    def __init__(self, input_size, num_classes=2, dropout=0.3):
        super(CNN1D, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3      = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3        = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.feature_maps = None

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))

        self.feature_maps = x.detach()

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# =============================================================================
# Training helpers
# =============================================================================

def make_dataloader(X, y, batch_size, shuffle, num_workers):
    dataset = KmerDataset(X, y)
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
        total / (2 * c) if c > 0 else 1.0 for c in class_counts[:2]
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
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def extract_feature_importance(model, dataloader, device):
    """Gradient-based feature importance (saliency map)."""
    model.eval()
    model.zero_grad()
    all_gradients = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_x.requires_grad_(True)

        outputs = model(batch_x)
        gradients = torch.autograd.grad(
            outputs=outputs[:, 1].sum(),
            inputs=batch_x,
            create_graph=False,
            retain_graph=False
        )[0]

        gradients = torch.abs(gradients).mean(dim=0).squeeze()
        all_gradients.append(gradients.cpu().detach().numpy())
        batch_x.requires_grad_(False)

    return np.mean(all_gradients, axis=0)


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
# Cross-validation
# =============================================================================

def cross_validation(X, y, location_year_groups=None, cv_folds=5,
                     random_state=42, num_workers=0,
                     use_geographic_cv=False, **model_params):
    device     = get_device()
    batch_size = model_params.get('batch_size', 32)
    epochs     = model_params.get('epochs', 100)
    patience   = model_params.get('patience', 15)
    dropout    = model_params.get('dropout', 0.3)
    lr         = model_params.get('learning_rate', 0.0005)
    wd         = model_params.get('weight_decay', 1e-4)

    print(f"DataLoader workers: {num_workers}")

    if use_geographic_cv and location_year_groups is not None:
        try:
            cv_splitter = GeographicTemporalKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            print("Using geographic-temporal cross-validation")
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

    cv_results  = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(
        cv_splitter.split(X, y, groups=location_year_groups)
    ):
        print(f"\nFold {fold + 1}/{cv_folds}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        if location_year_groups is not None:
            loc_tr  = location_year_groups[train_idx]
            loc_val = location_year_groups[val_idx]
            print(f"  Train groups={len(np.unique(loc_tr))} | "
                  f"Val groups={len(np.unique(loc_val))}")
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

        model = CNN1D(input_size=X.shape[1], dropout=dropout).to(device)

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

        if len(np.unique(val_labels)) < 2:
            fold_results = {
                'fold': fold, 'best_epoch': best_epoch,
                'f1': 0.0, 'balanced_accuracy': 0.5, 'auc': 0.5,
                'confusion_matrix': [[0,0],[0,0]]
            }
        else:
            fold_results = {
                'fold':              fold,
                'best_epoch':        best_epoch,
                'f1':                float(f1_score(val_labels, val_preds, zero_division=0)),
                'balanced_accuracy': float(balanced_accuracy_score(val_labels, val_preds)),
                'auc':               float(roc_auc_score(val_labels, val_probs)),
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
                      num_workers=0, **model_params):
    """
    Train final model using a 10% early-stopping holdout.
    Prevents the overfitting seen when training to max epochs with no stopping
    signal (train acc → 100%, test generalisation degrades).
    """
    device     = get_device()
    batch_size = model_params.get('batch_size', 32)
    epochs     = model_params.get('epochs', 100)
    patience   = model_params.get('patience', 15)
    dropout    = model_params.get('dropout', 0.3)
    lr         = model_params.get('learning_rate', 0.0005)
    wd         = model_params.get('weight_decay', 1e-4)

    X_ft, X_es, y_ft, y_es = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_ft   = scaler.fit_transform(X_ft)
    X_es   = scaler.transform(X_es)
    X_test = scaler.transform(X_test)

    print(f"\nFinal model — fine-tune: {len(X_ft)} | "
          f"ES holdout: {len(X_es)} | Test: {len(X_test)}")
    print(f"Fine-tune class dist: {np.bincount(y_ft)}")

    train_loader = make_dataloader(X_ft,   y_ft,   batch_size, True,  num_workers)
    es_loader    = make_dataloader(X_es,   y_es,   batch_size, False, num_workers)
    test_loader  = make_dataloader(X_test, y_test, batch_size, False, num_workers)

    model = CNN1D(input_size=X_ft.shape[1], dropout=dropout).to(device)

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

    if len(test_labels) == 0:
        test_results = {
            'f1': 0.0, 'balanced_accuracy': 0.5, 'auc': 0.5,
            'confusion_matrix': [[0,0],[0,0]], 'classification_report': {}
        }
        test_preds_list, test_probs_list, test_labels_list = [], [], []
    else:
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
        test_preds_list  = test_preds.tolist()
        test_probs_list  = test_probs.tolist()
        test_labels_list = test_labels.tolist()

    print("Extracting feature importance...")
    feature_importance = extract_feature_importance(model, train_loader, device)

    return (model, test_results, feature_importance,
            test_preds_list, test_probs_list, test_labels_list)


# =============================================================================
# Visualization — PNG output (replaces broken PdfPages → .png)
# =============================================================================

def create_visualizations(cv_results, test_results, feature_importance,
                           kmer_names, plots_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    folds = [r['fold'] + 1 for r in cv_results]

    f1_scores = [r['f1'] for r in cv_results]
    axes[0, 0].bar(folds, f1_scores, color='steelblue')
    axes[0, 0].axhline(np.mean(f1_scores), color='red', linestyle='--',
                        label=f'Mean={np.mean(f1_scores):.3f}')
    axes[0, 0].set_title('CV F1 Scores')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    auc_scores = [r['auc'] for r in cv_results]
    axes[0, 1].bar(folds, auc_scores, color='darkorange')
    axes[0, 1].axhline(np.mean(auc_scores), color='red', linestyle='--',
                        label=f'Mean={np.mean(auc_scores):.3f}')
    axes[0, 1].set_title('CV AUC Scores')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

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

    if len(feature_importance) > 0:
        top_idx        = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_idx]
        top_kmers      = [
            kmer_names[i] if i < len(kmer_names) else f'kmer_{i}'
            for i in top_idx
        ]
        axes[1, 1].barh(range(len(top_importance)), top_importance, color='teal')
        axes[1, 1].set_yticks(range(len(top_importance)))
        axes[1, 1].set_yticklabels(top_kmers, fontsize=7)
        axes[1, 1].set_title('Top 20 K-mers (Gradient Importance)')
        axes[1, 1].set_xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig(plots_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plots_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    num_workers = configure_pytorch_threads()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        num_workers = 0
        print("MPS device detected: forcing num_workers=0")

    # Load k-mer dataset
    # Key fix: step 12 saves feature names as 'feature_names', not 'kmer_names'
    kmer_data      = np.load(snakemake.input.train, allow_pickle=True)
    kmer_test_data = np.load(snakemake.input.test,  allow_pickle=True)

    X_train          = kmer_data['X']
    y_train          = kmer_data['y']
    train_sample_ids = kmer_data['sample_ids'] if 'sample_ids' in kmer_data else None
    kmer_names       = (kmer_data['feature_names'].tolist()
                        if 'feature_names' in kmer_data else [])

    X_test = kmer_test_data['X']
    y_test = kmer_test_data['y']

    # Geographic-temporal groups
    location_year_train = None
    if train_sample_ids is not None:
        try:
            antibiotic = None
            for ab in ['amikacin', 'ciprofloxacin', 'ceftazidime', 'meropenem']:
                if ab in str(snakemake.input.train):
                    antibiotic = ab
                    break

            if antibiotic:
                meta_path = str(snakemake.input.train).replace(
                    f'deep_models/{antibiotic}_kmer_train_final.npz',
                    f'tree_models/{antibiotic}_train_final.csv'
                )
                meta = pd.read_csv(meta_path)
                meta['loc_year'] = (
                    meta['Location'].fillna('unknown').astype(str) + '_' +
                    meta['Year'].fillna('unknown').astype(str)
                )
                sid_to_group = dict(zip(meta['sample_id'], meta['loc_year']))
                location_year_train = np.array([
                    sid_to_group.get(sid, 'unknown') for sid in train_sample_ids
                ])
                print(f"Location-year groups: {len(np.unique(location_year_train))} unique")
        except Exception as e:
            print(f"Warning: Could not load location-year info: {e}")

    print(f"\n=== 1D-CNN TRAINING ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}")
    print(f"K-mer features:   {X_train.shape[1]}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Test class dist:  {np.bincount(y_test)}")
    if kmer_names:
        print(f"Example features: {kmer_names[:5]}")

    model_params = {
        'epochs':        snakemake.params.get('epochs', 100),
        'batch_size':    snakemake.params.get('batch_size', 64),
        'learning_rate': snakemake.params.get('learning_rate', 0.0005),
        'dropout':       snakemake.params.get('dropout', 0.3),
        'weight_decay':  snakemake.params.get('weight_decay', 1e-4),
        'patience':      snakemake.params.get('patience', 15),
    }

    cv_folds          = snakemake.params.get('cv_folds', 5)
    random_state      = snakemake.params.get('random_state', 42)
    use_geographic_cv = snakemake.config.get('models', {}).get('use_geographic_cv', True)

    print(f"\nRunning {cv_folds}-fold cross-validation...")
    cv_results, fold_models = cross_validation(
        X_train, y_train, location_year_train,
        cv_folds, random_state, num_workers,
        use_geographic_cv, **model_params
    )

    print("\nTraining final model (with early-stopping holdout)...")
    (final_model, test_results, feature_importance,
     test_preds_list, test_probs_list, test_labels_list) = train_final_model(
        X_train, y_train, X_test, y_test, num_workers, **model_params
    )

    results = {
        'cv_results':   cv_results,
        'test_results': test_results,
        'cv_mean_f1':   float(np.mean([r['f1']  for r in cv_results])),
        'cv_std_f1':    float(np.std( [r['f1']  for r in cv_results])),
        'cv_mean_auc':  float(np.mean([r['auc'] for r in cv_results])),
        'cv_std_auc':   float(np.std( [r['auc'] for r in cv_results])),
        'model_params': model_params,
        'n_features':   int(X_train.shape[1]),
        'test_predictions': {
            'y_true':  test_labels_list,
            'y_pred':  test_preds_list,
            'y_proba': test_probs_list
        },
        'feature_importance_top20': {
            'indices': np.argsort(feature_importance)[-20:].tolist(),
            'scores':  feature_importance[np.argsort(feature_importance)[-20:]].tolist(),
            'kmers':   [
                kmer_names[i] if i < len(kmer_names) else f'kmer_{i}'
                for i in np.argsort(feature_importance)[-20:]
            ]
        }
    }

    with open(snakemake.output.results, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(final_model.state_dict(), snakemake.output.model)

    importance_df = pd.DataFrame({
        'kmer': (kmer_names[:len(feature_importance)]
                 if len(kmer_names) >= len(feature_importance)
                 else [f'kmer_{i}' for i in range(len(feature_importance))]),
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(snakemake.output.importance, index=False)

    create_visualizations(
        cv_results, test_results, feature_importance,
        kmer_names, snakemake.output.plots
    )

    print(f"\n=== RESULTS SUMMARY ===")
    print(f"CV F1:    {results['cv_mean_f1']:.3f} ± {results['cv_std_f1']:.3f}")
    print(f"CV AUC:   {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    print(f"Test F1:  {test_results['f1']:.3f}")
    print(f"Test AUC: {test_results['auc']:.3f}")
    print(f"Best epoch: {test_results.get('best_epoch', 'N/A')}")


if __name__ == "__main__":
    main()
