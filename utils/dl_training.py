"""
Shared utilities for deep learning training scripts (1D-CNN, Sequence CNN, DNABERT).
Eliminates code duplication across scripts 16, 17, and 18.
"""

import os
import numpy as np
import torch
from collections import Counter


def configure_pytorch_threads(snakemake_obj=None):
    """
    Configure PyTorch to use all available threads from Snakemake.
    
    Args:
        snakemake_obj: Snakemake object (pass globals().get('snakemake'))
        
    Returns:
        int: Number of workers for DataLoader
    """
    try:
        if snakemake_obj is not None:
            num_threads = snakemake_obj.threads
        else:
            # Try to get from global snakemake variable
            import builtins
            num_threads = getattr(builtins, 'snakemake', None)
            if num_threads is not None:
                num_threads = num_threads.threads
            else:
                raise AttributeError("No snakemake object available")
        
        print(f"Configuring PyTorch to use {num_threads} threads")

        # Set PyTorch intraop parallelism (within operations)
        torch.set_num_threads(num_threads)

        # Set environment variables for various backends
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

        # Use approximately 1/4 of threads for data loading
        num_workers = max(1, num_threads // 4)

        return num_workers
    except (NameError, AttributeError):
        print("Warning: Not running under Snakemake, using default thread settings")
        return 2


def compute_class_weights_tensor(y_train, device=None):
    """
    Compute class weights as a PyTorch tensor for CrossEntropyLoss.
    
    Args:
        y_train: Training labels (numpy array)
        device: PyTorch device
        
    Returns:
        torch.FloatTensor of class weights
    """
    class_counts = np.bincount(y_train, minlength=2)
    total_samples = len(y_train)
    
    weights = []
    for count in class_counts:
        if count > 0:
            weights.append(total_samples / (len(class_counts) * count))
        else:
            weights.append(0.0)
    
    class_weights = torch.FloatTensor(weights)
    if device is not None:
        class_weights = class_weights.to(device)
    
    print(f"Class distribution: {class_counts}")
    print(f"Computed class weights: {class_weights.cpu().numpy()}")
    
    return class_weights


def get_device():
    """Get the best available PyTorch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def log_fold_info(fold, cv_folds, y_train_fold, y_val_fold, 
                  location_year_groups=None, train_idx=None, val_idx=None):
    """
    Log cross-validation fold information consistently.
    
    Args:
        fold: Current fold index
        cv_folds: Total number of folds
        y_train_fold: Training labels for this fold
        y_val_fold: Validation labels for this fold
        location_year_groups: Optional location-year groups array
        train_idx: Training indices
        val_idx: Validation indices
    """
    print(f"\nFold {fold + 1}/{cv_folds}")
    print(f"  Train={len(y_train_fold)}, Val={len(y_val_fold)}")
    print(f"  Train classes: {Counter(y_train_fold)}, Val classes: {Counter(y_val_fold)}")
    
    if location_year_groups is not None and train_idx is not None and val_idx is not None:
        loc_year_tr = location_year_groups[train_idx]
        loc_year_val = location_year_groups[val_idx]
        print(f"  Train location-years={len(np.unique(loc_year_tr))}, "
              f"Val location-years={len(np.unique(loc_year_val))}")
