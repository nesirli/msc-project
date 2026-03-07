"""Unit tests for dl_training.py shared utilities module."""

import pytest
import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Only import torch-dependent functions if torch is available
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass

from utils.dl_training import configure_pytorch_threads, get_device


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestConfigurePytorchThreads:
    def test_returns_num_workers(self):
        # Without snakemake context, should return fallback
        num_workers = configure_pytorch_threads()
        assert isinstance(num_workers, int)
        assert num_workers >= 1


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestGetDevice:
    def test_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        # On most test machines, should be CPU
        assert device.type in ('cpu', 'cuda', 'mps')


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestComputeClassWeightsTensor:
    @pytest.mark.parametrize(
        "labels, check",
        [
            pytest.param(
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                lambda w: abs(w[0].item() - w[1].item()) < 0.1,
                id="balanced",
            ),
            pytest.param(
                np.array([0] * 90 + [1] * 10),
                lambda w: w[1].item() > w[0].item(),
                id="imbalanced",
            ),
            pytest.param(
                np.array([0, 0, 0, 0]),
                lambda w: len(w) >= 1,
                id="single_class",
            ),
        ],
    )
    def test_class_weights(self, labels, check):
        from utils.dl_training import compute_class_weights_tensor
        weights = compute_class_weights_tensor(labels)
        assert isinstance(weights, torch.Tensor)
        assert check(weights)


class TestLogFoldInfo:
    def test_does_not_crash(self, capsys):
        from utils.dl_training import log_fold_info
        y_train = np.array([0, 0, 0, 1, 1])
        y_val = np.array([0, 1, 1])
        log_fold_info(fold=0, cv_folds=5, y_train_fold=y_train, y_val_fold=y_val)
        captured = capsys.readouterr()
        assert 'Fold 1' in captured.out
