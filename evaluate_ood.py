"""
OOD evaluation function for in-context linear regression models.
Reuses existing evaluation code from ic_regression.py.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from ic_regression import (
    ICLinearRegressionTransformer,
    ICRegConfig,
    InContextLinearRegressionDataset,
    evaluate_ic_regression,
)


def evaluate_ood_score(
    model: ICLinearRegressionTransformer,
    cfg: ICRegConfig,
    n_samples: int = 10000,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> float:
    """
    Evaluate OOD score (M=inf) for a model.
    
    Creates a dataset with M="inf" (Gaussian task prior: theta ~ N(0, I_D))
    and evaluates the model's MSE loss on this dataset.
    
    Args:
        model: The trained model to evaluate
        cfg: Configuration object
        n_samples: Number of samples for evaluation
        batch_size: Batch size for evaluation
        device: Device to use (uses cfg.device if None)
    
    Returns:
        Average MSE loss (OOD score)
    """
    if device is None:
        device = cfg.device
    
    # Create OOD dataset with M="inf" (Gaussian task prior)
    ood_dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=None,  # ignore tasks, sample t ~ N(0, I_D)
        M="inf",
        num_samples=n_samples,
        device="cpu",  # Always CPU in dataset
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    # Evaluate using existing function
    ood_score = evaluate_ic_regression(model, cfg, ood_loader, device)
    
    return ood_score

