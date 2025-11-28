"""
Functions for loading checkpoints and making predictions with trained models.
"""

import math
from typing import Optional, Union, List, Tuple

import torch

from ic_regression import (
    ICLinearRegressionTransformer,
    ICRegConfig,
    encode_sequence_tokens,
    load_checkpoint,
)


def create_prompt(
    thetas: List[Tuple[torch.Tensor, int]],
    query_theta: Optional[torch.Tensor] = None,
    n_query: int = 1,
    D: int = 8,
    sigma2: float = 0.125,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a prompt from a series of thetas and example counts.
    
    Args:
        thetas: List of (theta, n_examples) tuples where:
            - theta: Task vector of shape (D,)
            - n_examples: Number of (x, y) examples to generate from this theta
        query_theta: Optional task vector for query points. If None, uses the last theta.
        n_query: Number of query points to generate
        D: Dimension of the task (default: 8)
        sigma2: Noise variance (default: 0.125)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (x_context, y_context, x_query) ready for predict_from_prompt()
    
    Example:
        >>> import torch
        >>> theta1 = torch.randn(8)
        >>> theta2 = torch.randn(8)
        >>> x_context, y_context, x_query = create_prompt(
        ...     [(theta1, 3), (theta2, 2)],  # 3 examples from theta1, 2 from theta2
        ...     query_theta=theta2,  # Query points from theta2
        ...     n_query=2
        ... )
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Collect all context examples
    x_context_list = []
    y_context_list = []
    
    for theta, n_examples in thetas:
        if theta.shape[0] != D:
            raise ValueError(f"Theta dimension {theta.shape[0]} doesn't match D={D}")
        
        # Generate n_examples from this theta
        x = torch.randn(n_examples, D)
        noise = torch.randn(n_examples) * math.sqrt(sigma2)
        y = x @ theta + noise
        
        x_context_list.append(x)
        y_context_list.append(y)
    
    # Concatenate all context examples
    x_context = torch.cat(x_context_list, dim=0)  # (total_context, D)
    y_context = torch.cat(y_context_list, dim=0)  # (total_context,)
    
    # Generate query points
    if query_theta is None:
        # Use the last theta if not specified
        query_theta = thetas[-1][0]
    
    if query_theta.shape[0] != D:
        raise ValueError(f"Query theta dimension {query_theta.shape[0]} doesn't match D={D}")
    
    x_query = torch.randn(n_query, D)
    
    return x_context, y_context, x_query


def predict_from_prompt(
    checkpoint_path: str,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    device: Optional[str] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Load a checkpoint and generate predictions for query points given context.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        x_context: Context x values, shape (K_context, D) where K_context is number of context examples
        y_context: Context y values, shape (K_context,)
        x_query: Query x values to predict, shape (K_query, D) or (D,) for single query
        device: Device to run on (uses cfg.device if None)
        verbose: Whether to print information about the prediction
    
    Returns:
        Predictions for x_query, shape (K_query,) or scalar if single query
    
    Example:
        >>> import torch
        >>> x_context = torch.randn(5, 8)  # 5 context examples, D=8
        >>> y_context = torch.randn(5)     # 5 context y values
        >>> x_query = torch.randn(3, 8)    # 3 query points
        >>> y_pred = predict_from_prompt("checkpoints/checkpoint_final.pt", x_context, y_context, x_query)
    """
    # Load checkpoint
    model, _, _, step, cfg, M = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    
    if device is None:
        device = cfg.device
    
    # Handle single query case
    single_query = x_query.dim() == 1
    if single_query:
        x_query = x_query.unsqueeze(0)  # (1, D)
    
    K_context, D = x_context.shape
    K_query, D_query = x_query.shape
    
    assert D == D_query == cfg.D, f"Dimension mismatch: context D={D}, query D={D_query}, cfg.D={cfg.D}"
    
    # The model expects exactly K examples (cfg.K), so we need to pad or truncate
    K_total = K_context + K_query
    K_expected = cfg.K
    
    if K_total > K_expected:
        # Truncate: take first K_context context examples and first (K_expected - K_context) query points
        K_context_use = min(K_context, K_expected)
        K_query_use = K_expected - K_context_use
        x_context_use = x_context[:K_context_use]
        y_context_use = y_context[:K_context_use]
        x_query_use = x_query[:K_query_use]
        if verbose and (K_context_use < K_context or K_query_use < K_query):
            print(f"Warning: Truncating to {K_expected} total examples (model expects K={K_expected})")
    else:
        # Pad with zeros to reach K_expected
        K_pad = K_expected - K_total
        x_context_use = x_context
        y_context_use = y_context
        x_query_use = x_query
        if K_pad > 0:
            x_pad = torch.zeros(K_pad, D, dtype=x_context.dtype)
            y_pad = torch.zeros(K_pad, dtype=y_context.dtype)
            x_query_use = torch.cat([x_query, x_pad], dim=0)
            if verbose:
                print(f"Padding with {K_pad} zero examples to reach K={K_expected}")
    
    # Encode the full sequence: context (x, y) pairs + query x values
    # We need to pad y_query with zeros since we don't know them yet
    y_query_placeholder = torch.zeros(x_query_use.shape[0], dtype=y_context.dtype)
    
    # Combine context and query
    x_full = torch.cat([x_context_use, x_query_use], dim=0)  # (K_expected, D)
    y_full = torch.cat([y_context_use, y_query_placeholder], dim=0)  # (K_expected,)
    
    # Encode to tokens
    tokens = encode_sequence_tokens(x_full, y_full)  # (2*K_expected, D+1)
    tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension: (1, 2*K_expected, D+1)
    
    # Get predictions
    with torch.no_grad():
        y_pred_all = model.predict_y_from_x_tokens(tokens)  # (1, K_expected)
        y_pred_all = y_pred_all.squeeze(0)  # (K_expected,)
    
    # Extract only the predictions for query points (accounting for padding/truncation)
    K_context_actual = x_context_use.shape[0]
    K_query_actual = min(K_query, K_expected - K_context_actual)
    y_pred_query = y_pred_all[K_context_actual:K_context_actual + K_query_actual]  # (K_query_actual,)
    
    # If we truncated, we only have predictions for the first K_query_actual query points
    if K_query_actual < K_query:
        if verbose:
            print(f"Warning: Only returning predictions for first {K_query_actual} query points (due to truncation)")
    
    if verbose:
        print(f"Loaded checkpoint from step {step} (M={M})")
        print(f"Context: {K_context} (x, y) pairs")
        print(f"Query: {K_query} x values to predict")
        print(f"\nContext examples:")
        for i in range(min(K_context, 5)):  # Show first 5 context examples
            x_vals = x_context[i].cpu().tolist()
            print(f"  x_{i+1} = {x_vals}, y_{i+1} = {y_context[i].item():.4f}")
        if K_context > 5:
            print(f"  ... ({K_context - 5} more)")
        print(f"\nQuery predictions:")
        for i in range(len(y_pred_query)):
            x_vals = x_query[i].cpu().tolist() if i < x_query.shape[0] else [0.0] * D
            print(f"  x_query_{i+1} = {x_vals}, y_pred = {y_pred_query[i].item():.4f}")
    
    # Return scalar if single query, otherwise tensor
    if single_query:
        return y_pred_query[0]
    return y_pred_query


if __name__ == "__main__":
    # Example usage
    import torch
    
    print("=" * 60)
    print("Demonstrating create_prompt and predict_from_prompt functions")
    print("=" * 60)
    
    # Example 1: Create prompt from multiple thetas
    torch.manual_seed(42)  # For reproducibility
    
    theta1 = torch.randn(8)  # First task
    theta2 = torch.randn(8)  # Second task
    
    print("\nCreating prompt with:")
    print(f"  - 3 examples from theta1")
    print(f"  - 2 examples from theta2")
    print(f"  - 3 query points from theta2")
    
    x_context, y_context, x_query = create_prompt(
        [(theta1, 3), (theta2, 2)],  # 3 examples from theta1, 2 from theta2
        query_theta=theta2,  # Query points from theta2
        n_query=3,
        D=8,
        sigma2=0.125,
        seed=42
    )
    
    print(f"\nGenerated context: {x_context.shape[0]} examples")
    print(f"Generated query: {x_query.shape[0]} points")
    
    print("\nUsing checkpoint: checkpoints/checkpoint_final.pt")
    print("\n" + "-" * 60)
    
    # Get predictions
    y_pred = predict_from_prompt("checkpoints/checkpoint_final.pt", x_context, y_context, x_query)
    
    print("\n" + "-" * 60)
    print(f"\nReturned predictions shape: {y_pred.shape}")
    print(f"Predictions: {y_pred.cpu().tolist()}")
    
    # Also show what the true values would be (for comparison)
    y_true = x_query @ theta2
    print(f"\nTrue values (for comparison): {y_true.tolist()}")
    if len(y_pred) == len(y_true):
        # Move y_pred to CPU if needed for comparison
        y_pred_cpu = y_pred.cpu() if y_pred.device.type == 'cuda' else y_pred
        print(f"MSE: {torch.mean((y_pred_cpu - y_true) ** 2).item():.4f}")

