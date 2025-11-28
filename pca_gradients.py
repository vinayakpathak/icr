"""
Perform PCA on gradient vectors from checkpoints.
Each checkpoint is treated as a point in parameter-gradient space.
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path


def load_gradients(json_file: str = "gradient_results.json"):
    """Load gradients from JSON file and return as matrix."""
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Extract parameter names (should be consistent across all checkpoints)
    param_names = ['mu_x', 'sigma_x', 'mu_theta', 'sigma_theta', 'mu_noise', 'sigma_noise']
    
    # Build matrix: rows = checkpoints, columns = parameters
    gradient_matrix = []
    checkpoint_info = []
    
    for result in results:
        gradients = result['gradients']
        gradient_vector = [gradients[param] for param in param_names]
        gradient_matrix.append(gradient_vector)
        checkpoint_info.append({
            'checkpoint': result['checkpoint'],
            'step': result['step'],
            'expected_loss': result['expected_loss']
        })
    
    gradient_matrix = np.array(gradient_matrix)  # (n_checkpoints, n_parameters)
    
    return gradient_matrix, param_names, checkpoint_info


def perform_pca(gradient_matrix, n_components=None, standardize=True):
    """
    Perform PCA on gradient matrix.
    
    Args:
        gradient_matrix: (n_checkpoints, n_parameters) array
        n_components: Number of components to keep (None = all)
        standardize: Whether to standardize before PCA
    
    Returns:
        pca: Fitted PCA object
        transformed: Transformed data (n_checkpoints, n_components)
        scaler: StandardScaler if used, else None
    """
    if standardize:
        scaler = StandardScaler()
        gradient_matrix_scaled = scaler.fit_transform(gradient_matrix)
    else:
        scaler = None
        gradient_matrix_scaled = gradient_matrix
    
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(gradient_matrix_scaled)
    
    return pca, transformed, scaler


def print_pca_results(pca, param_names, checkpoint_info, transformed):
    """Print PCA results in a readable format."""
    print("=" * 80)
    print("PCA Results on Gradient Vectors")
    print("=" * 80)
    print(f"\nNumber of checkpoints: {len(checkpoint_info)}")
    print(f"Number of parameters: {len(param_names)}")
    print(f"Number of PCA components: {pca.n_components_}")
    
    print("\n" + "-" * 80)
    print("Explained Variance Ratio:")
    print("-" * 80)
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        cumsum = pca.explained_variance_ratio_[:i+1].sum()
        print(f"  PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)  [Cumulative: {cumsum:.4f} ({cumsum*100:.2f}%)]")
    
    print("\n" + "-" * 80)
    print("Principal Component Loadings (how each parameter contributes to each PC):")
    print("-" * 80)
    print(f"{'Parameter':<15} " + " ".join([f"PC{i+1:>8}" for i in range(pca.n_components_)]))
    print("-" * 80)
    for i, param in enumerate(param_names):
        loadings = pca.components_[:, i]  # Loadings for this parameter across all PCs
        print(f"{param:<15} " + " ".join([f"{val:>8.4f}" for val in loadings]))
    
    print("\n" + "-" * 80)
    print("Transformed Coordinates (checkpoints in PC space):")
    print("-" * 80)
    print(f"{'Checkpoint':<30} {'Step':<8} {'Loss':<12} " + 
          " ".join([f"PC{i+1:>10}" for i in range(min(3, pca.n_components_))]))
    print("-" * 80)
    for i, info in enumerate(checkpoint_info):
        checkpoint_name = Path(info['checkpoint']).name
        step = info['step']
        loss = info['expected_loss']
        pc_coords = transformed[i, :min(3, pca.n_components_)]
        print(f"{checkpoint_name:<30} {step:<8} {loss:<12.4f} " + 
              " ".join([f"{coord:>10.4f}" for coord in pc_coords]))
    
    print("\n" + "=" * 80)


def plot_gradient_evolution(checkpoint_info, param_names, gradient_matrix, output_file="plots/gradient_evolution.png"):
    """Plot evolution of each gradient over training steps."""
    n_params = len(param_names)
    steps = [info['step'] for info in checkpoint_info]
    
    # Create subplots: one for each parameter
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params))
    if n_params == 1:
        axes = [axes]
    
    fig.suptitle('Gradient Evolution Over Training', fontsize=14, fontweight='bold')
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        gradients = gradient_matrix[:, i]
        
        # Plot line with markers
        ax.plot(steps, gradients, marker='o', linewidth=2, markersize=8, label=param)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel(f'Gradient\n{param}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Add value annotations
        for j, (step, grad) in enumerate(zip(steps, gradients)):
            ax.annotate(f'{grad:.2f}', 
                       (step, grad),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=8,
                       alpha=0.7)
    
    # Only label x-axis on bottom plot
    axes[-1].set_xlabel('Training Step', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gradient evolution plot saved to {output_file}")


def plot_pca_results(transformed, checkpoint_info, pca, output_file="plots/pca_gradients.png"):
    """Plot PCA results."""
    n_components = transformed.shape[1]
    
    if n_components < 2:
        print("Not enough components to plot")
        return
    
    fig, axes = plt.subplots(1, min(2, n_components), figsize=(12, 5))
    if n_components == 1:
        axes = [axes]
    
    # Plot PC1 vs PC2
    if n_components >= 2:
        ax = axes[0]
        steps = [info['step'] for info in checkpoint_info]
        losses = [info['expected_loss'] for info in checkpoint_info]
        
        scatter = ax.scatter(transformed[:, 0], transformed[:, 1], 
                           c=steps, cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title('PCA of Gradient Vectors\n(Colored by Training Step)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Training Step')
        
        # Annotate points
        for i, info in enumerate(checkpoint_info):
            ax.annotate(f"Step {info['step']}", 
                       (transformed[i, 0], transformed[i, 1]),
                       fontsize=8, alpha=0.7)
    
    # Plot PC1 vs Loss
    if n_components >= 1:
        ax = axes[1] if n_components >= 2 else axes[0]
        steps = [info['step'] for info in checkpoint_info]
        losses = [info['expected_loss'] for info in checkpoint_info]
        
        scatter = ax.scatter(transformed[:, 0], losses, 
                           c=steps, cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel('Expected Loss')
        ax.set_title('PC1 vs Expected Loss\n(Colored by Training Step)')
        ax.grid(True, alpha=0.3)
        if n_components >= 2:
            plt.colorbar(scatter, ax=ax, label='Training Step')
        
        # Annotate points
        for i, info in enumerate(checkpoint_info):
            ax.annotate(f"Step {info['step']}", 
                       (transformed[i, 0], losses[i]),
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform PCA on gradient vectors")
    parser.add_argument("--input", type=str, default="gradient_results.json", 
                       help="Input JSON file with gradients")
    parser.add_argument("--n_components", type=int, default=None,
                       help="Number of PCA components (None = all)")
    parser.add_argument("--no_standardize", action="store_true",
                       help="Don't standardize before PCA")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--output_plot", type=str, default="plots/pca_gradients.png",
                       help="Output plot file")
    parser.add_argument("--output_evolution", type=str, default="plots/gradient_evolution.png",
                       help="Output gradient evolution plot file")
    
    args = parser.parse_args()
    
    # Load gradients
    print("Loading gradients...")
    gradient_matrix, param_names, checkpoint_info = load_gradients(args.input)
    print(f"Loaded {gradient_matrix.shape[0]} checkpoints with {gradient_matrix.shape[1]} parameters")
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca, transformed, scaler = perform_pca(
        gradient_matrix, 
        n_components=args.n_components,
        standardize=not args.no_standardize
    )
    
    # Print results
    print_pca_results(pca, param_names, checkpoint_info, transformed)
    
    # Plot if requested
    if args.plot:
        try:
            # Ensure plots directory exists
            import os
            os.makedirs(os.path.dirname(args.output_plot) if os.path.dirname(args.output_plot) else "plots", exist_ok=True)
            os.makedirs(os.path.dirname(args.output_evolution) if os.path.dirname(args.output_evolution) else "plots", exist_ok=True)
            
            plot_pca_results(transformed, checkpoint_info, pca, args.output_plot)
            # Also plot gradient evolution
            plot_gradient_evolution(checkpoint_info, param_names, gradient_matrix, args.output_evolution)
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            print("(This might be because matplotlib is not available in headless mode)")

