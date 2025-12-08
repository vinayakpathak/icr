"""
Regression analysis to predict OOD score from gradient features.
"""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def load_analysis_results(input_file: str = "analysis_results.json") -> list:
    """
    Load analysis results from JSON file.
    
    Args:
        input_file: Path to JSON file
    
    Returns:
        List of result dictionaries
    """
    with open(input_file, 'r') as f:
        results = json.load(f)
    return results


def extract_features_and_target(results: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract gradient features and OOD score target from results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Tuple of (features, target) where:
        - features: (n_samples, 6) array of gradient values
        - target: (n_samples,) array of OOD scores
    """
    features = []
    target = []
    
    for result in results:
        gradients = result["gradients"]
        feature_vector = [
            gradients["mu_x"],
            gradients["sigma_x"],
            gradients["mu_theta"],
            gradients["sigma_theta"],
            gradients["mu_noise"],
            gradients["sigma_noise"],
        ]
        features.append(feature_vector)
        target.append(result["ood_score"])
    
    return np.array(features), np.array(target)


def fit_regression_model(
    features: np.ndarray,
    target: np.ndarray,
    model_type: str = "linear",
    alpha: Optional[float] = None,
) -> tuple:
    """
    Fit a regression model to predict OOD score from gradients.
    
    Args:
        features: (n_samples, n_features) array of gradient values
        target: (n_samples,) array of OOD scores
        model_type: Type of regression model ("linear", "ridge", "lasso")
        alpha: Regularization strength (for ridge/lasso)
    
    Returns:
        Tuple of (model, predictions, r2, mse, feature_names)
    """
    feature_names = ["mu_x", "sigma_x", "mu_theta", "sigma_theta", "mu_noise", "sigma_noise"]
    
    # Create model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        if alpha is None:
            alpha = 1.0
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        if alpha is None:
            alpha = 1.0
        model = Lasso(alpha=alpha)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Fit model
    model.fit(features, target)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Compute metrics
    r2 = r2_score(target, predictions)
    mse = mean_squared_error(target, predictions)
    
    return model, predictions, r2, mse, feature_names


def plot_regression_results(
    target: np.ndarray,
    predictions: np.ndarray,
    model_type: str,
    r2: float,
    mse: float,
    output_file: str = "plots/regression_analysis.png",
) -> None:
    """
    Create visualization of regression results.
    
    Args:
        target: True OOD scores
        predictions: Predicted OOD scores
        model_type: Type of regression model
        r2: R² score
        mse: Mean squared error
        output_file: Path to save plot
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: predicted vs actual
    ax1 = axes[0]
    ax1.scatter(target, predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(target.min(), predictions.min())
    max_val = max(target.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax1.set_xlabel("Actual OOD Score", fontsize=12)
    ax1.set_ylabel("Predicted OOD Score", fontsize=12)
    ax1.set_title(f"{model_type.capitalize()} Regression: Predicted vs Actual\nR² = {r2:.4f}, MSE = {mse:.6f}", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2 = axes[1]
    residuals = target - predictions
    ax2.scatter(predictions, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel("Predicted OOD Score", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_title("Residuals Plot", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list[str],
    output_file: str = "plots/feature_importance.png",
) -> None:
    """
    Plot feature importance (coefficients) from regression model.
    
    Args:
        model: Fitted regression model
        feature_names: Names of features
        output_file: Path to save plot
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    coefficients = model.coef_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute value
    indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefs = coefficients[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    colors = ['red' if c < 0 else 'blue' for c in sorted_coefs]
    ax.barh(range(len(sorted_coefs)), sorted_coefs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_coefs)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Feature Importance (Regression Coefficients)", fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to {output_file}")
    plt.close()


def main(
    input_file: str = "analysis_results.json",
    output_plot: str = "plots/regression_analysis.png",
    output_importance: str = "plots/feature_importance.png",
    model_type: str = "linear",
    alpha: Optional[float] = None,
) -> None:
    """
    Main function to perform regression analysis.
    
    Args:
        input_file: Input JSON file with analysis results
        output_plot: Output plot file for regression results
        output_importance: Output plot file for feature importance
        model_type: Type of regression model ("linear", "ridge", "lasso")
        alpha: Regularization strength (for ridge/lasso)
    """
    print(f"Loading analysis results from {input_file}")
    results = load_analysis_results(input_file)
    
    if len(results) == 0:
        print("No results found in input file")
        return
    
    print(f"Loaded {len(results)} results")
    
    # Extract features and target
    print("Extracting features and target...")
    features, target = extract_features_and_target(results)
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    
    # Fit regression model
    print(f"\nFitting {model_type} regression model...")
    model, predictions, r2, mse, feature_names = fit_regression_model(
        features, target, model_type=model_type, alpha=alpha
    )
    
    print(f"\nRegression Results:")
    print(f"  Model type: {model_type}")
    print(f"  R² score: {r2:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Print coefficients
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.6f}")
    if hasattr(model, 'intercept_'):
        print(f"  intercept: {model.intercept_:.6f}")
    
    # Create plots
    print(f"\nCreating plots...")
    plot_regression_results(target, predictions, model_type, r2, mse, output_plot)
    plot_feature_importance(model, feature_names, output_importance)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit regression model to predict OOD score from gradients")
    parser.add_argument("--input", type=str, default="analysis_results.json", help="Input JSON file")
    parser.add_argument("--output_plot", type=str, default="plots/regression_analysis.png", help="Output plot file")
    parser.add_argument("--output_importance", type=str, default="plots/feature_importance.png", help="Output feature importance plot")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "ridge", "lasso"], help="Regression model type")
    parser.add_argument("--alpha", type=float, default=None, help="Regularization strength (for ridge/lasso)")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        output_plot=args.output_plot,
        output_importance=args.output_importance,
        model_type=args.model_type,
        alpha=args.alpha,
    )

