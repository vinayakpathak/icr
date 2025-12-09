# In-Context Linear Regression (ICR)

A PyTorch implementation for studying in-context learning in transformers applied to linear regression tasks. This project investigates how task diversity affects the learning dynamics and generalization capabilities of transformer models.

## Overview

This project implements transformer-based models that learn to perform linear regression in-context. Given a sequence of (x, y) pairs from a linear regression task, the model learns to predict y values for new x inputs without explicit parameter updates. The main research question is how the diversity of training tasks (parameterized by M) affects model performance and learning dynamics.

### Key Features

- **Transformer-based in-context learning**: Implements a transformer architecture specifically designed for in-context linear regression
- **Task diversity experiments**: Systematically studies models trained with different levels of task diversity (M values from 2^1 to 2^20)
- **Comprehensive evaluation**: Includes in-distribution and out-of-distribution (OOD) evaluation
- **Gradient analysis**: Tools for analyzing gradient dynamics during training
- **Checkpoint management**: Automatic checkpointing and resuming capabilities
- **Early stopping**: Configurable early stopping based on validation loss

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd icr
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependency is PyTorch (>=2.0.0) with CUDA support.

## Project Structure

```
icr/
├── ic_regression.py              # Core implementation: model, training, dataset
├── run_m_diversity_experiment.py # Main experiment runner for M diversity studies
├── predict.py                    # Functions for loading models and making predictions
├── evaluate_ood.py               # Out-of-distribution evaluation
├── analyze_all_checkpoints.py    # Batch analysis of all checkpoints
├── compute_gradients.py          # Gradient computation and analysis
├── gradient_analysis.py          # Gradient analysis utilities
├── pca_gradients.py              # PCA analysis of gradients
├── regression_analysis.py        # Regression analysis of results
├── test_predict.py               # Unit tests for prediction functions
├── run_experiment.sh             # Shell script wrapper for experiments
├── requirements.txt              # Python dependencies
├── checkpoints/                  # Saved model checkpoints (created during training)
├── plots/                        # Generated plots and visualizations
└── icl-task-diversity/          # Additional ICL task diversity code
```

## Quick Start

### Training a Single Model

Train a model with a specific task diversity M:

```python
from ic_regression import ICRegConfig, train_ic_regression

cfg = ICRegConfig()
train_ic_regression(
    cfg=cfg,
    M=64,  # Task diversity: uniform over 64 tasks
    num_steps=150000,
    batch_size=2048,
    learning_rate=1e-3,
    checkpoint_dir="checkpoints/model_M64"
)
```

### Running M Diversity Experiments

Train multiple models with exponentially increasing M values (2^1 to 2^20):

```bash
python run_m_diversity_experiment.py \
    --max_power 20 \
    --num_steps 500000 \
    --batch_size 2048 \
    --learning_rate 1e-3 \
    --warmup_steps 50000 \
    --checkpoint_every 50000 \
    --eval_every 10000 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.001
```

### Running in Background (Survives SSH Disconnection)

Use `nohup` to run experiments that continue even after SSH disconnection:

```bash
nohup python run_m_diversity_experiment.py \
    --max_power 20 \
    --num_steps 500000 \
    --batch_size 2048 \
    --learning_rate 1e-3 \
    --warmup_steps 50000 \
    --checkpoint_every 50000 \
    --eval_every 10000 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.001 \
    > experiment_output.log 2>&1 &
```

Or use the provided shell script:

```bash
./run_experiment.sh --max_power 20 --num_steps 500000
```

Monitor progress:
```bash
tail -f experiment_output.log
```

## Configuration

### Model Configuration (`ICRegConfig`)

- `D`: Task dimension (default: 8)
- `K`: Number of (x, y) pairs per sequence (default: 16)
- `sigma2`: Noise variance (default: 0.125)
- `d_model`: Transformer model dimension (default: 512)
- `d_mlp`: MLP dimension (default: 512)
- `n_heads`: Number of attention heads (default: 4)
- `n_layers`: Number of transformer layers (default: 3)
- `use_prenorm`: Use pre-layer normalization (default: True)
- `M`: Task diversity - integer for uniform over {t_1,...,t_M}, "inf" for Gaussian (default: 64)
- `max_M`: Maximum number of discrete tasks to pre-sample (default: 32768)

### Training Hyperparameters

- `num_steps`: Number of training steps (default: 150,000)
- `batch_size`: Batch size (default: 2048)
- `learning_rate`: Learning rate (default: 1e-3)
- `warmup_steps`: Number of warmup steps for triangle LR schedule (default: None = constant LR)
- `grad_clip`: Gradient clipping value (default: 1.0)
- `checkpoint_every`: Save checkpoint every N steps (default: None = only final)
- `eval_every`: Evaluate OOD every N steps (default: None = no evaluation)
- `print_every`: Print loss every N steps (default: 1000)

### Early Stopping

- `early_stopping_patience`: Stop if loss doesn't improve for N evaluations (default: None)
- `early_stopping_min_delta`: Minimum change to qualify as improvement (default: 1e-6)

**Note**: Early stopping requires `eval_every` to be set.

## Usage Examples

### Making Predictions

```python
from predict import load_model, predict_from_prompt
from ic_regression import ICRegConfig
import torch

# Load a trained model
cfg = ICRegConfig()
model = load_model("checkpoints/checkpoints_M64/checkpoint_step_150000.pt", cfg)

# Create a prompt with context examples
x_context = torch.randn(5, 8)  # 5 examples, dimension 8
y_context = torch.randn(5)     # 5 target values
x_query = torch.randn(3, 8)    # 3 query points

# Make predictions
y_pred = predict_from_prompt(model, x_context, y_context, x_query)
print(f"Predictions: {y_pred}")
```

### Evaluating Out-of-Distribution Performance

```python
from evaluate_ood import evaluate_ood_score
from ic_regression import ICRegConfig, load_checkpoint

cfg = ICRegConfig()
model, _ = load_checkpoint("checkpoints/checkpoints_M64/checkpoint_step_150000.pt", cfg)

# Evaluate on Gaussian task prior (M="inf")
ood_score = evaluate_ood_score(model, cfg, n_samples=10000)
print(f"OOD MSE: {ood_score}")
```

### Analyzing All Checkpoints

Analyze all checkpoints across different M values:

```bash
python analyze_all_checkpoints.py \
    --checkpoint_base_dir checkpoints \
    --output_file analysis_results.json \
    --n_prompts 1000 \
    --n_ood_samples 10000
```

### Gradient Analysis

Compute and analyze gradients:

```bash
python compute_gradients.py \
    --checkpoint_dir checkpoints/checkpoints_M64 \
    --output_file gradient_results.json
```

### Regression Analysis

Perform regression analysis on results:

```bash
python regression_analysis.py \
    --input_file analysis_results.json \
    --output_plot plots/regression_analysis.png \
    --model_type ridge
```

## Experiment Workflow

1. **Train models**: Run `run_m_diversity_experiment.py` to train models with different M values
2. **Analyze checkpoints**: Use `analyze_all_checkpoints.py` to evaluate all trained models
3. **Compute gradients**: Use `compute_gradients.py` to analyze gradient dynamics
4. **Regression analysis**: Use `regression_analysis.py` to understand relationships between M and performance

## Monitoring Experiments

### Check if experiment is running:
```bash
ps aux | grep run_m_diversity_experiment
```

### View live output:
```bash
tail -f experiment_output.log
```

### View recent output:
```bash
tail -50 experiment_output.log
```

### Check checkpoint progress:
```bash
ls -lh checkpoints/checkpoints_M*/
```

## Output Files

- **Checkpoints**: Saved in `checkpoints/checkpoints_M{M}/` directories
- **Experiment metadata**: `experiment_metadata.json` (created after experiment completes)
- **Analysis results**: JSON files with evaluation metrics
- **Plots**: Saved in `plots/` directory

## Task Diversity (M)

The parameter M controls task diversity:

- **M = integer**: Tasks are sampled uniformly from a set of M pre-sampled tasks {t_1, ..., t_M}
- **M = "inf"**: Tasks are sampled from a continuous Gaussian distribution N(0, I_D)

Higher M values correspond to greater task diversity. The experiment runner trains models with M = 2^1, 2^2, ..., 2^max_power to study how task diversity affects learning.

## Citation

If you use this code in your research, please cite the relevant papers on in-context learning and task diversity.

## License

[Specify your license here]

## Contributing

[Contributing guidelines if applicable]

## Contact

[Contact information if applicable]

