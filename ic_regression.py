import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# =====================
#  Config
# =====================

@dataclass
class ICRegConfig:
    D: int = 8            # task dimension
    K: int = 16           # number of (x, y) pairs per sequence
    sigma2: float = 0.25  # noise variance (matching reference: noise_scale=0.5 -> sigma2=0.25)
    d_model: int = 512
    d_mlp: int = 512
    n_heads: int = 4
    n_layers: int = 8     # number of transformer blocks (matching reference)
    use_prenorm: bool = True  # True for pre-layer-norm, False for post-layer-norm (GPT2 style)
    M: Union[int, str] = 64  # task diversity: int for uniform over {t_1,...,t_M}, "inf" for Gaussian
    max_M: int = 33554432  # max number of discrete tasks to pre-sample (2^25)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
#  Data: tasks + sequences
# =====================

def sample_task_sequence(max_M: int, D: int, seed: int = 0) -> torch.Tensor:
    """
    Sample an infinite-sequence approximation t_1,...,t_max_M ~ N(0, I_D).
    These are shared across all Ms to get nested task sets.
    Shape: (max_M, D)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(max_M, D, generator=g)


def encode_sequence_tokens(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Encode a sequence S = (x_1, y_1, ..., x_K, y_K) into 2K tokens of dim D+1.

    x: (K, D)
    y: (K,)

    Token layout (per k, 0-based indexing):
      token[2k]   = (0, x_k)        # "x token"
      token[2k+1] = (y_k, 0,...,0)  # "y token"

    We'll *predict* y_k from the transformer output at the x-token positions,
    so the model never gets to see y_k when it predicts it.
    """
    K, D = x.shape
    tokens = torch.zeros(2 * K, D + 1, dtype=x.dtype)
    # x tokens: first coord = 0, remaining D coords = x
    tokens[0::2, 1:] = x
    # y tokens: first coord = y_k, remaining D coords = 0
    tokens[1::2, 0] = y
    return tokens


class InContextLinearRegressionDataset(Dataset):
    """
    Dataset for in-context linear regression as in the paper.

    For a fixed M:

    - If 1 <= M <= max_M: sample t ~ Uniform({t_1,...,t_M}).
    - If M is None or 'inf': sample t ~ N(0, I_D) directly (Gaussian task prior).

    Each item:
      tokens: (2K, D+1)
      y:      (K,)
    """

    def __init__(
        self,
        cfg: ICRegConfig,
        tasks: Optional[torch.Tensor],
        M: Optional[Union[int, str]],
        num_samples: int = 100_000,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        # Keep tasks on CPU for dataset (will be moved to device in training loop)
        self.tasks = tasks.cpu() if tasks is not None else None
        self.M = M
        self.num_samples = num_samples
        # Always use CPU in dataset - tensors will be moved to device in training loop
        self.device = "cpu"

        if isinstance(M, str):
            assert M == "inf"
            self.M_int = None
        else:
            self.M_int = M
            # Validate that M doesn't exceed max_M
            if self.M_int > cfg.max_M:
                raise ValueError(
                    f"M={self.M_int} exceeds max_M={cfg.max_M}. "
                    f"Increase max_M or use M='inf' for Gaussian task prior."
                )

        if self.tasks is not None:
            assert self.tasks.shape[1] == cfg.D
            # Additional validation: if M is an integer, ensure we have enough tasks
            if self.M_int is not None and self.tasks.shape[0] < self.M_int:
                raise ValueError(
                    f"Not enough pre-sampled tasks: have {self.tasks.shape[0]}, "
                    f"but M={self.M_int} requires at least {self.M_int} tasks."
                )

    def __len__(self):
        return self.num_samples

    def _sample_task(self) -> torch.Tensor:
        D = self.cfg.D
        if self.tasks is None or self.M_int is None:
            # "infinite" task diversity: sample t ~ N(0, I_D)
            return torch.randn(D)  # Always on CPU
        else:
            # Sample uniformly from {t_1, ..., t_M}
            idx = torch.randint(0, self.M_int, (1,))
            return self.tasks[idx].squeeze(0)  # Tasks are on CPU

    def __getitem__(self, idx):
        D, K, sigma2 = self.cfg.D, self.cfg.K, self.cfg.sigma2

        t = self._sample_task()  # (D,) on CPU
        x = torch.randn(K, D)  # Always on CPU
        noise = torch.randn(K) * math.sqrt(sigma2)  # Always on CPU
        y = x @ t + noise  # (K,) on CPU

        tokens = encode_sequence_tokens(x, y)  # (2K, D+1) on CPU

        return tokens, y


# =====================
#  Transformer blocks
# =====================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, use_prenorm: bool = True):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        if self.use_prenorm:
            # Pre-layer-norm: norm before attention/MLP
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
            x = x + attn_out
            h = self.ln2(x)
            h = self.mlp(h)
            x = x + h
        else:
            # Post-layer-norm (GPT2 style): norm after residual connection
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
            x = self.ln1(x)
            h = self.mlp(x)
            x = x + h
            x = self.ln2(x)
        return x


class ICLinearRegressionTransformer(nn.Module):
    """
    2-layer decoder-only transformer with learnable positional embeddings.

    Input:  tokens of shape (B, 2K, D+1)
    Output: same shape, then we read predictions from specific positions.
    """

    def __init__(self, cfg: ICRegConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D
        K = cfg.K
        d_model = cfg.d_model

        self.seq_len = 2 * K
        self.token_dim = D + 1

        # Token embedding (linear in, no discrete vocab here)
        self.input_proj = nn.Linear(self.token_dim, d_model)

        # Learned positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        # Transformer blocks (L layers, each with attention + MLP)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, cfg.n_heads, cfg.d_mlp, cfg.use_prenorm)
             for _ in range(cfg.n_layers)]
        )

        # Final layer norm + projection back to token_dim
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.token_dim)

        # Causal mask (prevent attending to future tokens)
        # For PyTorch's MultiheadAttention: attn_mask[i, j] = True means position i cannot attend to position j
        # We want a lower triangular mask (can attend to past and present, not future)
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("attn_mask", mask)

        self._init_parameters()

    def _init_parameters(self):
        # Initialization matching GPT2/nanoGPT: normal(0, 0.02) for embeddings and linear layers
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

        # Initialize MLP layers with normal(0, 0.02)
        for block in self.blocks:
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, 2K, D+1)
        returns: (B, 2K, D+1)
        """
        B, T, dim = tokens.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert dim == self.token_dim, f"Expected token_dim={self.token_dim}, got {dim}"

        x = self.input_proj(tokens)  # (B, T, d_model)
        x = x + self.pos_emb[:, :T, :]

        for block in self.blocks:
            x = block(x, attn_mask=self.attn_mask)

        x = self.ln_f(x)
        out = self.output_proj(x)  # (B, T, D+1)
        return out

    def predict_y_from_x_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Given input tokens (B, 2K, D+1), run the transformer and
        return predictions \hat{y}_k for k = 1..K.

        We interpret:
          - x tokens are at positions 0,2,4,...,2K-2
          - For each k, we take the output at the x-token position (2k),
            and its first coordinate is \hat{y}_k.

        This matches f(S_{<=k}) seeing only the context (x_1, y_1, ..., x_{k-1}, y_{k-1}, x_k).
        """
        out_tokens = self(tokens)  # (B, 2K, D+1)
        # Take outputs at x-token positions (0,2,4,...)
        x_positions = out_tokens[:, 0::2, :]  # (B, K, D+1)
        y_pred = x_positions[..., 0]          # first coord => (B, K)
        return y_pred


# =====================
#  Training loop
# =====================

def train_ic_regression(
    cfg: ICRegConfig,
    M: Optional[Union[int, str]] = None,
    num_steps: int = 150_000,
    batch_size: int = 1024,
    print_every: int = 1000,
    eval_every: Optional[int] = None,
    learning_rate: float = 1e-3,  # Default matching reference implementation
    grad_clip: Optional[float] = 1.0,
    warmup_steps: Optional[int] = None,  # If None, uses constant LR; if set, uses triangle schedule
    skip_first_prediction: bool = False,  # Reference implementation computes loss on all predictions
    checkpoint_dir: Optional[str] = "checkpoints",  # Directory to save checkpoints
    checkpoint_every: Optional[int] = None,  # Save checkpoint every N steps (None = only at end)
    early_stopping_patience: Optional[int] = None,  # Stop if loss doesn't improve for N evaluations (None = no early stopping)
    early_stopping_min_delta: float = 1e-6,  # Minimum change to qualify as improvement
    early_stopping_eval_every: Optional[int] = None,  # Evaluate for early stopping every N steps (defaults to print_every)
):
    """
    Basic training loop for a single task diversity M.

    - M integer  => Uniform over {t_1,...,t_M}
    - M == 'inf' => Gaussian task prior N(0, I_D)
    - If M is None, uses cfg.M from config

    This matches the high-level setup of Section 3 and 4 in the paper.
    """

    device = cfg.device
    # Use M from parameter if provided, otherwise use cfg.M
    M = M if M is not None else cfg.M

    # Create checkpoint directory
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Pre-sample nested tasks (shared across all M if you re-use)
    # Keep on CPU for dataset - will be moved to device in training loop
    tasks = sample_task_sequence(cfg.max_M, cfg.D)

    # Build dataset & dataloader
    # Dataset always uses CPU - tensors moved to device in training loop
    dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=tasks,
        M=M,
        num_samples=1_000_000,  # effectively infinite
        device="cpu",  # Always CPU in dataset
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,  # Parallel data loading (increased for better throughput)
        pin_memory=True if device.startswith("cuda") else False,  # Faster GPU transfer
        persistent_workers=True if device.startswith("cuda") else False,  # Keep workers alive between epochs
    )

    # Validation dataset for early stopping (same distribution as training)
    if early_stopping_patience is not None:
        val_dataset = InContextLinearRegressionDataset(
            cfg=cfg,
            tasks=tasks,  # Same tasks as training
            M=M,  # Same M as training
            num_samples=10_000,  # Validation set size
            device="cpu",  # Always CPU in dataset
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,  # Parallel data loading for validation
            pin_memory=True if device.startswith("cuda") else False,
        )
    else:
        val_loader = None

    # Optional: OOD eval dataset with M = 'inf'
    if eval_every is not None:
        ood_dataset = InContextLinearRegressionDataset(
            cfg=cfg,
            tasks=None,   # ignore tasks, sample t ~ N(0, I_D)
            M="inf",
            num_samples=10_000,
            device="cpu",  # Always CPU in dataset
        )
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,  # Parallel data loading for OOD evaluation
            pin_memory=True if device.startswith("cuda") else False,
        )
    else:
        ood_loader = None

    # Model
    model = ICLinearRegressionTransformer(cfg).to(device)

    # Optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate schedule: triangle (warmup + linear decay) if warmup_steps is set
    if warmup_steps is not None:
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup: linear increase from 0 to learning_rate
                return step / warmup_steps
            else:
                # Linear decay from learning_rate to 0
                decay_steps = num_steps - warmup_steps
                return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Print device information
    print(f"Training on device: {device}")
    if device.startswith("cuda"):
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    if warmup_steps is not None:
        print(f"  Using triangle LR schedule: warmup={warmup_steps} steps, then linear decay")
    else:
        print(f"  Using constant LR: {learning_rate}")

    step = 0
    running_loss = 0.0

    data_iter = iter(dataloader)

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_eval_interval = early_stopping_eval_every if early_stopping_eval_every is not None else print_every
    recent_losses = []  # Track recent losses for convergence detection

    model.train()
    while step < num_steps:
        try:
            tokens, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens, y = next(data_iter)

        tokens = tokens.to(device)  # (B, 2K, D+1)
        y = y.to(device)            # (B, K)

        optimizer.zero_grad()
        y_pred = model.predict_y_from_x_tokens(tokens)  # (B, K)
        
        # Optionally skip first prediction (k=1) which has no context
        if skip_first_prediction and y_pred.shape[1] > 1:
            # Only compute loss on predictions k=2,...,K (indices 1,...,K-1)
            loss = F.mse_loss(y_pred[:, 1:], y[:, 1:])
        else:
            loss = F.mse_loss(y_pred, y)
        
        loss.backward()
        
        # Gradient clipping to prevent instability
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update learning rate schedule
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"[step {step}] train M={M}, loss={avg_loss:.4f}")
            running_loss = 0.0
            
            # Early stopping check (using validation loss from same distribution)
            if early_stopping_patience is not None and step % early_stopping_eval_interval == 0:
                # Evaluate on fresh validation set
                model.eval()
                val_loss = evaluate_ic_regression(model, cfg, val_loader, device)
                model.train()
                
                recent_losses.append(val_loss)
                # Keep only last few losses for comparison
                if len(recent_losses) > 10:
                    recent_losses.pop(0)
                
                # Check if validation loss improved
                if val_loss < best_loss - early_stopping_min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Check convergence: if validation loss hasn't improved for patience steps
                if patience_counter >= early_stopping_patience:
                    print(f"[step {step}] Early stopping: validation loss hasn't improved for {patience_counter} evaluations")
                    print(f"  Best validation loss: {best_loss:.6f}, Current validation loss: {val_loss:.6f}")
                    break

        if eval_every is not None and step % eval_every == 0:
            eval_loss = evaluate_ic_regression(model, cfg, ood_loader, device)
            print(f"[step {step}] OOD eval (M=inf) loss={eval_loss:.4f}")

        # Save checkpoint periodically
        if checkpoint_dir is not None and checkpoint_every is not None and step % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, cfg, M)
            print(f"[step {step}] Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    if checkpoint_dir is not None:
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, step, cfg, M)
        print(f"Saved final checkpoint to {final_checkpoint_path}")

    return model


@torch.no_grad()
def evaluate_ic_regression(
    model: ICLinearRegressionTransformer,
    cfg: ICRegConfig,
    dataloader: DataLoader,
    device: str,
) -> float:
    """
    Evaluate MSE on a provided dataloader (e.g. OOD q_∞(S)).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for tokens, y in dataloader:
        tokens = tokens.to(device)
        y = y.to(device)
        y_pred = model.predict_y_from_x_tokens(tokens)
        loss = F.mse_loss(y_pred, y, reduction="mean")
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / n_batches


# =====================
#  Checkpoint saving and loading
# =====================

def save_checkpoint(
    checkpoint_path: str,
    model: ICLinearRegressionTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    cfg: ICRegConfig,
    M: Union[int, str],
):
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler (optional)
        step: Current training step
        cfg: Configuration used for training
        M: Task diversity parameter
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
        "M": M,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> tuple[ICLinearRegressionTransformer, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], int, ICRegConfig, Union[int, str]]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (uses cfg.device if None)
    
    Returns:
        Tuple of (model, optimizer, scheduler, step, cfg, M)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = checkpoint["cfg"]
    if device is None:
        device = cfg.device
    
    # Create model and optimizer
    model = ICLinearRegressionTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # LR will be overridden by state_dict
    
    # Load state dicts
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Recreate scheduler if it existed
    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        # Note: Scheduler state is saved but we need to recreate it with the same parameters
        # For simplicity, we'll just load the state if it exists
        # In practice, you may want to save scheduler parameters separately
        pass  # Scheduler recreation would need the original parameters
    
    step = checkpoint["step"]
    M = checkpoint["M"]
    
    return model, optimizer, scheduler, step, cfg, M


def recover_training_tasks(
    max_M: int = 32768,
    D: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """
    Recover the exact tasks that were used during training.
    
    Since tasks are generated deterministically from a fixed seed, we can
    regenerate the exact same tasks that were used during training.
    
    Args:
        max_M: Maximum number of tasks to generate (default: 32768, the old value used during training)
        D: Task dimension (default: 8, or use cfg.D from a checkpoint)
        seed: Random seed used during training (default: 0)
    
    Returns:
        Tasks tensor of shape (max_M, D) - the exact tasks used during training
    
    Example:
        >>> # Recover tasks used during training
        >>> tasks = recover_training_tasks(max_M=32768, D=8, seed=0)
        >>> # For a specific M, use first M tasks: tasks[:M]
        >>> tasks_M64 = tasks[:64]  # Tasks used for M=64
    """
    return sample_task_sequence(max_M=max_M, D=D, seed=seed)


# =====================
#  Example usage
# =====================

if __name__ == "__main__":
    cfg = ICRegConfig()
    # M is set in the config (default 80), can override here if needed
    # cfg.M = 1  # Example: train on single task

    # Train with M from config
    model_M64 = train_ic_regression(
        cfg,
        num_steps=50_000,   # increased to see if loss improves; paper uses 150k
        batch_size=256,     # smaller batch to fit on modest GPUs if needed
        print_every=1_000,
        eval_every=5_000,
        warmup_steps=25_000,  # 50% warmup (matching reference: 250k warmup for 500k total)
        checkpoint_dir="checkpoints",  # Directory to save checkpoints
        checkpoint_every=10_000,  # Save checkpoint every 10k steps
    )

    # Example: Load a checkpoint for evaluation
    # model, optimizer, scheduler, step, cfg, M = load_checkpoint("checkpoints/checkpoint_final.pt")
    # model.eval()
    # # Now you can use the model for evaluation

    # Example: train at M = 'inf' (Gaussian task prior)
    # cfg.M = "inf"
    # model_Minf = train_ic_regression(cfg, num_steps=10_000)

    # Example: Load checkpoint and make predictions
    # from predict import predict_from_prompt
    # import torch
    # # Create some context examples
    # x_context = torch.randn(5, 8)  # 5 context examples, D=8
    # y_context = torch.randn(5)     # 5 context y values
    # # Create query points
    # x_query = torch.randn(3, 8)    # 3 query points
    # # Get predictions
    # y_pred = predict_from_prompt("checkpoints/checkpoint_final.pt", x_context, y_context, x_query)
