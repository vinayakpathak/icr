import math
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
    sigma2: float = 0.125 # noise variance
    d_model: int = 512
    d_mlp: int = 512
    n_heads: int = 4
    n_layers: int = 2
    max_M: int = 32768    # max number of discrete tasks to pre-sample
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
        self.tasks = tasks  # (max_M, D) or None
        self.M = M
        self.num_samples = num_samples
        self.device = device

        if isinstance(M, str):
            assert M == "inf"
            self.M_int = None
        else:
            self.M_int = M

        if self.tasks is not None:
            assert self.tasks.shape[1] == cfg.D

    def __len__(self):
        return self.num_samples

    def _sample_task(self) -> torch.Tensor:
        D = self.cfg.D
        if self.tasks is None or self.M_int is None:
            # "infinite" task diversity: sample t ~ N(0, I_D)
            return torch.randn(D, device=self.device)
        else:
            idx = torch.randint(0, self.M_int, (1,), device=self.device)
            return self.tasks[idx].squeeze(0)

    def __getitem__(self, idx):
        D, K, sigma2 = self.cfg.D, self.cfg.K, self.cfg.sigma2

        t = self._sample_task()  # (D,)
        x = torch.randn(K, D, device=self.device)
        noise = torch.randn(K, device=self.device) * math.sqrt(sigma2)
        y = x @ t + noise  # (K,)

        tokens = encode_sequence_tokens(x, y)  # (2K, D+1)

        return tokens, y


# =====================
#  Transformer blocks
# =====================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
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
        # Pre-LN attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        # Pre-LN MLP
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
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

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, cfg.n_heads, cfg.d_mlp)
             for _ in range(cfg.n_layers)]
        )

        # Final layer norm + projection back to token_dim
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.token_dim)

        # Causal mask (prevent attending to future tokens)
        # attn_mask[i, j] = True  ==> token i cannot attend to token j
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()
        self.register_buffer("attn_mask", mask)

        self._init_parameters()

    def _init_parameters(self):
        # Simple initialization; you can tweak if you want to match exact training dynamics
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        for block in self.blocks:
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
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
    M: Optional[Union[int, str]],
    num_steps: int = 150_000,
    batch_size: int = 1024,
    print_every: int = 1000,
    eval_every: Optional[int] = None,
):
    """
    Basic training loop for a single task diversity M.

    - M integer  => Uniform over {t_1,...,t_M}
    - M == 'inf' => Gaussian task prior N(0, I_D)

    This matches the high-level setup of Section 3 and 4 in the paper.
    """

    device = cfg.device

    # Pre-sample nested tasks (shared across all M if you re-use)
    tasks = sample_task_sequence(cfg.max_M, cfg.D).to(device)

    # Build dataset & dataloader
    dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=tasks,
        M=M,
        num_samples=1_000_000,  # effectively infinite
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Optional: OOD eval dataset with M = 'inf'
    if eval_every is not None:
        ood_dataset = InContextLinearRegressionDataset(
            cfg=cfg,
            tasks=None,   # ignore tasks, sample t ~ N(0, I_D)
            M="inf",
            num_samples=10_000,
            device=device,
        )
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        ood_loader = None

    # Model
    model = ICLinearRegressionTransformer(cfg).to(device)

    # Optimizer (Adam, constant LR like the paper after warm-up)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Simple cosine annealing or constant; here constant for simplicity
    step = 0
    running_loss = 0.0

    data_iter = iter(dataloader)

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
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"[step {step}] train M={M}, loss={avg_loss:.4f}")
            running_loss = 0.0

        if eval_every is not None and step % eval_every == 0:
            eval_loss = evaluate_ic_regression(model, cfg, ood_loader, device)
            print(f"[step {step}] OOD eval (M=inf) loss={eval_loss:.4f}")

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
#  Example usage
# =====================

if __name__ == "__main__":
    cfg = ICRegConfig()

    # Example: train at task diversity M = 64
    model_M64 = train_ic_regression(
        cfg,
        M=64,
        num_steps=10_000,   # reduce for a quick test; paper uses 150k
        batch_size=256,     # smaller batch to fit on modest GPUs if needed
        print_every=500,
        eval_every=2_000,
    )

    # Example: train at M = 'inf' (Gaussian task prior)
    # model_Minf = train_ic_regression(cfg, M="inf", num_steps=10_000)
