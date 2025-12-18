"""
Decode-only transformer module.

Resources:

* Code roughly following Karpathy's tutorial and nanoGPT implementation, but
  with some features removed (such as dropout) and with some micro-
  optimisations here and there.
* See also Phuong and Hutter's 'Formal algorithms for transformers', though
  this implementation differs in a few places e.g. using Pre-layer-norm
  rather than Post-layer-norm.

Notes:

* Takes vector tokens (rather than token indices) as input and output.
  So for language this would need to be one-hot encoded.
  * TODO: Embedding had different initalisation compared to Linear, namely
    N(0,1) rather than Uniform---should I care?
* Karpathy does a final transformation after the attention block, with
  dropout. Is this just for dropout? Because this goes straight into the
  MLP. So I got rid of that.
"""


import torch
import torch.nn as nn
import torch.nn.functional as fn


class DTransformer(nn.Module):
    def __init__(
        self,
        config,
    ):
        
        super().__init__()
        self.token_size = 1 + config['task_size'] # task_size for x + 1 for y
        self.max_tokens = 2 * config['num_examples'] # one x + one y per example
        self.mlp_size = config['mlp_size']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.use_embedding = config['use_embedding']
        self.use_layernorm = config['use_layernorm']
        self.use_softmax = config['use_softmax']

        if self.use_embedding:
            self.embed_size = config['embed_size']
            self.token_embedding = nn.Linear(
                in_features=self.token_size,
                out_features=self.embed_size,
                bias=False,
                device=self.device,
            )
            self.postn_embedding = nn.Linear(
                in_features=self.max_tokens,
                out_features=self.embed_size,
                bias=False,
                device=self.device,
            )
        else:
            # The forward pass below is modified to not expect embeddings
            self.embed_size = self.token_size # The size of the residual stream vector needs to match the input size
            self.token_embedding = None
            self.postn_embedding = None

        self.blocks = nn.ModuleList([
            MultiHeadedCausalSelfAttentionTransformerBlock(
                embed_size=self.embed_size,
                mlp_size=self.mlp_size,
                max_tokens=self.max_tokens,
                num_heads=self.num_heads,
                use_layernorm=self.use_layernorm,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ])

        if self.use_embedding:
            # unembedding
            self.unembedding = nn.Sequential(
                nn.LayerNorm(
                    normalized_shape=self.embed_size,
                    device=self.device,
                ),
                nn.Linear(
                    in_features=self.embed_size,
                    out_features=self.token_size,
                    device=self.device,
                ),
            )
        

    def forward(self, toks):
        _B, T, _V = toks.shape
        assert T<=self.max_tokens, f"too many tokens! {T} > {self.max_tokens}"

        if self.use_embedding:
            # semantic and positional token embeddings
            x_positions = self.postn_embedding.weight.T[:T, :] # Tmax C ->   T C
            x_semantics = self.token_embedding(toks)    # B T V @ . V C -> B T C
            x = x_semantics + x_positions               # B T C + . T C -> B T C
        else: 
            # Here V=C and the token embedding is the identity
            x = toks

        # apply the num_layers layers / attention blocks in sequence
        for block in self.blocks:
            x = x + block(x)                        # B T C + B T C -> B T C

        if self.use_embedding:
            # unembedding: transform back to predicted next tokens
            y = self.unembedding(x)                     # B T C @ . C V -> B T V
        else:
            y = x

        return y
        # NOTE:
        # during training,  we only care about y[:, :-1, :]...
        # during inference, we only care about y[:, -1:, :]...
        # TODO: optimise!
        # (moreover in the in-context regression setting, we really only care
        # about every second token prediction to begin with...)


class MultiHeadedCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        mlp_size,
        max_tokens,
        num_heads,
        use_layernorm=True, 
        device='cpu',
    ):
        super().__init__()
        self.attention = MultiHeadedCausalSelfAttention(
            embed_size=embed_size,
            max_tokens=max_tokens,
            num_heads=num_heads,
            device=device,
        )
        # TODO: Implement this as a flag (new parameter) instead
        if mlp_size==0:
            # The MLP is (we think) implicitly acting as the output matrix W_O which is otherwise missing. 
            # Therefore instead of the identity, this should be a linear layer with no bias.
            self.compute = nn.Linear(embed_size, embed_size, device=device)
        else: 
            self.compute = nn.Sequential(
                nn.Linear(embed_size, mlp_size, device=device),
                nn.ReLU(),
                nn.Linear(mlp_size, embed_size, device=device),
            )

        if use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(normalized_shape=embed_size, device=device)
                for _ in ('before-attention', 'before-compute')
            ])
        else:
            self.layer_norms = nn.ModuleList([
                nn.Identity()
                for _ in ('before-attention', 'before-compute')
            ])


    def forward(self, x):
        # B, T, C = x.shape
        x = x + self.attention(self.layer_norms[0](x))
        x = x + self.compute(self.layer_norms[1](x))
        return x


class MultiHeadedCausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        max_tokens,
        num_heads,
        use_softmax=True,
        device='cpu',
    ):
        super().__init__()
        self.use_softmax = use_softmax

        # validate dimensions
        if embed_size % num_heads:
            raise ValueError("num_heads must divide embed_size")
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        # batched key/query/value projections
        self.attention = nn.Linear(
            in_features=embed_size,
            out_features=3*embed_size,
            bias=False,
            device=device,
        )
        # precompute causal mask
        mask_shape = (max_tokens, max_tokens)
        causal_mask = torch.log(torch.tril(torch.ones(mask_shape, device=device)))
        self.register_buffer('causal_mask', causal_mask)
        # precompute attention normalisation factor
        self.attention_scale = self.head_size ** 0.5


    def forward(self, x):
        # unpack dimensions
        B, T, C = x.size()  # batch size, num_tokens, embed_size
        H = self.num_heads  # num_heads
        c = self.head_size  # head size

        # perform Q, K, V transforms, all at once
        Q, K, V = (self.attention(x)    # B T C @ C 3C  -> B T 3C
                .view(B, T, H, 3*c)     #               -> B T H 3c
                .transpose(-2, -3)      #               -> B H T 3c
                .split(c, dim=-1)       #               -> (B H T c) * 3
            )
        # now Q, K, V are each of shape (B, H, T, c)

        # compute affinities, scaled and with causal mask
        A = Q @ K.transpose(-2, -1)     # B H T c @ B H c T -> B H T T
        A = A / self.attention_scale    # B H T T / . . . T -> B H T T
        A = A + self.causal_mask[:T,:T] # B H T T + . . T T -> B H T T

        # convert affinities to mixing weights and mix value vectors
        if self.use_softmax:
            p = fn.softmax(A, dim=-1)   # B H T T -> B H T T(sum to 1)
        else: 
            p = A

        y = p @ V                   # B H T T @ B H T c -> B H T c

        # recombine / concatenate heads into new embedding
        y = (y                      #    B H T c
                .transpose(-3, -2)  # -> B T H c
                .contiguous()       # -> (make underlying memory match view)
                .view(B, T, C)      # -> B T C
             )

        return y

