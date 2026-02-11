import argparse
import time

import torch
import torch.nn as nn

class Simple_MLP(nn.Module):
    def __init__(self, d_input, d_output, latent_dim=64, num_hidden_layers=2):
        super(Simple_MLP, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(d_input, latent_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(latent_dim, d_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class NaiveNetwork(nn.Module):
    def __init__(self, d):
        super(NaiveNetwork, self).__init__()
        self.d = d

    def forward(self, v):
        return torch.zeros(v.shape[0], self.d).to(v.device)


class MAB(nn.Module):
    """
    Multihead Attention Block (MAB)
    """
    def __init__(self, dim_Q, dim_KV, num_heads, freeze_WQ=False):
        super(MAB, self).__init__()
        self.dim_V = dim_Q  # Output dimension matches the query dimension
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.dim_V,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(self.dim_V)
        self.ln2 = nn.LayerNorm(self.dim_V)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_V, self.dim_V),
            nn.ReLU(),
            nn.Linear(self.dim_V, self.dim_V)
        )
        self.freeze_WQ = freeze_WQ
        if self.freeze_WQ:
            self._freeze_WQ()

    def _freeze_WQ(self):
        # Get the in_proj_weight parameter
        in_proj_weight = self.multihead_attn.in_proj_weight
        embed_dim = self.multihead_attn.embed_dim
        # Set W_Q to the identity matrix
        with torch.no_grad():
            in_proj_weight[:embed_dim, :] = torch.eye(embed_dim)
        # Register a backward hook to zero out gradients for W_Q
        def hook(grad):
            grad[:embed_dim, :] = 0
            return grad
        in_proj_weight.register_hook(hook)
        # Handle the bias term if it exists
        if self.multihead_attn.in_proj_bias is not None:
            in_proj_bias = self.multihead_attn.in_proj_bias
            with torch.no_grad():
                in_proj_bias[:embed_dim] = 0
            # Register a backward hook for the bias
            def bias_hook(grad):
                grad[:embed_dim] = 0
                return grad
            in_proj_bias.register_hook(bias_hook)

    def forward(self, Q, K):
        # Q, K: [batch_size, N, dim_V]
        Q_norm = self.ln1(Q)
        K_norm = self.ln1(K)
        # Skip returning attention weights to reduce overhead and unlock
        # optimized SDPA paths in recent PyTorch versions.
        attn_output = self.multihead_attn(
            Q_norm,
            K_norm,
            K_norm,
            need_weights=False
        )[0]
        H = Q + attn_output  # Residual connection
        H_norm = self.ln2(H)
        H = H + self.ffn(H_norm)  # Residual connection
        return H

class SAB(nn.Module):
    """
    Self-Attention Block (SAB)
    """
    def __init__(self, dim_in, dim_out, num_heads):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, num_heads)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        H = self.mab(X, X)
        return self.fc(H)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA)
    """
    def __init__(self, dim, num_heads, num_seeds, freeze_WQ=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))  # Seed vectors
        self.mab = MAB(dim, dim, num_heads, freeze_WQ)

    def forward(self, X):
        batch_size = X.size(0)
        S = self.S.expand(batch_size, -1, -1)  # [batch_size, num_seeds, dim]
        return self.mab(S, X)

class SetTransformer(nn.Module):
    """
    Set Transformer main model with customizable number of SAB layers.
    """
    def __init__(self, input_dim, num_heads, num_inds, output_dim, hidden_dim, num_layers=2, freeze_WQ=False):
        super(SetTransformer, self).__init__()
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Self-attention encoder blocks
        self.enc = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Pooling layer
        self.pma = PMA(hidden_dim, num_heads, num_inds, freeze_WQ)
        # Self-attention decoder blocks
        self.dec = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Output layer
        self.fc_out = nn.Linear(hidden_dim * num_inds, output_dim)
    
    def forward(self, X):
        # X: [batch_size, N, input_dim]
        H = self.embedding(X)
        H = self.enc(H)
        H = self.pma(H)
        H = self.dec(H)
        H = H.view(H.size(0), -1)  # Flatten
        output = self.fc_out(H)
        return output

class CustomTransformerBlock(nn.Module):
    """
    Custom Transformer Block with self-attention and cross-attention
    
    Architecture:
    u_1 = LayerNorm_1(u + SelfAttention(u))
    u_2 = LayerNorm_2(u_1 + FeedForward(u_1))
    u_3 = LayerNorm_3(u_2 + CrossAttention(u_2, [y]))
    output = LayerNorm_4(u_3 + FeedForward(u_3))
    """
    def __init__(self, u_dim, y_dim, hidden_dim=None, num_heads=8, ff_expansion=4):
        super(CustomTransformerBlock, self).__init__()
        
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else u_dim
        self.num_heads = num_heads
        
        # Ensure dimension is divisible by number of heads
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Input projection layers (if u_dim != hidden_dim)
        self.u_projection = nn.Linear(u_dim, self.hidden_dim) if u_dim != self.hidden_dim else nn.Identity()
        self.y_projection = nn.Linear(y_dim, self.hidden_dim) if y_dim != self.hidden_dim else nn.Identity()
        
        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward networks
        ff_hidden_dim = self.hidden_dim * ff_expansion
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, self.hidden_dim)
        )
        
        self.feed_forward_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, self.hidden_dim)
        )
        
        # Layer Normalization layers
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_3 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_4 = nn.LayerNorm(self.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, u, y):
        """
        Forward pass
        
        Args:
            u: Input sequence [batch_size, seq_len, u_dim]
            y: Single vector [batch_size, y_dim] or [batch_size, 1, y_dim]
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size = u.size(0)
        seq_len = u.size(1)
        
        # Project to unified dimension
        u_proj = self.u_projection(u)  # [batch_size, seq_len, hidden_dim]
        
        # Handle y dimension: ensure y is [batch_size, 1, hidden_dim]
        if y.dim() == 2:
            y = y.unsqueeze(1)  # [batch_size, 1, y_dim]
        y_proj = self.y_projection(y)  # [batch_size, 1, hidden_dim]
        
        # Step 1: u_1 = LayerNorm_1(u + SelfAttention(u))
        attn_output = self.self_attention(
            u_proj,
            u_proj,
            u_proj,
            need_weights=False
        )[0]
        u_1 = self.layer_norm_1(u_proj + attn_output)
        
        # Step 2: u_2 = LayerNorm_2(u_1 + FeedForward(u_1))
        ff_output_1 = self.feed_forward_1(u_1)
        u_2 = self.layer_norm_2(u_1 + ff_output_1)
        
        # Step 3: u_3 = LayerNorm_3(u_2 + CrossAttention(u_2, [y]))
        # Cross-attention: u_2 as query, y_proj as key and value
        cross_attn_output = self.cross_attention(
            u_2,
            y_proj,
            y_proj,
            need_weights=False
        )[0]
        u_3 = self.layer_norm_3(u_2 + cross_attn_output)
        
        # Step 4: output = LayerNorm_4(u_3 + FeedForward(u_3))
        ff_output_2 = self.feed_forward_2(u_3)
        output = self.layer_norm_4(u_3 + ff_output_2)
        
        return output


class ConditionTransformerNetwork(nn.Module):
    """
    Complete network structure based on CustomTransformerBlock
    """
    def __init__(self, 
                 u_dim,           # Input sequence u feature dimension
                 y_dim,           # Input vector y feature dimension
                 output_dim,      # Output sequence feature dimension
                 hidden_dim=512,  # Hidden layer dimension
                 num_blocks=4,    # Number of Transformer blocks
                 num_heads=8,     # Number of attention heads
                 ff_expansion=4): # Feed-forward network expansion factor
        super(ConditionTransformerNetwork, self).__init__()
        
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Input embedding layers
        self.input_embedding = nn.Linear(u_dim, hidden_dim)
        self.y_embedding = nn.Linear(y_dim, hidden_dim)
        
        # Multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(
                u_dim=hidden_dim,  # After first block, all are hidden_dim
                y_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_expansion=ff_expansion
            ) for _ in range(num_blocks)
        ])
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, u, y):
        """
        Forward pass
        
        Args:
            u: Input sequence [batch_size, seq_len, u_dim]
            y: Input vector [batch_size, y_dim]
            
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # Input embedding
        u_embedded = self.input_embedding(u)  # [batch_size, seq_len, hidden_dim]
        y_embedded = self.y_embedding(y)      # [batch_size, hidden_dim]
        
        # Pass through multiple Transformer blocks
        current_u = u_embedded
        for block in self.transformer_blocks:
            current_u = block(current_u, y_embedded)
        
        # Output projection
        final_output = self.output_projection(current_u)  # [batch_size, seq_len, output_dim]
        
        return final_output


class MABReference(nn.Module):
    """
    Reference (pre-optimization) MAB implementation for correctness/speed comparison.
    """
    def __init__(self, dim_Q, dim_KV, num_heads, freeze_WQ=False):
        super(MABReference, self).__init__()
        self.dim_V = dim_Q
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.dim_V, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(self.dim_V)
        self.ln2 = nn.LayerNorm(self.dim_V)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_V, self.dim_V),
            nn.ReLU(),
            nn.Linear(self.dim_V, self.dim_V)
        )
        self.freeze_WQ = freeze_WQ
        if self.freeze_WQ:
            self._freeze_WQ()

    def _freeze_WQ(self):
        in_proj_weight = self.multihead_attn.in_proj_weight
        embed_dim = self.multihead_attn.embed_dim
        with torch.no_grad():
            in_proj_weight[:embed_dim, :] = torch.eye(embed_dim)

        def hook(grad):
            grad[:embed_dim, :] = 0
            return grad

        in_proj_weight.register_hook(hook)
        if self.multihead_attn.in_proj_bias is not None:
            in_proj_bias = self.multihead_attn.in_proj_bias
            with torch.no_grad():
                in_proj_bias[:embed_dim] = 0

            def bias_hook(grad):
                grad[:embed_dim] = 0
                return grad

            in_proj_bias.register_hook(bias_hook)

    def forward(self, Q, K):
        Q_norm = self.ln1(Q)
        K_norm = self.ln1(K)
        Q_norm = Q_norm.transpose(0, 1)
        K_norm = K_norm.transpose(0, 1)
        attn_output, _ = self.multihead_attn(Q_norm, K_norm, K_norm)
        attn_output = attn_output.transpose(0, 1)
        H = Q + attn_output
        H_norm = self.ln2(H)
        H = H + self.ffn(H_norm)
        return H


class CustomTransformerBlockReference(CustomTransformerBlock):
    """
    Reference (pre-optimization) block using attention calls that return weights.
    """
    def forward(self, u, y):
        u_proj = self.u_projection(u)
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y_proj = self.y_projection(y)

        attn_output, _ = self.self_attention(u_proj, u_proj, u_proj)
        u_1 = self.layer_norm_1(u_proj + attn_output)

        ff_output_1 = self.feed_forward_1(u_1)
        u_2 = self.layer_norm_2(u_1 + ff_output_1)

        cross_attn_output, _ = self.cross_attention(u_2, y_proj, y_proj)
        u_3 = self.layer_norm_3(u_2 + cross_attn_output)

        ff_output_2 = self.feed_forward_2(u_3)
        output = self.layer_norm_4(u_3 + ff_output_2)
        return output


def _sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _compare_outputs(ref_model, fast_model, inputs):
    with torch.no_grad():
        ref_out = ref_model(*inputs)
        fast_out = fast_model(*inputs)
    diff = (ref_out - fast_out).abs()
    return diff.max().item(), diff.mean().item()


def _benchmark_forward_ms(model, inputs, warmup_steps, iters):
    model.eval()
    device = inputs[0].device
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(*inputs)
        _sync_cuda(device)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(*inputs)
        _sync_cuda(device)
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def _resolve_dtype(dtype_name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def run_attention_benchmark(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = _resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("float16/bfloat16 benchmark on CPU is not stable for attention. Fallback to float32.")
        dtype = torch.float32

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")

    print("=== Attention Benchmark Config ===")
    print(f"device={device}, dtype={dtype}, batch_size={args.batch_size}")
    print(f"num_heads={args.num_heads}, hidden_dim={args.hidden_dim}")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print("==================================")

    # MAB benchmark
    mab_fast = MAB(
        dim_Q=args.hidden_dim,
        dim_KV=args.hidden_dim,
        num_heads=args.num_heads,
        freeze_WQ=False,
    ).to(device=device, dtype=dtype)
    mab_ref = MABReference(
        dim_Q=args.hidden_dim,
        dim_KV=args.hidden_dim,
        num_heads=args.num_heads,
        freeze_WQ=False,
    ).to(device=device, dtype=dtype)
    mab_ref.load_state_dict(mab_fast.state_dict(), strict=True)
    mab_fast.eval()
    mab_ref.eval()

    Q = torch.randn(args.batch_size, args.mab_q_len, args.hidden_dim, device=device, dtype=dtype)
    K = torch.randn(args.batch_size, args.mab_k_len, args.hidden_dim, device=device, dtype=dtype)

    mab_max_diff, mab_mean_diff = _compare_outputs(mab_ref, mab_fast, (Q, K))
    mab_ref_ms = _benchmark_forward_ms(mab_ref, (Q, K), args.warmup, args.iters)
    mab_fast_ms = _benchmark_forward_ms(mab_fast, (Q, K), args.warmup, args.iters)
    mab_speedup = mab_ref_ms / mab_fast_ms if mab_fast_ms > 0 else float("inf")

    # CustomTransformerBlock benchmark
    block_fast = CustomTransformerBlock(
        u_dim=args.hidden_dim,
        y_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_expansion=args.ff_expansion,
    ).to(device=device, dtype=dtype)
    block_ref = CustomTransformerBlockReference(
        u_dim=args.hidden_dim,
        y_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_expansion=args.ff_expansion,
    ).to(device=device, dtype=dtype)
    block_ref.load_state_dict(block_fast.state_dict(), strict=True)
    block_fast.eval()
    block_ref.eval()

    u = torch.randn(args.batch_size, args.seq_len, args.hidden_dim, device=device, dtype=dtype)
    y = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=dtype)

    block_max_diff, block_mean_diff = _compare_outputs(block_ref, block_fast, (u, y))
    block_ref_ms = _benchmark_forward_ms(block_ref, (u, y), args.warmup, args.iters)
    block_fast_ms = _benchmark_forward_ms(block_fast, (u, y), args.warmup, args.iters)
    block_speedup = block_ref_ms / block_fast_ms if block_fast_ms > 0 else float("inf")

    print("\n=== Results ===")
    print("[MAB]")
    print(f"max_abs_diff={mab_max_diff:.6e}, mean_abs_diff={mab_mean_diff:.6e}")
    print(f"ref_ms={mab_ref_ms:.4f}, fast_ms={mab_fast_ms:.4f}, speedup={mab_speedup:.3f}x")
    print("[CustomTransformerBlock]")
    print(f"max_abs_diff={block_max_diff:.6e}, mean_abs_diff={block_mean_diff:.6e}")
    print(f"ref_ms={block_ref_ms:.4f}, fast_ms={block_fast_ms:.4f}, speedup={block_speedup:.3f}x")


def run_network_demo(args):
    model = ConditionTransformerNetwork(
        u_dim=args.demo_u_dim,
        y_dim=args.demo_y_dim,
        output_dim=args.demo_output_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.demo_num_blocks,
        num_heads=args.num_heads,
        ff_expansion=args.ff_expansion,
    )
    u = torch.randn(args.batch_size, args.seq_len, args.demo_u_dim)
    y = torch.randn(args.batch_size, args.demo_y_dim)
    output = model(u, y)
    print("\n=== Network Demo ===")
    print(f"Input u shape: {u.shape}")
    print(f"Input y shape: {y.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


def _build_cli():
    parser = argparse.ArgumentParser(description="Attention correctness and speed benchmark for networks.py")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "demo", "all"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--mab_q_len", type=int, default=16)
    parser.add_argument("--mab_k_len", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_expansion", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--demo_u_dim", type=int, default=32)
    parser.add_argument("--demo_y_dim", type=int, default=16)
    parser.add_argument("--demo_output_dim", type=int, default=16)
    parser.add_argument("--demo_num_blocks", type=int, default=3)
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    if args.mode in ["benchmark", "all"]:
        run_attention_benchmark(args)
    if args.mode in ["demo", "all"]:
        run_network_demo(args)
