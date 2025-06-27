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
        Q_norm = Q_norm.transpose(0, 1)  # Convert to [N, batch_size, dim_V]
        K_norm = K_norm.transpose(0, 1)
        attn_output, _ = self.multihead_attn(Q_norm, K_norm, K_norm)
        attn_output = attn_output.transpose(0, 1)  # Convert back to [batch_size, N, dim_V]
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
        S = self.S.repeat(batch_size, 1, 1)  # [batch_size, num_seeds, dim]
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
        attn_output, _ = self.self_attention(u_proj, u_proj, u_proj)
        u_1 = self.layer_norm_1(u_proj + attn_output)
        
        # Step 2: u_2 = LayerNorm_2(u_1 + FeedForward(u_1))
        ff_output_1 = self.feed_forward_1(u_1)
        u_2 = self.layer_norm_2(u_1 + ff_output_1)
        
        # Step 3: u_3 = LayerNorm_3(u_2 + CrossAttention(u_2, [y]))
        # Cross-attention: u_2 as query, y_proj as key and value
        cross_attn_output, _ = self.cross_attention(u_2, y_proj, y_proj)
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


# Usage example and test code
if __name__ == "__main__":
    # Set parameters
    batch_size = 4
    seq_len = 10
    u_dim = 32
    y_dim = 16
    output_dim = 16
    hidden_dim = 64
    
    # Create network
    model = ConditionTransformerNetwork(
        u_dim=u_dim,
        y_dim=y_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_blocks=3,
        num_heads=8,
        ff_expansion=2
    )
    
    # Create test data
    u = torch.randn(batch_size, seq_len, u_dim)
    y = torch.randn(batch_size, y_dim)
    
    # Forward pass
    output = model(u, y)
    print(f"Input u shape: {u.shape}")
    print(f"Input y shape: {y.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
