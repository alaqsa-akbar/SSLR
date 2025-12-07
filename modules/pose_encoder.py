import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

# ==========================================
# Config & Utilities
# ==========================================
class PoseEncoderConfig(PretrainedConfig):
    model_type = "pose_encoder"
    def __init__(
        self,
        num_keypoints=86,
        channels=2,  # Changed: now 2 for each stream (pose and velocity)
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        projection_dim=256,
        gcn_layers=2,
        gcn_hidden=256,
        num_decoder_layers=2,
        vocab_size=None,
        use_dual_stream=True,  # New: separate pose and velocity streams
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.gcn_layers = gcn_layers
        self.gcn_hidden = gcn_hidden
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.use_dual_stream = use_dual_stream


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len_cached=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_seq_len_cached
        self.update_cache(max_seq_len_cached)

    def update_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.update_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ==========================================
# Graph Convolution Layers
# ==========================================
class GraphConvolution(nn.Module):
    """Graph Convolution with normalized adjacency matrix."""
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Normalized Adjacency: D^(-1/2) * A * D^(-1/2)
        A = adj + torch.eye(adj.shape[0])
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag(D_inv_sqrt)
        norm_adj = torch.mm(torch.mm(D_mat, A), D_mat)
        self.register_buffer('adj', norm_adj)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C) input features
        Returns:
            (B, T, N, out_features)
        """
        B, T, N, C = x.shape
        x = self.fc(x)
        x_flat = x.view(-1, N, x.shape[-1])
        out = torch.matmul(self.adj, x_flat)
        out = out.view(B, T, N, -1)
        return out + self.bias


class GroupGCN(nn.Module):
    """
    Group-specific Graph Convolution Network.
    Processes a specific body part (hand, face, body) with its own adjacency.
    """
    def __init__(self, in_channels, hidden_dim, out_dim, num_layers, keypoint_indices, adjacency_pairs, dropout=0.1):
        super().__init__()
        
        self.keypoint_indices = keypoint_indices
        self.num_keypoints = len(keypoint_indices)
        
        # Build adjacency matrix for this group
        adj = self._build_adjacency(adjacency_pairs)
        
        # GCN layers
        self.layers = nn.ModuleList()
        curr_dim = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                layer_out = out_dim
            else:
                layer_out = hidden_dim
            self.layers.append(GraphConvolution(curr_dim, layer_out, adj))
            curr_dim = layer_out
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
    def _build_adjacency(self, pairs):
        """Build adjacency matrix from pairs of connected keypoints."""
        adj = torch.zeros((self.num_keypoints, self.num_keypoints))
        
        # Create mapping from global index to local index
        global_to_local = {g: l for l, g in enumerate(self.keypoint_indices)}
        
        for i, j in pairs:
            if i in global_to_local and j in global_to_local:
                li, lj = global_to_local[i], global_to_local[j]
                adj[li, lj] = 1
                adj[lj, li] = 1
                
        return adj
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 86, C) full keypoints
        Returns:
            (B, T, out_dim) pooled features for this group
        """
        # Extract keypoints for this group
        group_x = x[:, :, self.keypoint_indices, :]  # (B, T, N_group, C)
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            group_x = layer(group_x)
            if i < len(self.layers) - 1:
                group_x = self.dropout(self.act(group_x))
        
        # Pool over keypoints (spatial mean pooling)
        pooled = group_x.mean(dim=2)  # (B, T, out_dim)
        
        return self.norm(pooled)


class GroupedGCNEncoder(nn.Module):
    """
    Encoder with separate GCNs for each body part group.
    Groups: right_hand, left_hand, face, body
    """
    def __init__(self, in_channels, hidden_dim, gcn_hidden, num_gcn_layers, dropout=0.1):
        super().__init__()
        
        # Per-group output dimension (will be concatenated)
        group_out_dim = hidden_dim // 4
        
        # Define keypoint groups and their adjacencies
        # Right hand: indices 0-20
        right_hand_indices = list(range(0, 21))
        right_hand_pairs = self._get_hand_pairs(0)
        
        # Left hand: indices 21-41  
        left_hand_indices = list(range(21, 42))
        left_hand_pairs = self._get_hand_pairs(21)
        
        # Face: indices 42-69 (lips + eyes)
        face_indices = list(range(42, 70))
        face_pairs = self._get_face_pairs()
        
        # Body: indices 70-85
        body_indices = list(range(70, 86))
        body_pairs = self._get_body_pairs()
        
        # Create group-specific GCNs
        self.right_hand_gcn = GroupGCN(
            in_channels, gcn_hidden, group_out_dim, num_gcn_layers,
            right_hand_indices, right_hand_pairs, dropout
        )
        
        self.left_hand_gcn = GroupGCN(
            in_channels, gcn_hidden, group_out_dim, num_gcn_layers,
            left_hand_indices, left_hand_pairs, dropout
        )
        
        self.face_gcn = GroupGCN(
            in_channels, gcn_hidden, group_out_dim, num_gcn_layers,
            face_indices, face_pairs, dropout
        )
        
        self.body_gcn = GroupGCN(
            in_channels, gcn_hidden, group_out_dim, num_gcn_layers,
            body_indices, body_pairs, dropout
        )
        
        # Fusion layer after concatenation
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def _get_hand_pairs(self, base):
        """Get adjacency pairs for a hand (base=0 for right, base=21 for left)."""
        pairs = []
        # Finger connections from wrist/palm
        fingers = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
        for finger in fingers:
            pairs.append((base, base + finger[0]))  # Wrist to finger base
            for k in range(len(finger)-1):
                pairs.append((base + finger[k], base + finger[k+1]))
        # Palm connections
        pairs.extend([
            (base + 0, base + 5), (base + 0, base + 9), 
            (base + 0, base + 13), (base + 0, base + 17),
            (base + 5, base + 9), (base + 9, base + 13), (base + 13, base + 17)
        ])
        return pairs
    
    def _get_face_pairs(self):
        """Get adjacency pairs for face landmarks."""
        pairs = []
        # Lips (outer): 42-52, (inner): some subset
        lips_outer = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        for i in range(len(lips_outer)-1):
            pairs.append((lips_outer[i], lips_outer[i+1]))
        pairs.append((lips_outer[-1], lips_outer[0]))  # Close loop
        
        # Eyes: 61-69
        pairs.extend([
            (61, 62), (62, 63), (63, 64),  # Right eye
            (61, 65), (65, 66), (66, 67),  # Left eye  
            (61, 68), (61, 69)  # Nose/center
        ])
        return pairs
    
    def _get_body_pairs(self):
        """Get adjacency pairs for body landmarks."""
        pairs = []
        # Upper body skeleton (adjust based on your keypoint layout)
        # Assuming: 70-71 shoulders, 72-73 elbows, 74-75 wrists, etc.
        pairs.extend([
            (70, 72), (71, 73),  # Shoulder to elbow
            (72, 74), (73, 75),  # Elbow to wrist
            (74, 76), (75, 77),  # Wrist to hand
            (70, 71),  # Shoulder to shoulder
            (77, 79), (79, 81), (81, 83), (83, 77),  # Right side
            (76, 78), (78, 80), (80, 82), (82, 76),  # Left side
            (84, 85), (72, 84), (73, 85)  # Additional connections
        ])
        return pairs
        
    def forward(self, x):
        """
        Args:
            x: (B, T, 86, C) input keypoints
        Returns:
            (B, T, hidden_dim) fused features
        """
        # Process each group
        right_hand_feat = self.right_hand_gcn(x)  # (B, T, H/4)
        left_hand_feat = self.left_hand_gcn(x)   # (B, T, H/4)
        face_feat = self.face_gcn(x)             # (B, T, H/4)
        body_feat = self.body_gcn(x)             # (B, T, H/4)
        
        # Concatenate all group features
        fused = torch.cat([right_hand_feat, left_hand_feat, face_feat, body_feat], dim=-1)  # (B, T, H)
        
        # Apply fusion layer
        return self.fusion(fused)


class DualStreamGroupedEncoder(nn.Module):
    """
    Dual-stream encoder with separate pose and velocity streams.
    Each stream uses GroupedGCN, then features are fused.
    """
    def __init__(self, hidden_dim, gcn_hidden, num_gcn_layers, dropout=0.1):
        super().__init__()
        
        # Pose stream (x, y coordinates) - 2 channels
        self.pose_encoder = GroupedGCNEncoder(
            in_channels=2,
            hidden_dim=hidden_dim,
            gcn_hidden=gcn_hidden,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # Velocity stream (vx, vy) - 2 channels  
        self.velocity_encoder = GroupedGCNEncoder(
            in_channels=2,
            hidden_dim=hidden_dim,
            gcn_hidden=gcn_hidden,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # Fusion: concatenate pose and velocity features, then project
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, 86, 4) input with [x, y, vx, vy]
        Returns:
            (B, T, hidden_dim) fused features
        """
        # Split into pose and velocity
        pose = x[:, :, :, :2]      # (B, T, 86, 2)
        velocity = x[:, :, :, 2:]  # (B, T, 86, 2)
        
        # Process each stream
        pose_feat = self.pose_encoder(pose)        # (B, T, H)
        velocity_feat = self.velocity_encoder(velocity)  # (B, T, H)
        
        # Fuse streams
        fused = torch.cat([pose_feat, velocity_feat], dim=-1)  # (B, T, H*2)
        return self.fusion(fused)  # (B, T, H)


# ==========================================
# Transformer Layers
# ==========================================
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.dropout)
        )
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        x_norm = self.norm1(x)
        
        qkv = self.qkv(x_norm).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min
            
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        x_attn = x_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = residual + self.dropout(self.o_proj(x_attn))
        
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Decoder Layer with Cross-Attention and RoPE."""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Self-Attention
        self.norm1 = RMSNorm(config.hidden_dim)
        self.self_attn_qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.self_attn_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        # Cross-Attention
        self.norm2 = RMSNorm(config.hidden_dim)
        self.cross_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.cross_kv = nn.Linear(config.hidden_dim, 2 * config.hidden_dim, bias=False)
        self.cross_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        # MLP
        self.norm3 = RMSNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_out, cos, sin, tgt_mask=None, enc_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Self Attention
        residual = x
        x_norm = self.norm1(x)
        qkv = self.self_attn_qkv(x_norm).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=tgt_mask, dropout_p=self.dropout.p if self.training else 0.0)
        x = residual + self.dropout(self.self_attn_o(x_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)))
        
        # 2. Cross Attention
        residual = x
        x_norm = self.norm2(x)
        q = self.cross_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv = self.cross_kv(enc_out).view(batch_size, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_enc, v_enc = kv[0], kv[1]
        
        attn_mask = None
        if enc_mask is not None:
            attn_mask = enc_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min
             
        x_cross = F.scaled_dot_product_attention(q, k_enc, v_enc, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        x = residual + self.dropout(self.cross_o(x_cross.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)))
        
        # 3. MLP
        residual = x
        x = residual + self.mlp(self.norm3(x))
        return x


# ==========================================
# Main Models
# ==========================================
class PoseEncoder(PreTrainedModel):
    """
    Improved Pose Encoder with:
    - Grouped GCN (separate processing for hands, face, body)
    - Dual-stream (pose + velocity)
    - Transformer for temporal modeling
    """
    config_class = PoseEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Spatial encoder: Dual-stream grouped GCN
        if config.use_dual_stream:
            self.spatial_encoder = DualStreamGroupedEncoder(
                hidden_dim=config.hidden_dim,
                gcn_hidden=config.gcn_hidden,
                num_gcn_layers=config.gcn_layers,
                dropout=config.dropout
            )
        else:
            # Fallback: single stream with 4 channels
            self.spatial_encoder = GroupedGCNEncoder(
                in_channels=4,
                hidden_dim=config.hidden_dim,
                gcn_hidden=config.gcn_hidden,
                num_gcn_layers=config.gcn_layers,
                dropout=config.dropout
            )

        # Temporal encoder: Transformer
        self.rotary_emb = RotaryEmbedding(config.hidden_dim // config.num_heads)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, T, 86, 4) input with [x, y, vx, vy]
            attention_mask: (B, T) mask for valid frames
        Returns:
            (B, T, hidden_dim) encoded features
        """
        B, T, N, C = x.shape
        
        # Spatial encoding (grouped GCN with dual-stream)
        x = self.spatial_encoder(x)  # (B, T, H)
        
        # Temporal encoding (Transformer)
        cos, sin = self.rotary_emb(x, seq_len=T)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
            
        return self.final_norm(x)


class TinyAdvancedDecoder(nn.Module):
    """Decoder for sequence generation."""
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.rotary_emb = RotaryEmbedding(config.hidden_dim // config.num_heads)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)
        self.fc_out = nn.Linear(config.hidden_dim, vocab_size, bias=False)
        
    def get_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, input_ids, enc_out, enc_mask=None):
        x = self.embedding(input_ids)
        B, T, _ = x.shape
        cos, sin = self.rotary_emb(x, seq_len=T)
        tgt_mask = self.get_causal_mask(T).to(x.device)
        
        for layer in self.layers:
            x = layer(x, enc_out, cos, sin, tgt_mask=tgt_mask, enc_mask=enc_mask)
            
        x = self.final_norm(x)
        return self.fc_out(x)


class PoseEncoderForContrastive(PreTrainedModel):
    """Pose encoder with projection head for contrastive learning."""
    config_class = PoseEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoseEncoder(config)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.projection_dim)
        )
        
    def forward(self, x, attention_mask=None, return_projection=False):
        emb = self.encoder(x, attention_mask)
        if return_projection:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = emb.mean(dim=1)
            return self.projection(pooled)
        return emb


class TextProjectionHead(nn.Module):
    """Text projection head for contrastive learning."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): 
        return self.net(x)


# ==========================================
# Testing
# ==========================================
def test_model():
    """Test the model with random input."""
    print("Testing Improved Pose Encoder...")
    
    config = PoseEncoderConfig(
        num_keypoints=86,
        channels=2,
        hidden_dim=256,  # Smaller for testing
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        gcn_layers=2,
        gcn_hidden=128,
        use_dual_stream=True
    )
    
    model = PoseEncoder(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 86, 4)  # [x, y, vx, vy]
    mask = torch.ones(batch_size, seq_len)
    mask[1, 30:] = 0  # Second sample has 30 valid frames
    
    output = model(x, attention_mask=mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 50, {config.hidden_dim})")
    
    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    print("✓ Test passed!")


if __name__ == "__main__":
    test_model()