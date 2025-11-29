import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import math

class PoseEncoderConfig(PretrainedConfig):
    model_type = "pose_encoder"
    def __init__(
        self,
        input_dim=172,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len_cached=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_seq_len_cached
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bs, heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, attention_mask=None):
        # x: [batch, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape
        
        # Pre-Norm
        residual = x
        x_norm = self.norm1(x)
        
        # Self-Attention
        qkv = self.qkv(x_norm)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled Dot Product Attention (SDPA)
        attn_mask = None
        if attention_mask is not None:
             # attention_mask is 1 for valid, 0 for pad. 
             # scaled_dot_product_attention expects mask where True is kept, False is masked (if boolean)
             # OR 0 for keep, -inf for mask (if float)
             attn_mask = attention_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
             # Convert 1/0 to 0/-inf
             attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else 0.1)
        
        x_attn = x_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        x_attn = self.o_proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        x = residual + x_attn
        
        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = residual + x_mlp
        
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register adjacency matrix as buffer (constant)
        # Normalize A: D^-0.5 * (A + I) * D^-0.5
        A = adj + torch.eye(adj.shape[0])
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag(D_inv_sqrt)
        norm_adj = torch.mm(torch.mm(D_mat, A), D_mat)
        
        self.register_buffer('adj', norm_adj)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: (Batch, Frames, Nodes, In_Dim)
        # Support (Batch, Nodes, In_Dim) or (Batch, Frames, Nodes, In_Dim)
        
        if x.dim() == 4:
            B, T, N, C = x.shape
            # Merge B and T: (B*T, N, C)
            x = x.view(B*T, N, C)
            support = torch.matmul(x, self.weight) # (B*T, N, Out)
            output = torch.matmul(self.adj, support) # (B*T, N, Out)
            output = output.view(B, T, N, self.out_features)
        else:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adj, support)
            
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def get_adjacency_matrix(num_nodes=86):
    adj = torch.zeros((num_nodes, num_nodes))
    
    pairs = []
    # --- Right Hand (0-20) ---
    pairs.extend([(0,1), (1,2), (2,3), (3,4)])
    pairs.extend([(0,5), (5,6), (6,7), (7,8)])
    pairs.extend([(0,9), (9,10), (10,11), (11,12)])
    pairs.extend([(0,13), (13,14), (14,15), (15,16)])
    pairs.extend([(0,17), (17,18), (18,19), (19,20)])
    
    # --- Left Hand (21-41) ---
    pairs.extend([(21,22), (22,23), (23,24), (24,25)])
    pairs.extend([(21,26), (26,27), (27,28), (28,29)])
    pairs.extend([(21,30), (30,31), (31,32), (32,33)])
    pairs.extend([(21,34), (34,35), (35,36), (36,37)])
    pairs.extend([(21,38), (38,39), (39,40), (40,41)])
    
    # --- Body (72-85) ---
    pairs.append((72, 73))
    pairs.extend([(72, 74), (74, 76)])
    pairs.extend([(73, 75), (75, 77)])
    pairs.extend([(72, 84), (73, 85)])
    pairs.append((84, 85))
    pairs.append((76, 21))
    pairs.append((77, 0))
    
    # --- Face (42-71) ---
    pairs.extend([(61, 62), (61, 65)])
    pairs.extend([(62, 63), (63, 64)])
    pairs.extend([(65, 66), (66, 67)])
    
    # Lips Loop
    lips_indices = [42, 53, 54, 55, 56, 59, 58, 60, 57, 43, 48, 51, 49, 50, 47, 52, 46, 45, 44]
    for i in range(len(lips_indices) - 1):
        pairs.append((lips_indices[i], lips_indices[i+1]))
    pairs.append((lips_indices[-1], lips_indices[0]))
    
    # Cheekbones
    pairs.extend([(61, 68), (61, 69)])
    
    # 70/71 to Corners (56, 47)
    pairs.append((70, 56)) # 70 to Left Corner
    pairs.append((71, 47)) # 71 to Right Corner
    
    # Symmetric
    for i, j in pairs:
        adj[i, j] = 1
        adj[j, i] = 1
        
    return adj

class PoseEncoder(PreTrainedModel):
    config_class = PoseEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # GCN Projection
        # Input: (B, T, 86, 2) -> GCN -> (B, T, 86, Hidden) -> Pool -> (B, T, Hidden)
        # We project 2 coords to hidden_dim using GCN
        adj = get_adjacency_matrix(86)
        self.gcn = GraphConvolution(2, config.hidden_dim, adj)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.rotary_emb = RotaryEmbedding(config.hidden_dim // config.num_heads)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.norm_final = RMSNorm(config.hidden_dim)
        
        self.post_init()

    def forward(self, x, attention_mask=None):
        # x: (Batch, Frames, 86, 2)
        batch_size, frames, num_points, coords = x.shape
        
        # Apply GCN
        # (B, T, 86, 2) -> (B, T, 86, H)
        x = self.gcn(x)
        x = self.dropout(x)
        
        # Flatten/Pool nodes to get sequence of frames
        # Strategy: Mean pool over nodes (86) to get (B, T, H)
        x = x.mean(dim=2) # (B, T, H)
        
        # Prepare RoPE
        cos, sin = self.rotary_emb(x, seq_len=frames)
        
        # Transformer Layers
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
            
        x = self.norm_final(x)
        
        return x
