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
        channels=4,
        hidden_dim=768,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        projection_dim=256,
        gcn_layers=3,
        gcn_hidden=128,
        num_decoder_layers=2, # For Decoder
        vocab_size=None,      # For Decoder
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
# Layers
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Normalized Adjacency
        A = adj + torch.eye(adj.shape[0])
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag(D_inv_sqrt)
        norm_adj = torch.mm(torch.mm(D_mat, A), D_mat)
        self.register_buffer('adj', norm_adj)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x = self.fc(x)
        x_flat = x.view(-1, N, x.shape[-1])
        out = torch.matmul(self.adj, x_flat)
        out = out.view(B, T, N, -1)
        return out + self.bias

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
    """Advanced Decoder Layer with Cross-Attention and RoPE"""
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
# Models
# ==========================================
class PoseEncoder(PreTrainedModel):
    config_class = PoseEncoderConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # GCN Setup
        adj = self.get_custom_adjacency_matrix()
        self.gcn_layers = nn.ModuleList()
        curr_dim = config.channels
        for i in range(config.gcn_layers):
            out_dim = config.gcn_hidden if i < config.gcn_layers - 1 else config.hidden_dim
            self.gcn_layers.append(GraphConvolution(curr_dim, out_dim, adj))
            curr_dim = out_dim
            
        self.gcn_act = nn.GELU()
        self.gcn_dropout = nn.Dropout(config.dropout)
        self.gcn_norm = nn.LayerNorm(config.hidden_dim)

        # Transformer Setup
        self.rotary_emb = RotaryEmbedding(config.hidden_dim // config.num_heads)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)
        
    def get_custom_adjacency_matrix(self):
        adj = torch.zeros((86, 86))
        pairs = []
        # Hands
        rh_bases = [0]*5; rh_fingers = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
        lh_bases = [21]*5; lh_fingers = [[22,23,24,25], [26,27,28,29], [30,31,32,33], [34,35,36,37], [38,39,40,41]]
        
        for b, f in zip(rh_bases + lh_bases, rh_fingers + lh_fingers):
            pairs.append((b, f[0]))
            for k in range(len(f)-1): pairs.append((f[k], f[k+1]))
            
        # Face
        lips = [42, 53, 54, 55, 56, 59, 58, 60, 57, 43, 48, 51, 49, 50, 47, 52, 46, 45, 44]
        for i in range(len(lips)-1): pairs.append((lips[i], lips[i+1]))
        pairs.append((lips[-1], lips[0]))
        pairs.extend([(61,62), (62,63), (63,64), (61,65), (65,66), (66,67), (61,68), (61,69)])
        
        # Body (Custom)
        pairs.extend([(70,72), (71,73), (72,74), (74,76), (73,75), (75,77), (72,73)])
        pairs.extend([(77,79), (79,81), (81,83), (83,77)]) # Right loop
        pairs.extend([(76,78), (78,80), (80,82), (82,76)]) # Left loop
        pairs.extend([(84,85), (72,84), (73,85)])
        
        for i, j in pairs: 
            if i<86 and j<86: adj[i,j] = adj[j,i] = 1
        return adj

    def forward(self, x, attention_mask=None):
        B, T, N, C = x.shape
        # Spatial GCN
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers)-1: x = self.gcn_dropout(self.gcn_act(x))
        
        x = x.mean(dim=2) # Pool nodes -> (B, T, H)
        x = self.gcn_norm(x)
        
        # Temporal Transformer
        cos, sin = self.rotary_emb(x, seq_len=T)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return self.final_norm(x)

class TinyAdvancedDecoder(nn.Module):
    """The Decoder for Stage 2"""
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)