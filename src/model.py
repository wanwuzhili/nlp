import torch
from torch import nn
from torch.nn import functional as F


def attention(Q, K, V):
    """
    Q: [B, T_q, d_q], d_q == d_k
    K: [B, T_k, d_k]
    V: [B, T_k, d_v]
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) # [B, T_q, T_k]
    scores = scores / (K.shape[-1]**0.5) # [B, T_q, T_k]
    weights = F.softmax(scores, dim=-1) # [B, T_q, T_k]
    return torch.matmul(weights, V), weights # [B, T_q, d_v]

class SelfAttention(nn.Module):
    def __init__(self, input_size, d_k, d_v):
        super().__init__()
        self.attention_weights = None
        self.W_qx = nn.Linear(input_size, d_k)
        self.W_kx = nn.Linear(input_size, d_k)
        self.W_vx = nn.Linear(input_size, d_v)

    def forward(self, X):
        """X: [B, T, input_size]"""
        Q = self.W_qx(X) # [B, T, d_k]
        K = self.W_kx(X) # [B, T, d_k]
        V = self.W_vx(X) # [B, T, d_v]

        output, self.attention_weights = attention(Q, K, V) # [B, T, d_v]
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention_weights = None
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X):
        """X: [B, T, D]"""
        B, T, D = X.shape
        d_head = D // self.num_heads

        Q = self.W_Q(X) # [B, T, D]
        K = self.W_K(X) # [B, T, D]
        V = self.W_V(X) # [B, T, D]

        Q = Q.view(B, T, self.num_heads, d_head).transpose(1, 2) #[B, H, T, d_head]
        K = K.view(B, T, self.num_heads, d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, d_head).transpose(1, 2)

        output, self.attention_weights = attention(Q, K, V)
        output = output.transpose(1, 2).contiguous() # [B, T, H, d_head]
        output = output.view(B, T, D)
        return self.W_O(output) # [B, T, D]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, preLN=True):
        super().__init__()
        self.preLN = preLN
        self.MHA = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, X):
        """"X: [B, T, D]"""
        if self.preLN:
            X = X + self.MHA(self.ln1(X))
            X = X + self.ffn(self.ln2(X))
        else:
            X = self.ln1(X + self.MHA(X))
            X = self.ln2(X + self.ffn(X))
        return X # [B, T, D]

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, num_heads):
        super().__init__()
        self.x_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.TBlock = TransformerBlock(d_model, num_heads, 4*d_model, preLN=True)
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        """X: [B, T]"""
        T = X.shape[-1]
        X = self.x_emb(X)
        pos = self.pos_emb(torch.arange(0, T, 1, dtype=torch.long))
        X += pos
        return self.dense(self.TBlock(X)) # [B, T, V] 