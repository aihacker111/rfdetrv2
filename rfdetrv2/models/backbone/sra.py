# rfdetrv2/models/backbone/sra.py
"""Semantic Routing Attention (SRA).

Giảm số tham số / chi phí (gợi ý):
- Giảm ``G`` (số centroid): ``W_r`` có ``dim × G`` tham số.
- Giảm ``n_heads``: nhẹ hơn ở ``MultiheadAttention`` và các ``*_proj``.
- Dùng **một** module SRA chung cho mọi mức feature (cùng ``dim``), thay vì nhân bản theo số mức
  (xem ``sra_shared`` trong ``Backbone``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticRoutingAttention(nn.Module):
    """
    Thay thế window attention: O(N*G + G²) thay vì O(N²/K).
    
    Tính chất:
    - Soft semantic routing (differentiable, không hard partition)
    - Cross-region context thông qua G centroids
    - Fully compatible với full-resolution DINOv3 (không cần windowing)
    
    Args:
        dim:      feature dimension d từ DINOv3
        G:        số semantic groups (centroids). Default 64.
                  G << N (N thường là 1600 cho 640×640 với patch=16)
        n_heads:  attention heads cho cross-attn và centroid self-attn
    """
    def __init__(self, dim: int, G: int = 64, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0

        # Routing: project tokens → G soft group scores
        self.W_r = nn.Linear(dim, G, bias=False)

        # Cross-attention: tokens attend to centroids
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Centroid self-attention (O(G²), rất rẻ)
        self.centroid_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=0.0, batch_first=True
        )

        self.norm_x = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.dim = dim
        self.G = G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d) — flattened spatial tokens
        Returns: (B, N, d) — enriched tokens, same shape
        """
        B, N, d = x.shape

        # ── Step 1: Routing matrix R ∈ ℝ^(B, N, G) ──────────────────
        R = F.softmax(self.W_r(x) / (self.G ** 0.5), dim=-1)  # (B, N, G)

        # ── Step 2: Soft centroids C ∈ ℝ^(B, G, d) ──────────────────
        # C[b, g, :] = weighted mean of tokens assigned to group g
        C = torch.bmm(R.transpose(1, 2), x)         # (B, G, d)
        C = self.norm_c(C)                           # stabilize

        # ── Step 3: Centroid self-attention (O(G²)) ───────────────────
        C, _ = self.centroid_attn(C, C, C, need_weights=False)

        # ── Step 4: Token → Centroid cross-attention (O(N*G)) ─────────
        Q = self.q_proj(x)   # (B, N, d)
        K = self.k_proj(C)   # (B, G, d)
        V = self.v_proj(C)   # (B, G, d)

        # Split into heads
        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.n_heads, d // self.n_heads) \
                    .transpose(1, 2)  # (B, h, seq, d_h)

        Q_h = split_heads(Q, N)   # (B, h, N, d_h)
        K_h = split_heads(K, self.G)  # (B, h, G, d_h)
        V_h = split_heads(V, self.G)

        # Scaled dot-product — N×G (cheap)
        scale = (d // self.n_heads) ** -0.5
        attn  = (Q_h @ K_h.transpose(-2, -1)) * scale   # (B, h, N, G)
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ V_h)                               # (B, h, N, d_h)
        out = out.transpose(1, 2).reshape(B, N, d)
        out = self.out_proj(out)

        # ── Step 5: Residual + norm ────────────────────────────────────
        return self.norm_x(x + out)