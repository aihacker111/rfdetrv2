# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Cortical Perceptual Feature Enhancement (CPFE)
===============================================

Module sinh học-inspired để cải thiện chất lượng features từ DINOv3 backbone
trước khi đưa vào MultiScaleProjector → LW-DETR Transformer.

Ba thành phần, mỗi thành phần mô phỏng một cơ chế cụ thể trong hệ thống
thị giác của người:

┌─────────────────────────────────────────────────────────────────────┐
│  Cơ chế sinh học          │  Thành phần CPFE                        │
├───────────────────────────┼─────────────────────────────────────────┤
│  Tế bào ganglion võng mạc │  SDG — Spectral Decomposition Gate      │
│  Center-Surround (DoG)    │  Phân tách low-freq + high-freq          │
│                           │  → gating theo tầm quan trọng           │
├───────────────────────────┼─────────────────────────────────────────┤
│  Lateral inhibition V1    │  DN — Divisive Normalization             │
│  Tế bào ức chế lẫn nhau  │  Chia mỗi feature cho local RMS energy  │
│                           │  → sparse, high-contrast activation      │
├───────────────────────────┼─────────────────────────────────────────┤
│  Predictive Coding        │  TPR — Top-Down Predictive Refinement    │
│  Cortical feedback        │  Scale sâu (global) dự đoán scale nông  │
│  (Rao & Ballard, 1999)   │  → prediction error truyền ngược lên    │
└───────────────────────────┴─────────────────────────────────────────┘

Toán học tóm tắt
-----------------

SDG:
  F_surround = DWConv_{k×k}(F)           # low-freq (Gaussian-like)
  F_cs       = F - F_surround            # high-freq (DoG response)
  g_low      = σ(W_l · GAP(F_surround))  # channel gate from context
  g_high     = σ(W_h · GMP(F_cs))        # channel gate from detail
  F_sdg      = g_low ⊙ F_surround + g_high ⊙ F_cs

DN:
  E_spatial = sqrt(DWConv_{k×k}(F²) + ε)          # local RMS energy
  E_cross   = sqrt(mean_c(F²) + ε)                 # cross-channel energy
  E_total   = α·E_spatial + (1-α)·E_cross           # α learnable ∈ [0,1]
  F_dn      = F / (E_total + ε)

TPR (scale i = fine, scale i+1 = coarse/deep):
  P         = Conv(F_{i+1})               # top-down prediction
  E_pred    = F_i - P                      # prediction error
  g         = σ(Conv(P))                   # context gate
  F_tpr     = F_i + g ⊙ E_pred            # residual gating

Gradient property của TPR:
  ∂L/∂F_i = ∂L/∂F_tpr · (1 + g)
  → gradient được khuếch đại tại vùng coarse scale xác nhận quan trọng.

Tích hợp:
  DINOv3 → [F_0, F_1, ..., F_{L-1}]  (L scales, cùng spatial resolution)
         ↓ CPFE
         [F̃_0, F̃_1, ..., F̃_{L-1}]
         ↓ MultiScaleProjector
         [P3, P4, P5, (P6)]
         ↓ LW-DETR Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component 1: Spectral Decomposition Gate (SDG)
# ---------------------------------------------------------------------------

class SpectralDecompositionGate(nn.Module):
    """Center-Surround decomposition + frequency-selective channel gating.

    Lấy cảm hứng từ tế bào ganglion võng mạc (Difference of Gaussians).

    Args:
        channels:    số channels của feature map đầu vào.
        kernel_size: kích thước depthwise conv để tạo surround (thường 7 hoặc 5).
        reduction:   ratio giảm dim cho channel gating bottleneck.
    """

    def __init__(self, channels: int, kernel_size: int = 7, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)

        # Surround: depthwise conv mô phỏng Gaussian blur (low-freq filter)
        self.surround = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )

        # Channel gating từ low-frequency context (GAP → MLP → sigmoid)
        self.gate_low = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

        # Channel gating từ high-frequency detail (GMP → MLP → sigmoid)
        self.gate_high = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

        # Khởi tạo surround weights gần với uniform average (Gaussian-init)
        nn.init.constant_(self.surround.weight, 1.0 / (kernel_size * kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Center-Surround decomposition (DoG)
        f_surround = self.surround(x)      # low-freq  (B, C, H, W)
        f_cs       = x - f_surround        # high-freq (B, C, H, W)

        # Global pooling → channel statistics
        mu_low  = f_surround.mean(dim=(-2, -1))   # GAP: (B, C)
        mu_high = f_cs.amax(dim=(-2, -1))          # GMP peak: (B, C)

        # Frequency-selective gates
        g_low  = self.gate_low(mu_low).unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)
        g_high = self.gate_high(mu_high).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)

        return g_low * f_surround + g_high * f_cs


# ---------------------------------------------------------------------------
# Component 2: Divisive Normalization (DN)
# ---------------------------------------------------------------------------

class DivisiveNormalization(nn.Module):
    """Lateral inhibition via divisive normalization.

    Mô phỏng cơ chế V1: mỗi neuron được chia cho tổng năng lượng của
    các neuron lân cận (spatial + cross-channel).

    Công thức sinh học (Heeger 1992):
        r_i = x_i^n / (σ^n + Σ_j w_ij x_j^n)

    Xấp xỉ học được:
        E_spatial = sqrt(DWConv(F²) + ε)
        E_cross   = sqrt(mean_c(F²) + ε)
        E_total   = α·E_spatial + (1-α)·E_cross
        F_dn      = F / (E_total + ε)

    Args:
        channels:    số channels.
        kernel_size: spatial extent của local energy pooling.
        eps:         numerical stability constant.
    """

    def __init__(self, channels: int, kernel_size: int = 5, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Depthwise conv để tính local energy trong spatial neighborhood
        self.energy_pool = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )

        # α: mix ratio giữa within-channel và cross-channel inhibition
        # khởi tạo = 0.5 (balanced), học được trong training
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Init energy_pool như average filter (uniform kernel)
        nn.init.constant_(self.energy_pool.weight, 1.0 / (kernel_size * kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.pow(2)

        # Within-channel local spatial energy
        e_spatial = (self.energy_pool(x2) + self.eps).sqrt()  # (B, C, H, W)

        # Cross-channel energy (mean over channels → broadcast)
        e_cross = (x2.mean(dim=1, keepdim=True) + self.eps).sqrt()  # (B, 1, H, W)
        e_cross = e_cross.expand_as(e_spatial)

        # Blend spatial and cross-channel inhibition
        alpha = self.alpha.clamp(0.0, 1.0)
        e_total = alpha * e_spatial + (1.0 - alpha) * e_cross

        return x / (e_total + self.eps)


# ---------------------------------------------------------------------------
# Component 3: Top-Down Predictive Refinement (TPR)
# ---------------------------------------------------------------------------

class TopDownPredictiveRefinement(nn.Module):
    """Cortical feedback: deep (global) scale dự đoán shallow (local) scale.

    Lấy cảm hứng từ Predictive Coding (Rao & Ballard, 1999):
      - Vùng cao hơn (global context) gửi top-down prediction xuống.
      - Vùng thấp hơn tính prediction error: E = actual - predicted.
      - Context gate g quyết định bao nhiêu error được truyền lên.

    Công thức:
        P      = Conv(F_coarse)           # top-down prediction
        E_pred = F_fine - P              # prediction error
        g      = σ(Conv(P))              # context gate từ prediction
        F_out  = F_fine + g ⊙ E_pred    # residual gating

    Gradient property:
        ∂L/∂F_fine = ∂L/∂F_out · (1 + g)  ≥ 1
        → gradient luôn ≥ gradient gốc tại vùng được confirm bởi context.

    Args:
        channels: số channels (fine và coarse phải cùng channels).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pred_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Init gate_conv gần 0 → bắt đầu gần identity (residual ≈ 0)
        nn.init.constant_(self.gate_conv.weight, 0.0)

    def forward(self, f_fine: torch.Tensor, f_coarse: torch.Tensor) -> torch.Tensor:
        # Align spatial resolution (no-op nếu đã cùng shape — trường hợp DINOv3)
        if f_coarse.shape[-2:] != f_fine.shape[-2:]:
            f_up = F.interpolate(
                f_coarse, size=f_fine.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        else:
            f_up = f_coarse

        # Top-down prediction
        pred   = self.pred_conv(f_up)          # (B, C, H, W)
        e_pred = f_fine - pred                  # prediction error

        # Context gate
        g = torch.sigmoid(self.gate_conv(pred))  # (B, C, H, W) ∈ [0, 1]

        return f_fine + g * e_pred


# ---------------------------------------------------------------------------
# CPFE — Full Module
# ---------------------------------------------------------------------------

class CorticalPerceptualFeatureEnhancement(nn.Module):
    """Cortical Perceptual Feature Enhancement (CPFE).

    Kết hợp 3 cơ chế sinh học để nâng cao chất lượng backbone features:

        SDG → DN → TPR (top-down, từ deep về shallow)

    Thiết kế để insert giữa DINOv3 encoder và MultiScaleProjector:

        DINOv3 → [F_0, ..., F_{L-1}]
              ↓ CPFE
              [F̃_0, ..., F̃_{L-1}]
              ↓ MultiScaleProjector
              [P3, P4, P5]

    Trong DINOv3, tất cả scales có cùng spatial resolution (ViT không
    downsample). F_0 = layer nông (local features), F_{L-1} = layer sâu
    (global/semantic features).

    TPR hoạt động theo hướng: F_{L-1} (global) → cải thiện → F_0 (local).

    Args:
        in_channels_list: list channel dims từ backbone (1 per scale).
                          Với DINOv3, thường [C, C, C, C] với C = 384 hoặc 768.
        sdg_kernel:       kernel size cho SDG surround conv (default 7).
        dn_kernel:        kernel size cho DN energy pooling (default 5).
        use_sdg:          bật/tắt SDG component.
        use_dn:           bật/tắt DN component.
        use_tpr:          bật/tắt TPR component.
        reduction:        bottleneck reduction ratio cho SDG channel gating.
    """

    def __init__(
        self,
        in_channels_list: list,
        sdg_kernel: int = 7,
        dn_kernel: int = 5,
        use_sdg: bool = True,
        use_dn: bool = True,
        use_tpr: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        self.use_sdg = use_sdg
        self.use_dn  = use_dn
        self.use_tpr = use_tpr

        n = len(in_channels_list)

        if use_sdg:
            self.sdg_modules = nn.ModuleList([
                SpectralDecompositionGate(c, kernel_size=sdg_kernel, reduction=reduction)
                for c in in_channels_list
            ])

        if use_dn:
            self.dn_modules = nn.ModuleList([
                DivisiveNormalization(c, kernel_size=dn_kernel)
                for c in in_channels_list
            ])

        # TPR: n-1 modules, mỗi module kết nối scale i (fine) ← scale i+1 (coarse)
        if use_tpr and n > 1:
            self.tpr_modules = nn.ModuleList([
                TopDownPredictiveRefinement(in_channels_list[i])
                for i in range(n - 1)
            ])

    def forward(self, feats: list) -> list:
        """
        Args:
            feats: list of (B, C_i, H_i, W_i) — backbone features per scale.
                   feats[0] = shallowest layer (local), feats[-1] = deepest (global).

        Returns:
            list of same shape tensors, enhanced.
        """
        # 1. SDG — per-scale frequency decomposition + gating
        if self.use_sdg:
            feats = [self.sdg_modules[i](f) for i, f in enumerate(feats)]

        # 2. DN — per-scale divisive normalization
        if self.use_dn:
            feats = [self.dn_modules[i](f) for i, f in enumerate(feats)]

        # 3. TPR — top-down predictive refinement (deep → shallow)
        #    Xử lý từ scale sâu nhất về nông nhất: i = L-2, L-3, ..., 0
        if self.use_tpr and hasattr(self, 'tpr_modules'):
            n = len(feats)
            for i in range(n - 2, -1, -1):
                feats[i] = self.tpr_modules[i](
                    f_fine=feats[i],
                    f_coarse=feats[i + 1],
                )

        return feats
