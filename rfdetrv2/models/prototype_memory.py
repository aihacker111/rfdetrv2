# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PrototypeMemory — EMA class prototype memory for query feature alignment.

Design decisions
----------------
* EMA buffers (not learnable parameters) so prototypes evolve without
  gradient interference and persist across DDP processes.
* Adaptive momentum: starts at 0.99 (fast update) → ramps to target momentum
  over `warmup_steps × 10` steps so early prototypes catch up quickly.
* Class-frequency weighting [ENH-2]: rare classes seen infrequently get
  higher loss weight, breaking the feedback loop of neglected classes.
* Prototype quality score [ENH-4]: tracks EMA update magnitude so
  unstable prototypes contribute less to the loss signal.
* Inter-class repulsion [ENH-3]: penalises cosine similarity between
  prototype pairs, pushing class embeddings apart.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rfdetrv2.util.misc import is_dist_avail_and_initialized


class PrototypeMemory(nn.Module):
    """EMA class prototype memory for query feature alignment.

    Args:
        num_classes:        Number of foreground classes (background excluded).
        feat_dim:           Query feature dimension (= hidden_dim).
        momentum:           Target EMA decay.  Adaptive ramp 0.99 → momentum.
        warmup_steps:       Steps before prototype loss is enabled.
        temperature:        Cosine-classifier temperature τ.  0.1 = hard negatives.
        repulsion_coef:     Weight for inter-class repulsion loss.
        use_freq_weight:    Enable class-frequency loss weighting.
        use_quality_weight: Enable prototype-quality loss weighting.
        use_repulsion:      Enable inter-class repulsion.
    """

    def __init__(
        self,
        num_classes:        int,
        feat_dim:           int,
        momentum:           float = 0.999,
        warmup_steps:       int   = 200,
        temperature:        float = 0.1,
        repulsion_coef:     float = 0.1,
        use_freq_weight:    bool  = True,
        use_quality_weight: bool  = True,
        use_repulsion:      bool  = True,
    ):
        super().__init__()

        self.register_buffer("prototypes",        torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("step",              torch.tensor(0, dtype=torch.long))
        self.register_buffer("proto_update_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("proto_variance",     torch.ones(num_classes, dtype=torch.float))

        self.num_classes        = num_classes
        self.feat_dim           = feat_dim
        self.momentum           = momentum
        self.warmup_steps       = warmup_steps
        self.temperature        = temperature
        self.repulsion_coef     = repulsion_coef
        self.use_freq_weight    = use_freq_weight
        self.use_quality_weight = use_quality_weight
        self.use_repulsion      = use_repulsion

    # ------------------------------------------------------------------
    # Adaptive momentum: slow at start → fast convergence
    # ------------------------------------------------------------------

    def _get_momentum(self) -> float:
        if self.warmup_steps <= 0:
            return self.momentum
        progress = min(1.0, float(self.step.item()) / max(1, self.warmup_steps * 10))
        return 0.99 + (self.momentum - 0.99) * progress

    # ------------------------------------------------------------------
    # EMA update (no grad, distributed-safe)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.numel() == 0:
            return

        device     = features.device
        labels     = labels.to(device)
        feats_norm = F.normalize(features.float().detach(), dim=-1)

        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            local_size = torch.tensor([feats_norm.shape[0]], dtype=torch.long, device=device)
            all_sizes  = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)

            max_size = int(max(s.item() for s in all_sizes))
            max_size = max(0, min(max_size, 65536))
            if max_size == 0:
                return

            pad_feat  = torch.zeros(max_size, self.feat_dim, device=device)
            pad_label = torch.full((max_size,), -1, dtype=labels.dtype, device=device)

            n = feats_norm.shape[0]
            if n > 0:
                pad_feat[:n]  = feats_norm
                pad_label[:n] = labels

            all_feats  = [torch.zeros(max_size, self.feat_dim, device=device) for _ in range(world_size)]
            all_labels = [torch.zeros(max_size, dtype=labels.dtype, device=device) for _ in range(world_size)]
            dist.all_gather(all_feats,  pad_feat)
            dist.all_gather(all_labels, pad_label)

            feats_norm = torch.cat(all_feats,  dim=0)
            labels     = torch.cat(all_labels, dim=0)
            valid      = labels >= 0
            feats_norm = feats_norm[valid]
            labels     = labels[valid]

        momentum = self._get_momentum()

        for cls_id_t in labels.unique():
            cls_id = cls_id_t.item()
            if cls_id < 0 or cls_id >= self.num_classes:
                continue

            cls_feat = feats_norm[labels == cls_id].mean(0)

            if not self.proto_initialized[cls_id]:
                self.prototypes[cls_id]        = cls_feat
                self.proto_initialized[cls_id] = True
            else:
                old_proto = self.prototypes[cls_id].clone()
                self.prototypes[cls_id] = momentum * old_proto + (1.0 - momentum) * cls_feat
                update_mag = (cls_feat - old_proto).norm().item()
                self.proto_variance[cls_id] = (
                    0.99 * self.proto_variance[cls_id].item() + 0.01 * update_mag
                )

            self.proto_update_count[cls_id] += 1

        self.step += 1

    # ------------------------------------------------------------------
    # Inter-class repulsion loss
    # ------------------------------------------------------------------

    def _repulsion_loss(self) -> torch.Tensor:
        idx = self.proto_initialized.nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, device=self.prototypes.device)
        protos = F.normalize(self.prototypes[idx].float(), dim=-1)
        sim    = protos @ protos.T
        mask   = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        return sim[mask].clamp(min=0).mean()

    # ------------------------------------------------------------------
    # Alignment loss
    # ------------------------------------------------------------------

    def loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute prototype alignment loss.

        L = L_pull + repulsion_coef × L_push

        Args:
            features: (M, D) matched query features (unnormalised).
            labels:   (M,)  GT class labels (0-indexed, no background).
        """
        if features.numel() == 0 or not self.proto_initialized.any():
            return features.sum() * 0.0
        if self.step < self.warmup_steps:
            return features.sum() * 0.0

        valid_mask = torch.tensor(
            [
                self.proto_initialized[l.item()].item()
                if 0 <= l.item() < self.num_classes else False
                for l in labels
            ],
            device=features.device,
        )
        if not valid_mask.any():
            return features.sum() * 0.0

        feats = features[valid_mask]
        lbls  = labels[valid_mask]

        feats_norm  = F.normalize(feats.float(),           dim=-1)
        protos_norm = F.normalize(self.prototypes.float(), dim=-1)
        logits      = feats_norm @ protos_norm.T / self.temperature

        loss_per = F.cross_entropy(logits, lbls.long(), reduction="none")

        if self.use_freq_weight:
            counts     = self.proto_update_count[lbls.long()].float().clamp(min=1)
            freq_w     = (1.0 / counts)
            freq_w     = (freq_w / freq_w.mean().clamp(min=1e-8)).clamp(max=5.0)
        else:
            freq_w = torch.ones_like(loss_per)

        if self.use_quality_weight:
            var    = self.proto_variance[lbls.long()].float().clamp(min=1e-4)
            qual_w = (1.0 / var)
            qual_w = (qual_w / qual_w.mean().clamp(min=1e-8)).clamp(max=5.0)
        else:
            qual_w = torch.ones_like(loss_per)

        w        = freq_w * qual_w
        w        = w / w.mean().clamp(min=1e-8)
        pull     = (loss_per * w).mean()

        if self.use_repulsion:
            return pull + self.repulsion_coef * self._repulsion_loss()
        return pull

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, feat_dim={self.feat_dim}, "
            f"momentum={self.momentum}, warmup={self.warmup_steps}, "
            f"temperature={self.temperature}, repulsion_coef={self.repulsion_coef}"
        )


class EnhancedPrototypeMemory(nn.Module):
    """EMA prototype memory with adaptive τ, dual (shadow) prototypes, hinge repulsion, hard negatives."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.999,
        warmup_steps: int = 200,
        temperature: float = 0.1,
        repulsion_coef: float = 0.1,
        repulsion_margin: float = 0.0,
        use_freq_weight: bool = True,
        use_quality_weight: bool = True,
        use_repulsion: bool = True,
        use_adaptive_temp: bool = True,
        use_dual_proto: bool = True,
        hard_neg_k: int = 5,
    ):
        super().__init__()

        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("proto_update_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("proto_variance", torch.ones(num_classes, dtype=torch.float32))

        if use_dual_proto:
            self.register_buffer("shadow_prototypes", torch.zeros(num_classes, feat_dim))
            self.register_buffer("shadow_initialized", torch.zeros(num_classes, dtype=torch.bool))

        if use_adaptive_temp:
            init_log_tau = math.log(temperature)
            self.log_tau = nn.Parameter(torch.full((num_classes,), init_log_tau))
            self._tau_min = math.log(0.01)
            self._tau_max = math.log(1.0)
        else:
            self.register_buffer(
                "log_tau", torch.full((num_classes,), math.log(temperature), dtype=torch.float32)
            )

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        self.repulsion_coef = repulsion_coef
        self.repulsion_margin = repulsion_margin
        self.use_freq_weight = use_freq_weight
        self.use_quality_weight = use_quality_weight
        self.use_repulsion = use_repulsion
        self.use_adaptive_temp = use_adaptive_temp
        self.use_dual_proto = use_dual_proto
        self.hard_neg_k = hard_neg_k

    def _get_momentum(self, cls_id: int) -> float:
        base = 0.99
        progress = min(1.0, float(self.step.item()) / max(1, self.warmup_steps * 10))
        if not self.proto_initialized[cls_id]:
            return base
        var = float(self.proto_variance[cls_id].item())
        stability = math.exp(-var)
        target = self.momentum
        return base + (target - base) * stability * progress

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.numel() == 0:
            return

        device = features.device
        labels = labels.to(device)
        feats_norm = F.normalize(features.float().detach(), dim=-1)

        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            local_size = torch.tensor([feats_norm.shape[0]], dtype=torch.long, device=device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            max_size = int(max(s.item() for s in all_sizes))
            max_size = max(0, min(max_size, 65536))
            if max_size == 0:
                return
            pad_feat = torch.zeros(max_size, self.feat_dim, device=device)
            pad_label = torch.full((max_size,), -1, dtype=labels.dtype, device=device)
            n = feats_norm.shape[0]
            if n > 0:
                pad_feat[:n] = feats_norm
                pad_label[:n] = labels
            all_feats = [torch.zeros(max_size, self.feat_dim, device=device) for _ in range(world_size)]
            all_labels = [torch.zeros(max_size, dtype=labels.dtype, device=device) for _ in range(world_size)]
            dist.all_gather(all_feats, pad_feat)
            dist.all_gather(all_labels, pad_label)
            feats_norm = torch.cat(all_feats, dim=0)
            labels = torch.cat(all_labels, dim=0)
            valid = labels >= 0
            feats_norm = feats_norm[valid]
            labels = labels[valid]

        for cls_id_t in labels.unique():
            cls_id = int(cls_id_t.item())
            if cls_id < 0 or cls_id >= self.num_classes:
                continue

            cls_feat = feats_norm[labels == cls_id].mean(0)
            mom = self._get_momentum(cls_id)

            if not self.proto_initialized[cls_id]:
                self.prototypes[cls_id] = cls_feat
                self.proto_initialized[cls_id] = True
            else:
                old = self.prototypes[cls_id].clone()
                self.prototypes[cls_id] = mom * old + (1.0 - mom) * cls_feat
                update_mag = (cls_feat - old).norm().item()
                self.proto_variance[cls_id] = (
                    0.99 * self.proto_variance[cls_id] + 0.01 * update_mag
                )

            if self.use_dual_proto:
                shadow_m = min(self.momentum + 0.0009, 0.9999)
                if not self.shadow_initialized[cls_id]:
                    self.shadow_prototypes[cls_id] = self.prototypes[cls_id].clone()
                    self.shadow_initialized[cls_id] = True
                else:
                    self.shadow_prototypes[cls_id] = (
                        shadow_m * self.shadow_prototypes[cls_id]
                        + (1.0 - shadow_m) * self.prototypes[cls_id]
                    )

            self.proto_update_count[cls_id] += 1

        self.step += 1

    def _repulsion_loss(self) -> torch.Tensor:
        idx = self.proto_initialized.nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, device=self.prototypes.device)
        protos = F.normalize(self.prototypes[idx].float(), dim=-1)
        sim = protos @ protos.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        hinge = (sim[mask] - self.repulsion_margin).clamp(min=0)
        return hinge.mean()

    def _hard_neg_weights(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        k = min(self.hard_neg_k, self.num_classes - 1)
        if k <= 0:
            return torch.ones(len(labels), device=logits.device, dtype=logits.dtype)

        with torch.no_grad():
            neg_logits = logits.clone()
            neg_logits.scatter_(1, labels.unsqueeze(1).long(), float("-inf"))
            top_neg = neg_logits.topk(k, dim=1).values
            hard_score = top_neg.mean(dim=1)
            w = torch.softmax(hard_score, dim=0) * float(len(labels))
        return w.clamp(max=5.0)

    def loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0 or not self.proto_initialized.any():
            return features.sum() * 0.0
        if self.step < self.warmup_steps:
            return features.sum() * 0.0

        lbl = labels.long()
        valid_mask = (lbl >= 0) & (lbl < self.num_classes) & self.proto_initialized[lbl]
        if not valid_mask.any():
            return features.sum() * 0.0

        feats = features[valid_mask]
        lbls = lbl[valid_mask]

        feats_norm = F.normalize(feats.float(), dim=-1)

        proto_mat = self.prototypes.float()
        if self.use_dual_proto and self.shadow_initialized.any():
            use_shadow = self.shadow_initialized & self.proto_initialized
            proto_mat = proto_mat.clone()
            proto_mat[use_shadow] = self.shadow_prototypes[use_shadow]

        protos_norm = F.normalize(proto_mat, dim=-1)

        if self.use_adaptive_temp:
            log_tau = self.log_tau.clamp(self._tau_min, self._tau_max)
        else:
            log_tau = self.log_tau
        tau_per_sample = log_tau[lbls].exp()

        raw_logits = feats_norm @ protos_norm.T
        logits = raw_logits / tau_per_sample.unsqueeze(1)

        loss_per = F.cross_entropy(logits, lbls, reduction="none")

        if self.use_freq_weight:
            counts = self.proto_update_count[lbls].float().clamp(min=1)
            freq_w = 1.0 / counts
            freq_w = (freq_w / freq_w.mean().clamp(min=1e-8)).clamp(max=5.0)
        else:
            freq_w = torch.ones_like(loss_per)

        if self.use_quality_weight:
            var = self.proto_variance[lbls].float().clamp(min=1e-4)
            qual_w = 1.0 / var
            qual_w = (qual_w / qual_w.mean().clamp(min=1e-8)).clamp(max=5.0)
        else:
            qual_w = torch.ones_like(loss_per)

        hard_w = self._hard_neg_weights(raw_logits, lbls)
        w = freq_w * qual_w * hard_w
        w = w / w.mean().clamp(min=1e-8)
        pull = (loss_per * w).mean()

        if self.use_repulsion:
            return pull + self.repulsion_coef * self._repulsion_loss()
        return pull

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, feat_dim={self.feat_dim}, "
            f"adaptive_temp={self.use_adaptive_temp}, dual_proto={self.use_dual_proto}, "
            f"hard_neg_k={self.hard_neg_k}"
        )
