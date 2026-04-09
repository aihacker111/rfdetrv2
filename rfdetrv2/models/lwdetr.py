# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""LW-DETR model and criterion classes."""

import copy
import math
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rfdetrv2.models.backbone import build_backbone
from rfdetrv2.models.matcher import build_matcher
from rfdetrv2.models.segmentation_head import (
    SegmentationHead,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from rfdetrv2.models.transformers import build_transformer
from rfdetrv2.util import box_ops
from rfdetrv2.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


# ---------------------------------------------------------------------------
# SuperpositionAwarePrototypeMemory
# ---------------------------------------------------------------------------

class SuperpositionAwarePrototypeMemory(nn.Module):
    """EMA prototype memory that handles co-located / overlapping classes.

    Loss components:
        [A] loss_proto_pull     — cosine cross-entropy to correct prototype
        [B] loss_proto_ortho    — co-occurrence-weighted orthogonality
        [C] loss_proto_disambig — IoU-conditioned feature separation
        [D] loss_proto_sparse   — L1 sparsity on prototype coefficients

    Args:
        num_classes:    number of foreground classes
        feat_dim:       query feature dimension (= hidden_dim)
        momentum:       EMA decay (adaptive: 0.99 → momentum)
        warmup_steps:   steps before any loss is applied
        temperature:    cosine-classifier temperature (τ)
        ortho_coef:     weight for [B] orthogonality loss
        disambig_coef:  weight for [C] disambiguation loss
        sparse_coef:    weight for [D] sparsity loss
        iou_threshold:  minimum IoU to trigger disambiguation
    """

    def __init__(
        self,
        num_classes:   int,
        feat_dim:      int,
        momentum:      float = 0.999,
        warmup_steps:  int   = 200,
        temperature:   float = 0.1,
        ortho_coef:    float = 0.1,
        disambig_coef: float = 0.1,
        sparse_coef:   float = 0.05,
        iou_threshold: float = 0.3,
    ):
        super().__init__()

        self.register_buffer("prototypes",        torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("step",              torch.tensor(0, dtype=torch.long))
        # Co-occurrence matrix: how often class i and j appear in same image
        self.register_buffer("cooccurrence",      torch.zeros(num_classes, num_classes))

        self.num_classes   = num_classes
        self.feat_dim      = feat_dim
        self.momentum      = momentum
        self.warmup_steps  = warmup_steps
        self.temperature   = temperature
        self.ortho_coef    = ortho_coef
        self.disambig_coef = disambig_coef
        self.sparse_coef   = sparse_coef
        self.iou_threshold = iou_threshold

    def _get_momentum(self) -> float:
        if self.warmup_steps <= 0:
            return self.momentum
        progress = min(1.0, float(self.step.item()) / max(1, self.warmup_steps * 10))
        return 0.99 + (self.momentum - 0.99) * progress

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels:   torch.Tensor,
        labels_per_image=None,
    ) -> None:
        if features.numel() == 0:
            return

        device     = features.device
        labels     = labels.to(device)
        feats_norm = F.normalize(features.float().detach(), dim=-1)

        # Distributed gather
        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            local_size = torch.tensor([feats_norm.shape[0]], dtype=torch.long, device=device)
            all_sizes  = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)

            max_size = max(0, min(int(max(s.item() for s in all_sizes)), 65536))
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
            dist.all_gather(all_feats, pad_feat)
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
                self.prototypes[cls_id] = momentum * self.prototypes[cls_id] + (1.0 - momentum) * cls_feat

        # [B] Update co-occurrence matrix
        if labels_per_image is not None:
            for img_labels in labels_per_image:
                img_labels = img_labels.to(device)
                unique_cls = img_labels[(img_labels >= 0) & (img_labels < self.num_classes)].unique()
                if len(unique_cls) < 2:
                    continue
                for i in range(len(unique_cls)):
                    for j in range(i + 1, len(unique_cls)):
                        ci, cj = unique_cls[i].long(), unique_cls[j].long()
                        self.cooccurrence[ci, cj] += 1
                        self.cooccurrence[cj, ci] += 1

        self.step += 1

    # ------------------------------------------------------------------
    # [B] Co-occurrence orthogonality loss
    # ------------------------------------------------------------------

    def _cooccurrence_ortho_loss(self) -> torch.Tensor:
        """Prototypes of frequently co-occurring classes should be orthogonal."""
        idx = self.proto_initialized.nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, device=self.prototypes.device)

        protos = F.normalize(self.prototypes[idx].float(), dim=-1)   # (K, D)
        sim    = protos @ protos.T                                    # (K, K)

        cooc      = self.cooccurrence[idx][:, idx].float()
        cooc_norm = cooc / cooc.max().clamp(min=1.0)

        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        return (sim.abs() * cooc_norm)[mask].mean()

    # ------------------------------------------------------------------
    # [C] IoU-conditioned disambiguation loss
    # ------------------------------------------------------------------

    def _iou_disambiguation_loss(
        self,
        features: torch.Tensor,
        labels:   torch.Tensor,
        boxes:    torch.Tensor,
    ) -> torch.Tensor:
        """Feature pairs with high IoU and different labels should be dissimilar."""
        if boxes is None or len(boxes) < 2:
            return torch.tensor(0.0, device=features.device)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        iou_mat, _ = box_ops.box_iou(boxes_xyxy, boxes_xyxy)   # (M, M)

        feats_norm = F.normalize(features.float(), dim=-1)
        M          = features.shape[0]

        loss  = torch.tensor(0.0, device=features.device)
        count = 0
        for i in range(M):
            for j in range(i + 1, M):
                if labels[i] == labels[j]:
                    continue
                iou = iou_mat[i, j].item()
                if iou < self.iou_threshold:
                    continue
                sim   = (feats_norm[i] * feats_norm[j]).sum()
                loss  = loss + sim.clamp(min=0) * iou
                count += 1

        return loss / max(count, 1)

    # ------------------------------------------------------------------
    # [D] Sparse decomposition loss
    # ------------------------------------------------------------------

    def _sparse_decomposition_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Features should decompose sparsely over prototypes (L1 on coefficients)."""
        if not self.proto_initialized.any():
            return torch.tensor(0.0, device=features.device)

        idx        = self.proto_initialized.nonzero(as_tuple=True)[0]
        protos     = F.normalize(self.prototypes[idx].float(), dim=-1)   # (K, D)
        feats_norm = F.normalize(features.float(), dim=-1)               # (M, D)
        coeffs     = feats_norm @ protos.T                               # (M, K)
        return coeffs.clamp(min=0).mean()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        features: torch.Tensor,
        labels:   torch.Tensor,
        boxes:    torch.Tensor = None,
    ) -> dict:
        """Return dict of four loss terms."""
        zero = features.sum() * 0.0
        empty = {'loss_proto_pull': zero, 'loss_proto_ortho': zero,
                 'loss_proto_disambig': zero, 'loss_proto_sparse': zero}

        if features.numel() == 0 or not self.proto_initialized.any():
            return empty
        if self.step < self.warmup_steps:
            return empty

        valid_mask = torch.tensor(
            [self.proto_initialized[l.item()].item() if 0 <= l.item() < self.num_classes else False
             for l in labels],
            device=features.device,
        )
        if not valid_mask.any():
            return empty

        feats = features[valid_mask]
        lbls  = labels[valid_mask]
        boxes_v = boxes[valid_mask] if boxes is not None and len(boxes) == len(valid_mask) else None

        feats_norm  = F.normalize(feats.float(),           dim=-1)
        protos_norm = F.normalize(self.prototypes.float(), dim=-1)
        logits      = feats_norm @ protos_norm.T / self.temperature

        pull_loss    = F.cross_entropy(logits, lbls.long())
        ortho_loss   = self._cooccurrence_ortho_loss()
        disambig_loss = self._iou_disambiguation_loss(feats, lbls, boxes_v)
        sparse_loss  = self._sparse_decomposition_loss(feats)

        return {
            'loss_proto_pull':    pull_loss,
            'loss_proto_ortho':   ortho_loss   * self.ortho_coef,
            'loss_proto_disambig': disambig_loss * self.disambig_coef,
            'loss_proto_sparse':  sparse_loss  * self.sparse_coef,
        }

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, feat_dim={self.feat_dim}, "
            f"momentum={self.momentum}, warmup_steps={self.warmup_steps}, "
            f"temperature={self.temperature}, ortho_coef={self.ortho_coef}, "
            f"disambig_coef={self.disambig_coef}, sparse_coef={self.sparse_coef}"
        )


# ---------------------------------------------------------------------------
# LWDETR
# ---------------------------------------------------------------------------

class LWDETR(nn.Module):
    """RF-DETR main model."""

    def __init__(self,
                 backbone,
                 transformer,
                 segmentation_head,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat     = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)
        self.backbone   = backbone
        self.aux_loss   = aux_loss
        self.group_detr = group_detr

        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False
        
        
    # ------------------------------------------------------------------
    # Standard methods
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes):
        base = self.class_embed.weight.shape[0]
        num_repeats = int(math.ceil(num_classes / base))
        self.class_embed.weight.data = self.class_embed.weight.data.repeat(num_repeats, 1)[:num_classes]
        self.class_embed.bias.data   = self.class_embed.bias.data.repeat(num_repeats)[:num_classes]
        if self.two_stage:
            for enc_out_class_embed in self.transformer.enc_out_class_embed:
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data.repeat(num_repeats, 1)[:num_classes]
                enc_out_class_embed.bias.data   = enc_out_class_embed.bias.data.repeat(num_repeats)[:num_classes]

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, samples: NestedTensor, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight     = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight     = self.query_feat.weight[:self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = (self.segmentation_head.sparse_forward
                            if self.training else self.segmentation_head.forward)

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy  = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh    = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord       = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            # Expose last-layer query features cho prototype alignment loss
            # Chỉ cần khi training — không overhead ở inference
            if self.training:
                out['pred_queries'] = hs[-1]   # (B, N, D)

            if self.segmentation_head is not None:
                out['pred_masks'] = outputs_masks[-1]
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )

        if self.two_stage:
            group_detr  = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc     = torch.cat([
                self.transformer.enc_out_class_embed[g](hs_enc_list[g])
                for g in range(group_detr)
            ], dim=1)

            if self.segmentation_head is not None:
                masks_enc = seg_head_fwd(
                    features[0].tensors, [hs_enc,],
                    samples.tensors.shape[-2:], skip_blocks=True,
                )[0]

            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['enc_outputs']['pred_masks'] = masks_enc
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['pred_masks'] = masks_enc

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight     = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        outputs_masks = None
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy  = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh    = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord       = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs,], tensors.shape[-2:])[0]
        else:
            assert self.two_stage
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0], [hs_enc,], tensors.shape[-2:], skip_blocks=True)[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        if outputs_masks is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])]
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'):
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else:
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


# ---------------------------------------------------------------------------
# SetCriterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,
                 mask_point_sample_ratio: int = 16,
                 prototype_memory: SuperpositionAwarePrototypeMemory = None,
                 prototype_loss_coef: float = 1.0,
                 ):
        super().__init__()
        self.num_classes                  = num_classes
        self.matcher                      = matcher
        self.weight_dict                  = weight_dict
        self.losses                       = losses
        self.focal_alpha                  = focal_alpha
        self.group_detr                   = group_detr
        self.sum_group_losses             = sum_group_losses
        self.use_varifocal_loss           = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss                  = ia_bce_loss
        self.mask_point_sample_ratio      = mask_point_sample_ratio
        self.prototype_memory             = prototype_memory
        self.prototype_loss_coef          = prototype_loss_coef

    # ------------------------------------------------------------------
    # Classification loss
    # ------------------------------------------------------------------

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious    = iou_targets.clone().detach()
            prob        = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            loss_ce = (neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)).sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious_func = iou_targets.clone().detach()
            cls_iou_func_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls = cls_iou_func_targets / (
                cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(
                src_logits, norm_cls, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            cls_iou_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(
                src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # ------------------------------------------------------------------
    # Box losses
    # ------------------------------------------------------------------

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device      = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred   = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {'cardinality_error': F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx          = self._get_src_permutation_idx(indices)
        src_boxes    = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    # ------------------------------------------------------------------
    # Mask loss
    # ------------------------------------------------------------------

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert 'pred_masks' in outputs
        idx        = self._get_src_permutation_idx(indices)
        pred_masks = outputs['pred_masks']

        if isinstance(pred_masks, torch.Tensor):
            src_masks = pred_masks[idx]
        else:
            spatial_features = outputs["pred_masks"]["spatial_features"]
            query_features   = outputs["pred_masks"]["query_features"]
            bias             = outputs["pred_masks"]["bias"]
            if idx[0].numel() == 0:
                src_masks = torch.tensor([], device=spatial_features.device)
            else:
                batched = []
                pbc = idx[0].unique(return_counts=True)[1]
                bi  = torch.cat((torch.zeros_like(pbc[:1]), pbc), 0).cumsum(0)
                for i in range(pbc.shape[0]):
                    bi_i = idx[0][bi[i]:bi[i+1]]
                    bx_i = idx[1][bi[i]:bi[i+1]]
                    q_i  = query_features[(bi_i, bx_i)]
                    sf_i = spatial_features[idx[0][bi[i+1]-1]]
                    batched.append(torch.einsum("chw,nc->nhw", sf_i, q_i) + bias)
                src_masks = torch.cat(batched)

        if src_masks.numel() == 0:
            return {'loss_mask_ce': src_masks.sum(), 'loss_mask_dice': src_masks.sum()}

        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_masks    = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            src_masks.shape[-2],
            src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio,
        )
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, lambda logits: calculate_uncertainty(logits), num_points, 3, 0.75)

        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        with torch.no_grad():
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False, mode="nearest").squeeze(1)

        losses = {
            "loss_mask_ce":   sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }
        del src_masks, target_masks
        return losses

    # ------------------------------------------------------------------
    # Prototype Alignment Loss  ← thay thế InfoNCE
    # ------------------------------------------------------------------

    def loss_prototype_align(self, outputs, targets, indices, num_boxes):
        if 'pred_queries' not in outputs or self.prototype_memory is None:
            return {}

        query_feats = outputs['pred_queries']   # (B, N, D)
        device = query_feats.device

        all_feats:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_boxes:  list[torch.Tensor] = []

        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_idx = src_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            lbls  = targets[b_idx]['labels'][tgt_idx]
            valid = lbls < self.num_classes
            if not valid.any():
                continue
            all_feats.append(query_feats[b_idx, src_idx[valid]])
            all_labels.append(lbls[valid])
            if 'boxes' in targets[b_idx]:
                all_boxes.append(targets[b_idx]['boxes'][tgt_idx][valid])

        if len(all_feats) == 0:
            zero = query_feats.sum() * 0.0
            return {'loss_proto_pull': zero, 'loss_proto_ortho': zero,
                    'loss_proto_disambig': zero, 'loss_proto_sparse': zero}

        feats  = torch.cat(all_feats,  dim=0)
        labels = torch.cat(all_labels, dim=0)
        boxes  = torch.cat(all_boxes,  dim=0) if all_boxes else None

        labels_per_image = [t['labels'] for t in targets]
        self.prototype_memory.update(feats, labels, labels_per_image=labels_per_image)
        return self.prototype_memory.loss(feats, labels, boxes=boxes)

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx   = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels':          self.loss_labels,
            'cardinality':     self.loss_cardinality,
            'boxes':           self.loss_boxes,
            'masks':           self.loss_masks,
            'prototype_align': self.loss_prototype_align,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        group_detr          = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices             = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes *= group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_i = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    # prototype_align chỉ apply ở last decoder layer
                    if loss == 'prototype_align':
                        continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_boxes, **kwargs)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices_e   = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                if loss == 'prototype_align':
                    continue
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, targets, indices_e, num_boxes, **kwargs)
                losses.update({k + '_enc': v for k, v in l_dict.items()})

        return losses


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob         = inputs.sigmoid()
    focal_weight = (targets * (targets > 0.0).float()
                    + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float())
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (ce_loss * focal_weight).mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss    = ce_loss * (torch.abs(targets - prob) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs      = inputs.sigmoid().flatten(1)
    numerator   = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss        = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


# ---------------------------------------------------------------------------
# PostProcess
# ---------------------------------------------------------------------------

class PostProcess(nn.Module):
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores     = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels     = topk_indexes % out_logits.shape[2]
        boxes      = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes      = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct    = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes        = boxes * scale_fct[:, None, :]

        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                k_idx   = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i], 0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]))
                h, w    = target_sizes[i].tolist()
                masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)),
                                        mode='bilinear', align_corners=False)
                results.append({'scores': scores[i], 'labels': labels[i],
                                'boxes': boxes[i], 'masks': masks_i > 0.0})
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b}
                       for s, l, b in zip(scores, labels, boxes)]
        return results


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# build_model / build_criterion_and_postprocessors
# ---------------------------------------------------------------------------

def build_model(args):
    num_classes = args.num_classes + 1
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=(args.shape if hasattr(args, 'shape')
                      else (args.resolution, args.resolution) if hasattr(args, 'resolution')
                      else (640, 640)),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov3_weights=True,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        use_windowed_attn=getattr(args, "use_windowed_attn", False),
        use_convnext_projector=getattr(args, "use_convnext_projector", True),
        use_fsca=getattr(args, "use_fsca", False),
        base_grid_size=getattr(args, "resolution", 0) // getattr(args, "patch_size", 16),
        fsca_heads=getattr(args, "fsca_heads", 8),
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = (
        SegmentationHead(args.hidden_dim, args.dec_layers,
                         downsample_ratio=args.mask_downsample_ratio)
        if args.segmentation_head else None
    )

    model = LWDETR(
        backbone, transformer, segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    return model


def build_criterion_and_postprocessors(args):
    device  = torch.device(args.device)
    matcher = build_matcher(args)
 
    weight_dict = {
        'loss_ce':   args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    if args.segmentation_head:
        weight_dict['loss_mask_ce']   = args.mask_ce_loss_coef
        weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef
 
    # ------------------------------------------------------------------
    # Prototype alignment loss — SuperpositionAwarePrototypeMemory
    # ------------------------------------------------------------------
    prototype_memory = None
    use_proto = getattr(args, 'use_prototype_align', True)

    if use_proto:
        proto_coef    = getattr(args, 'prototype_loss_coef',     0.1)
        ortho_coef    = getattr(args, 'prototype_ortho_coef',    0.1)
        disambig_coef = getattr(args, 'prototype_disambig_coef', 0.1)
        sparse_coef   = getattr(args, 'prototype_sparse_coef',   0.05)

        weight_dict['loss_proto_pull']     = proto_coef
        weight_dict['loss_proto_ortho']    = 1.0
        weight_dict['loss_proto_disambig'] = 1.0
        weight_dict['loss_proto_sparse']   = 1.0

        prototype_memory = SuperpositionAwarePrototypeMemory(
            num_classes   = args.num_classes,
            feat_dim      = args.hidden_dim,
            momentum      = getattr(args, 'prototype_momentum',      0.999),
            warmup_steps  = getattr(args, 'prototype_warmup_steps',  200),
            temperature   = getattr(args, 'prototype_temperature',   0.1),
            ortho_coef    = ortho_coef,
            disambig_coef = disambig_coef,
            sparse_coef   = sparse_coef,
            iou_threshold = getattr(args, 'prototype_iou_threshold', 0.3),
        )
        prototype_memory.to(device)
 
    # ------------------------------------------------------------------
    # Aux loss weight dict
    # ------------------------------------------------------------------
    _proto_keys = {'loss_proto_pull', 'loss_proto_ortho', 'loss_proto_disambig', 'loss_proto_sparse'}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
                if k not in _proto_keys
            })
        if args.two_stage:
            aux_weight_dict.update({
                k + '_enc': v for k, v in weight_dict.items()
                if k not in _proto_keys
            })
        weight_dict.update(aux_weight_dict)
 
    losses = ['labels', 'boxes', 'cardinality']
    if args.segmentation_head:
        losses.append('masks')
    if use_proto:
        losses.append('prototype_align')
 
    criterion_kwargs = dict(
        focal_alpha=args.focal_alpha,
        losses=losses,
        group_detr=args.group_detr,
        sum_group_losses=getattr(args, 'sum_group_losses', False),
        use_varifocal_loss=args.use_varifocal_loss,
        use_position_supervised_loss=args.use_position_supervised_loss,
        ia_bce_loss=args.ia_bce_loss,
        prototype_memory=prototype_memory,
        prototype_loss_coef=getattr(args, 'prototype_loss_coef', 0.1),
    )
    if args.segmentation_head:
        criterion_kwargs['mask_point_sample_ratio'] = args.mask_point_sample_ratio
 
    criterion = SetCriterion(
        args.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        **criterion_kwargs,
    )
    criterion.to(device)
    postprocess = PostProcess(num_select=args.num_select)
    return criterion, postprocess
