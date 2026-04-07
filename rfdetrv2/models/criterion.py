# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
SetCriterion — Hungarian-matching multi-task loss for RF-DETR.

Responsibilities
----------------
* Run Hungarian matching between predictions and ground-truth.
* Compute classification, box (L1 + GIoU), mask, and prototype-alignment losses.
* Aggregate weighted losses over decoder layers (aux_outputs) and encoder output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rfdetrv2.models.loss_fns import (
    dice_loss_jit,
    position_supervised_loss,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
)
from rfdetrv2.models.prototype_memory import PrototypeMemory
from rfdetrv2.models.segmentation_head import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from rfdetrv2.util import box_ops
from rfdetrv2.util.misc import accuracy, get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """Compute the total RF-DETR loss.

    Args:
        num_classes:                 Number of object categories (+ 1 background).
        matcher:                     Hungarian matcher.
        weight_dict:                 Mapping loss-name → scalar weight.
        focal_alpha:                 Focal-loss alpha.
        losses:                      List of active loss names.
        group_detr:                  Number of duplicate query groups (training only).
        sum_group_losses:            If True divide by total boxes, not boxes × groups.
        use_varifocal_loss:          Replace focal loss with VariFocal loss.
        use_position_supervised_loss: Replace with position-supervised loss.
        ia_bce_loss:                 IoU-aware BCE loss variant.
        mask_point_sample_ratio:     Mask sampling ratio for point-based mask loss.
        prototype_memory:            Optional PrototypeMemory for alignment loss.
        prototype_loss_coef:         Scalar weight for alignment loss.
    """

    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: dict,
        focal_alpha: float,
        losses: list,
        group_detr: int = 1,
        sum_group_losses: bool = False,
        use_varifocal_loss: bool = False,
        use_position_supervised_loss: bool = False,
        ia_bce_loss: bool = False,
        mask_point_sample_ratio: int = 16,
        prototype_memory: PrototypeMemory = None,
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
        assert "pred_logits" in outputs
        src_logits      = outputs["pred_logits"]
        idx             = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            loss_ce = self._ia_bce(src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes)

        elif self.use_position_supervised_loss:
            loss_ce = self._position_supervised(src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes)

        elif self.use_varifocal_loss:
            loss_ce = self._varifocal(src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes)

        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
            )
            onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            onehot = onehot[:, :, :-1]
            loss_ce = (
                sigmoid_focal_loss(src_logits, onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
                * src_logits.shape[1]
            )

        losses = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _iou_targets(self, outputs, idx, targets, indices):
        src_boxes    = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )[0])

    def _ia_bce(self, src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes):
        alpha, gamma = self.focal_alpha, 2
        pos_ious     = self._iou_targets(outputs, idx, targets, indices).clone().detach()
        prob         = src_logits.sigmoid()
        pos_weights  = torch.zeros_like(src_logits)
        neg_weights  = prob ** gamma
        pos_ind      = list(idx) + [target_classes_o]
        t            = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
        t            = t.clamp(0.01).detach()
        pos_weights[pos_ind] = t.to(pos_weights.dtype)
        neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
        return (neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)).sum() / num_boxes

    def _position_supervised(self, src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes):
        pos_ious = self._iou_targets(outputs, idx, targets, indices).clone().detach()
        cls_iou  = torch.zeros(
            (src_logits.shape[0], src_logits.shape[1], self.num_classes),
            dtype=src_logits.dtype, device=src_logits.device,
        )
        pos_ind  = list(idx) + [target_classes_o]
        cls_iou[pos_ind] = pos_ious
        norm     = cls_iou / (cls_iou.view(cls_iou.shape[0], -1, 1).amax(1, True) + 1e-8)
        return (
            position_supervised_loss(src_logits, norm, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )

    def _varifocal(self, src_logits, outputs, idx, targets, indices, target_classes_o, num_boxes):
        pos_ious    = self._iou_targets(outputs, idx, targets, indices).clone().detach()
        cls_iou     = torch.zeros(
            (src_logits.shape[0], src_logits.shape[1], self.num_classes),
            dtype=src_logits.dtype, device=src_logits.device,
        )
        pos_ind     = list(idx) + [target_classes_o]
        cls_iou[pos_ind] = pos_ious
        return (
            sigmoid_varifocal_loss(src_logits, cls_iou, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )

    # ------------------------------------------------------------------
    # Box losses
    # ------------------------------------------------------------------

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=pred_logits.device)
        card_pred   = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {"cardinality_error": F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx          = self._get_src_permutation_idx(indices)
        src_boxes    = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return {
            "loss_bbox": F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes,
            "loss_giou": (
                1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                ))
            ).sum() / num_boxes,
        }

    # ------------------------------------------------------------------
    # Mask loss
    # ------------------------------------------------------------------

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        idx        = self._get_src_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]

        if isinstance(pred_masks, torch.Tensor):
            src_masks = pred_masks[idx]
        else:
            sf = pred_masks["spatial_features"]
            qf = pred_masks["query_features"]
            bs = pred_masks["bias"]
            if idx[0].numel() == 0:
                src_masks = torch.tensor([], device=sf.device)
            else:
                batched = []
                pbc = idx[0].unique(return_counts=True)[1]
                bi  = torch.cat((torch.zeros_like(pbc[:1]), pbc), 0).cumsum(0)
                for i in range(pbc.shape[0]):
                    bi_i = idx[0][bi[i]:bi[i + 1]]
                    bx_i = idx[1][bi[i]:bi[i + 1]]
                    q_i  = qf[(bi_i, bx_i)]
                    sf_i = sf[idx[0][bi[i + 1] - 1]]
                    batched.append(torch.einsum("chw,nc->nhw", sf_i, q_i) + bs)
                src_masks = torch.cat(batched)

        if src_masks.numel() == 0:
            return {"loss_mask_ce": src_masks.sum(), "loss_mask_dice": src_masks.sum()}

        target_masks = torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_masks    = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            src_masks.shape[-2],
            src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio,
        )
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, lambda logits: calculate_uncertainty(logits), num_points, 3, 0.75,
            )

        point_logits = point_sample(src_masks,    point_coords, align_corners=False).squeeze(1)
        with torch.no_grad():
            point_labels = point_sample(target_masks, point_coords, align_corners=False, mode="nearest").squeeze(1)

        losses = {
            "loss_mask_ce":   sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }
        del src_masks, target_masks
        return losses

    # ------------------------------------------------------------------
    # Prototype alignment loss
    # ------------------------------------------------------------------

    def loss_prototype_align(self, outputs, targets, indices, num_boxes):
        if "pred_queries" not in outputs or self.prototype_memory is None:
            return {}

        query_feats = outputs["pred_queries"]
        device      = query_feats.device

        all_feats, all_labels = [], []
        for b_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_idx = src_idx.to(device)
            tgt_idx = tgt_idx.to(device)
            lbls    = targets[b_idx]["labels"][tgt_idx]
            valid   = lbls < self.num_classes
            if not valid.any():
                continue
            all_feats.append(query_feats[b_idx, src_idx[valid]])
            all_labels.append(lbls[valid])

        if not all_feats:
            return {"loss_proto_align": query_feats.sum() * 0.0}

        feats  = torch.cat(all_feats,  dim=0)
        labels = torch.cat(all_labels, dim=0)
        self.prototype_memory.update(feats, labels)
        return {"loss_proto_align": self.prototype_memory.loss(feats, labels)}

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    _LOSS_MAP = {
        "labels":          "loss_labels",
        "cardinality":     "loss_cardinality",
        "boxes":           "loss_boxes",
        "masks":           "loss_masks",
        "prototype_align": "loss_prototype_align",
    }

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        assert loss in self._LOSS_MAP, f"Unknown loss '{loss}'"
        return getattr(self, self._LOSS_MAP[loss])(outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        group_detr          = self.group_detr if self.training else 1
        outputs_no_aux      = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices             = self.matcher(outputs_no_aux, targets, group_detr=group_detr)

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes *= group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets, group_detr=group_detr)
                for loss in self.losses:
                    if loss == "prototype_align":
                        continue
                    kw = {"log": False} if loss == "labels" else {}
                    losses.update({
                        k + f"_{i}": v
                        for k, v in self.get_loss(loss, aux, targets, aux_indices, num_boxes, **kw).items()
                    })

        if "enc_outputs" in outputs:
            enc  = outputs["enc_outputs"]
            enc_indices = self.matcher(enc, targets, group_detr=group_detr)
            for loss in self.losses:
                if loss == "prototype_align":
                    continue
                kw = {"log": False} if loss == "labels" else {}
                losses.update({
                    k + "_enc": v
                    for k, v in self.get_loss(loss, enc, targets, enc_indices, num_boxes, **kw).items()
                })

        return losses

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src   = torch.cat([src for (src, _) in indices])
        return batch, src

    def _get_tgt_permutation_idx(self, indices):
        batch = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt   = torch.cat([tgt for (_, tgt) in indices])
        return batch, tgt
