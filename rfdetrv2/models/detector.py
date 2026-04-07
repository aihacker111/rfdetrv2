# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
LWDETR  — RF-DETR main detection model.
PostProcess — post-processing: logits → scored boxes.
MLP     — lightweight multi-layer perceptron (bbox head).
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from rfdetrv2.util.misc import NestedTensor, nested_tensor_from_tensor_list
from rfdetrv2.util import box_ops


class MLP(nn.Module):
    """Simple feed-forward network used as the bounding-box regression head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LWDETR(nn.Module):
    """RF-DETR end-to-end object detector.

    Backbone → multi-scale projector → CDN transformer decoder → heads.

    Args:
        backbone:            Joiner(Backbone, PositionEncoding).
        transformer:         CDN transformer decoder.
        segmentation_head:   Optional segmentation head (or None).
        num_classes:         Number of foreground classes + 1 background.
        num_queries:         Number of object queries per group.
        aux_loss:            Enable auxiliary decoder-layer losses.
        group_detr:          Query duplication factor for group-DETR training.
        two_stage:           Enable two-stage encoder output.
        lite_refpoint_refine: Skip per-layer refpoint prediction (bbox head = None).
        bbox_reparam:        Use (Δcx, Δcy, log w, log h) reparameterisation.
    """

    def __init__(
        self,
        backbone,
        transformer,
        segmentation_head,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
    ):
        super().__init__()
        self.num_queries    = num_queries
        self.transformer    = transformer
        hidden_dim          = transformer.d_model

        self.class_embed    = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed     = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        self.refpoint_embed = nn.Embedding(num_queries * group_detr, 4)
        self.query_feat     = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone    = backbone
        self.aux_loss    = aux_loss
        self.group_detr  = group_detr
        self.bbox_reparam = bbox_reparam

        self.lite_refpoint_refine = lite_refpoint_refine
        self.transformer.decoder.bbox_embed = None if lite_refpoint_refine else self.bbox_embed

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed  = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed)  for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )

        self._export = False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes: int) -> None:
        base        = self.class_embed.weight.shape[0]
        num_repeats = int(math.ceil(num_classes / base))
        self.class_embed.weight.data = self.class_embed.weight.data.repeat(num_repeats, 1)[:num_classes]
        self.class_embed.bias.data   = self.class_embed.bias.data.repeat(num_repeats)[:num_classes]
        if self.two_stage:
            for emb in self.transformer.enc_out_class_embed:
                emb.weight.data = emb.weight.data.repeat(num_repeats, 1)[:num_classes]
                emb.bias.data   = emb.bias.data.repeat(num_repeats)[:num_classes]

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for _, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and getattr(m, "_export", True) is False:
                m.export()

    def update_drop_path(self, drop_path_rate: float, vit_encoder_num_layers: int) -> None:
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            enc = self.backbone[0].encoder
            blocks = getattr(enc, "blocks", None) or getattr(getattr(enc, "trunk", None), "blocks", None)
            if blocks is not None and hasattr(blocks[i].drop_path, "drop_prob"):
                blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate: float) -> None:
        for m in self.transformer.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop_rate

    # ------------------------------------------------------------------
    # Forward — training
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
            ref_w  = self.refpoint_embed.weight
            feat_w = self.query_feat.weight
        else:
            ref_w  = self.refpoint_embed.weight[:self.num_queries]
            feat_w = self.query_feat.weight[:self.num_queries]

        if self.segmentation_head is not None:
            seg_fwd = (self.segmentation_head.sparse_forward if self.training
                       else self.segmentation_head.forward)

        hs, ref_unsig, hs_enc, ref_enc = self.transformer(srcs, masks, poss, ref_w, feat_w)

        if hs is not None:
            if self.bbox_reparam:
                delta        = self.bbox_embed(hs)
                cxcy         = delta[..., :2] * ref_unsig[..., 2:] + ref_unsig[..., :2]
                wh           = delta[..., 2:].exp() * ref_unsig[..., 2:]
                outputs_coord = torch.concat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsig).sigmoid()

            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = seg_fwd(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

            if self.training:
                out["pred_queries"] = hs[-1]

            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class, outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )

        if self.two_stage:
            g          = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(g, dim=1)
            cls_enc    = torch.cat(
                [self.transformer.enc_out_class_embed[i](hs_enc_list[i]) for i in range(g)], dim=1
            )
            if self.segmentation_head is not None:
                masks_enc = seg_fwd(features[0].tensors, [hs_enc], samples.tensors.shape[-2:], skip_blocks=True)[0]

            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["enc_outputs"]["pred_masks"] = masks_enc
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["pred_masks"] = masks_enc

        return out

    # ------------------------------------------------------------------
    # Forward — ONNX export
    # ------------------------------------------------------------------

    def forward_export(self, tensors: torch.Tensor):
        srcs, _, poss = self.backbone(tensors)
        ref_w  = self.refpoint_embed.weight[:self.num_queries]
        feat_w = self.query_feat.weight[:self.num_queries]

        hs, ref_unsig, hs_enc, ref_enc = self.transformer(srcs, None, poss, ref_w, feat_w)

        outputs_masks = None
        if hs is not None:
            if self.bbox_reparam:
                delta        = self.bbox_embed(hs)
                cxcy         = delta[..., :2] * ref_unsig[..., 2:] + ref_unsig[..., :2]
                wh           = delta[..., 2:].exp() * ref_unsig[..., 2:]
                outputs_coord = torch.concat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsig).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs], tensors.shape[-2:])[0]
        else:
            assert self.two_stage
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs_enc], tensors.shape[-2:], skip_blocks=True)[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        if outputs_masks is not None:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])
            ]
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


# ---------------------------------------------------------------------------
# PostProcess
# ---------------------------------------------------------------------------

class PostProcess(nn.Module):
    """Convert model outputs (logits, boxes) into detection results."""

    def __init__(self, num_select: int = 300):
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits = outputs["pred_logits"]
        out_bbox   = outputs["pred_boxes"]
        out_masks  = outputs.get("pred_masks", None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1
        )
        scores     = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels     = topk_indexes % out_logits.shape[2]
        boxes      = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes      = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale        = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes        = boxes * scale[:, None, :]

        if out_masks is not None:
            results = []
            for i in range(out_masks.shape[0]):
                k_idx   = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i], 0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                )
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)),
                                        mode="bilinear", align_corners=False)
                results.append({"scores": scores[i], "labels": labels[i],
                                "boxes": boxes[i], "masks": masks_i > 0.0})
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]
        return results
