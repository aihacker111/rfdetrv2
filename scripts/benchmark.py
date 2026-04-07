#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR — Model Benchmark Script
# Measures: Parameters (total / by component) + GFLOPs (backbone / full)
# Usage:
#   python scripts/benchmark.py                    # nano + small + base
#   python scripts/benchmark.py --model nano
#   python scripts/benchmark.py --compare-cpfe    # show CPFE param delta
#   python scripts/benchmark.py --no-latency      # skip latency (faster)
# ------------------------------------------------------------------------

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Direct imports — bypass main.py (avoids peft/transformers/torchvision chain)
from rfdetrv2.models.builder import build_model
from rfdetrv2.config import (
    RFDETRV2NanoConfig,
    RFDETRV2SmallConfig,
    RFDETRV2BaseConfig,
    pydantic_dump,
)

# ─────────────────────────────────────────────────────────────────────────────
# Build args Namespace directly (no main.py dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _make_args(cfg_cls, use_cpfe: bool, num_classes: int) -> argparse.Namespace:
    cfg = cfg_cls(num_classes=num_classes)
    kw  = pydantic_dump(cfg)

    return argparse.Namespace(
        # ── encoder ──────────────────────────────────────────────────────────
        encoder                  = kw["encoder"],
        vit_encoder_num_layers   = kw["out_feature_indexes"][-1] + 1,
        pretrained_encoder       = None,
        window_block_indexes     = None,
        drop_path                = 0.0,
        out_feature_indexes      = kw["out_feature_indexes"],
        use_cls_token            = False,
        freeze_encoder           = False,
        layer_norm               = False,
        rms_norm                 = False,
        backbone_lora            = False,
        force_no_pretrain        = True,        # skip weight download for benchmark
        gradient_checkpointing   = False,
        patch_size               = kw["patch_size"],
        num_windows              = kw["num_windows"],
        positional_encoding_size = kw["positional_encoding_size"],
        use_windowed_attn        = kw.get("use_windowed_attn", False),
        use_convnext_projector   = kw.get("use_convnext_projector", True),
        # ── CPFE ─────────────────────────────────────────────────────────────
        use_cpfe                 = use_cpfe,
        cpfe_use_sdg             = True,
        cpfe_use_dn              = True,
        cpfe_use_tpr             = True,
        # ── transformer ──────────────────────────────────────────────────────
        hidden_dim               = kw["hidden_dim"],
        dec_layers               = kw["dec_layers"],
        sa_nheads                = kw["sa_nheads"],
        ca_nheads                = kw["ca_nheads"],
        dec_n_points             = kw["dec_n_points"],
        dim_feedforward          = 2048,
        dropout                  = 0.0,
        projector_scale          = kw["projector_scale"],
        num_feature_levels       = len(kw["projector_scale"]),
        position_embedding       = "sine",
        two_stage                = kw.get("two_stage", True),
        bbox_reparam             = kw.get("bbox_reparam", True),
        lite_refpoint_refine     = kw.get("lite_refpoint_refine", True),
        decoder_norm             = "LN",
        # ── model ────────────────────────────────────────────────────────────
        num_classes              = num_classes + 1,   # +1 background
        num_queries              = kw.get("num_queries", 300),
        num_select               = kw.get("num_select", 300),
        group_detr               = 13,
        aux_loss                 = False,             # save compute for benchmark
        resolution               = kw["resolution"],
        shape                    = (kw["resolution"], kw["resolution"]),
        # ── misc ─────────────────────────────────────────────────────────────
        device                   = "cpu",
        encoder_only             = False,
        backbone_only            = False,
        segmentation_head        = False,
        mask_downsample_ratio    = 4,
        use_prototype_align      = False,   # no loss needed for benchmark
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> dict:
    def _sub(key: str) -> int:
        return sum(p.numel() for n, p in model.named_parameters() if key in n)

    return dict(
        total       = sum(p.numel() for p in model.parameters()),
        trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad),
        encoder     = _sub("backbone.0.encoder"),
        cpfe        = _sub("backbone.0.cpfe"),
        projector   = _sub("backbone.0.projector"),
        transformer = _sub("transformer"),
        heads       = sum(
            p.numel() for n, p in model.named_parameters()
            if any(k in n for k in (
                "class_embed", "bbox_embed", "enc_output",
                "query_feat",  "refpoint_embed",
            ))
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GFLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────

class _NestedTensorWrapper(nn.Module):
    """Lets thop / torchinfo call model(plain_tensor) via NestedTensor."""
    def __init__(self, model):
        super().__init__()
        self._m = model

    def forward(self, x: torch.Tensor):
        from rfdetrv2.util.misc import NestedTensor
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                           dtype=torch.bool, device=x.device)
        return self._m(NestedTensor(x, mask))


def _try_thop(model: nn.Module, dummy: torch.Tensor):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        return None


def _try_torchinfo(model: nn.Module, dummy: torch.Tensor):
    try:
        from torchinfo import summary
        s = summary(model, input_data=dummy, verbose=0, depth=0)
        return s.total_mult_adds * 2 / 1e9
    except Exception:
        return None


def estimate_gflops(model: nn.Module, resolution: int) -> float | None:
    model.eval()
    wrapped = _NestedTensorWrapper(model)
    dummy   = torch.zeros(1, 3, resolution, resolution)
    gflops  = _try_thop(wrapped, dummy)
    if gflops is None:
        gflops = _try_torchinfo(wrapped, dummy)
    return gflops


def estimate_backbone_gflops(model: nn.Module, resolution: int) -> float | None:
    """GFLOPs for backbone only (encoder + CPFE + projector)."""

    class _BackboneOnly(nn.Module):
        def __init__(self, joiner):
            super().__init__()
            self._j = joiner

        def forward(self, x: torch.Tensor):
            from rfdetrv2.util.misc import NestedTensor
            mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                               dtype=torch.bool, device=x.device)
            return self._j(NestedTensor(x, mask))

    bw    = _BackboneOnly(model.backbone)
    dummy = torch.zeros(1, 3, resolution, resolution)
    gf    = _try_thop(bw, dummy)
    if gf is None:
        gf = _try_torchinfo(bw, dummy)
    return gf


# ─────────────────────────────────────────────────────────────────────────────
# Latency
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency(model: nn.Module, resolution: int, n_runs: int = 20) -> float:
    from rfdetrv2.util.misc import NestedTensor
    dummy = torch.zeros(1, 3, resolution, resolution)
    mask  = torch.zeros(1, resolution, resolution, dtype=torch.bool)
    nt    = NestedTensor(dummy, mask)

    for _ in range(3):          # warmup
        model(nt)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(nt)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]   # median (ms)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

W     = 72
_SEP  = "─" * W
_SEP2 = "═" * W

def _M(n: int) -> str:
    if n >= 1_000_000: return f"{n/1e6:.2f} M"
    if n >= 1_000:     return f"{n/1e3:.1f} K"
    return str(n)

def _G(n) -> str:
    return f"{n:.2f} G" if n is not None else "N/A"

def _row(label: str, value: str, indent: int = 0) -> str:
    pad = "  " * indent
    return f"  {pad}{label:<36}{value:>22}"


def print_report(name, res, params, gf_bb, gf_full, lat, with_cpfe):
    tag = "[CPFE ✓]" if with_cpfe else "[CPFE ✗]"
    print()
    print(_SEP2)
    print(f"  RF-DETR v2 · {name.upper()} · {tag} · {res}×{res} px")
    print(_SEP2)

    print(f"\n  PARAMETERS")
    print(f"  {_SEP}")
    print(_row("DINOv3 encoder",         _M(params["encoder"]),      1))
    if with_cpfe and params["cpfe"] > 0:
        print(_row("CPFE (SDG + DN + TPR)",  _M(params["cpfe"]),         1))
    print(_row("ConvNeXt projector",     _M(params["projector"]),    1))
    print(_row("Transformer (enc+dec)",  _M(params["transformer"]),  1))
    print(_row("Detection heads",        _M(params["heads"]),        1))
    print(f"  {_SEP}")
    print(_row("Total",                  _M(params["total"]),        0))
    print(_row("  of which trainable",   _M(params["trainable"]),    0))

    print(f"\n  GFLOPs  (batch=1, single image)")
    print(f"  {_SEP}")
    print(_row("Backbone (enc+cpfe+proj)", _G(gf_bb),   1))
    print(_row("Full model",               _G(gf_full), 1))

    if lat is not None:
        print(f"\n  LATENCY  (CPU · batch=1 · median {20} runs)")
        print(f"  {_SEP}")
        print(_row("Forward pass", f"{lat:.1f} ms", 1))

    print()


def print_summary_table(rows: list):
    if len(rows) < 2:
        return
    print()
    print(_SEP2)
    print("  SUMMARY")
    print(_SEP2)
    hdr = (f"  {'Model':<10}{'CPFE':<6}{'Res':>5}"
           f"{'Params':>11}{'CPFE Δ':>9}"
           f"{'GFLOPs(bb)':>12}{'GFLOPs(full)':>14}{'Lat(ms)':>10}")
    print(hdr)
    print(f"  {_SEP}")

    for r in rows:
        base_p = next(
            (x["total"] for x in rows
             if x["name"] == r["name"] and not x["cpfe"]),
            None
        )
        if not r["cpfe"]:
            delta = "base"
        elif base_p:
            d = (r["total"] - base_p) / 1e6
            delta = f"+{d:.2f}M"
        else:
            delta = "N/A"

        cpfe_sym = "✓" if r["cpfe"] else "✗"
        gfb  = f"{r['gf_bb']:.1f}"   if r["gf_bb"]   else "N/A"
        gff  = f"{r['gf_full']:.1f}" if r["gf_full"]  else "N/A"
        lat  = f"{r['lat']:.0f}"     if r["lat"]       else "N/A"

        print(
            f"  {r['name']:<10}{cpfe_sym:<6}{r['res']:>5}"
            f"{_M(r['total']):>11}{delta:>9}"
            f"{gfb:>12}{gff:>14}{lat:>10}"
        )

    print(_SEP2)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

_CFG = {
    "nano":  RFDETRV2NanoConfig,
    "small": RFDETRV2SmallConfig,
    "base":  RFDETRV2BaseConfig,
}


def run(models: list, compare_cpfe: bool, num_classes: int, with_lat: bool):
    print()
    print(_SEP2)
    print("  RF-DETR v2 — Benchmark  (params · GFLOPs · latency)")
    print(f"  Models   : {', '.join(m.upper() for m in models)}")
    print(f"  Classes  : {num_classes}")
    print(f"  CPFE δ   : {'yes — show w/ and w/o' if compare_cpfe else 'CPFE ON only'}")
    print(_SEP2)

    summary_rows = []
    variants     = [True, False] if compare_cpfe else [True]

    for name in models:
        cfg_cls = _CFG[name]

        for cpfe in variants:
            label = f"{name.upper()} {'+ CPFE' if cpfe else '- CPFE'}"
            print(f"\n  ▸ Building {label} ...", end=" ", flush=True)

            try:
                args  = _make_args(cfg_cls, use_cpfe=cpfe, num_classes=num_classes)
                model = build_model(args)
                model.eval()
            except Exception as exc:
                print(f"FAILED\n  [ERROR] {exc}")
                import traceback; traceback.print_exc()
                continue

            print("OK")
            res    = args.resolution
            params = count_params(model)

            print(f"  ▸ Counting GFLOPs ...", end=" ", flush=True)
            gf_bb   = estimate_backbone_gflops(model, res)
            gf_full = estimate_gflops(model, res)
            print("OK" if (gf_bb or gf_full) else "N/A")

            lat = None
            if with_lat:
                print(f"  ▸ Measuring latency ...", end=" ", flush=True)
                try:
                    lat = measure_latency(model, res)
                    print(f"{lat:.0f} ms")
                except Exception:
                    print("N/A")

            print_report(name, res, params, gf_bb, gf_full, lat, cpfe)

            summary_rows.append(dict(
                name=name, cpfe=cpfe, res=res,
                total=params["total"],
                gf_bb=gf_bb, gf_full=gf_full, lat=lat,
            ))

    print_summary_table(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RF-DETR v2 — params / GFLOPs / latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", "-m", nargs="+",
                   choices=["nano", "small", "base", "all"], default=["all"],
                   help="Model(s) to benchmark ('all' = nano+small+base)")
    p.add_argument("--compare-cpfe", action="store_true",
                   help="Run each model w/ and w/o CPFE to show param delta")
    p.add_argument("--num-classes", type=int, default=80,
                   help="Number of detection classes (excluding background)")
    p.add_argument("--no-latency", action="store_true",
                   help="Skip CPU latency measurement (faster run)")
    args = p.parse_args()

    model_list = (["nano", "small", "base"]
                  if "all" in args.model else args.model)

    run(
        models       = model_list,
        compare_cpfe = args.compare_cpfe,
        num_classes  = args.num_classes,
        with_lat     = not args.no_latency,
    )
