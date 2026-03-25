"""
Tests for DINOv3 pretrained weight download helpers.

Loads ``rfdetrv2/util/dinov3_pretrained.py`` directly so ``rfdetrv2`` (torch/supervision)
is not required for these unit tests.

- Default tests use mocks (no network).
- Set RUN_DINOV3_NETWORK_TEST=1 to run a short HuggingFace smoke check (reads first bytes only).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DINOV3_PRETRAINED_PATH = _ROOT / "rfdetrv2" / "util" / "dinov3_pretrained.py"


def _load_dinov3_pretrained():
    """Import the module by file path (avoids ``import rfdetrv2`` side effects)."""
    name = "rfdetrv2.util.dinov3_pretrained"
    spec = importlib.util.spec_from_file_location(name, _DINOV3_PRETRAINED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_DINOV3_PRETRAINED_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dinov3 = _load_dinov3_pretrained()


def test_url_for_relative_builds_hf_resolve_url() -> None:
    rel = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    url = _dinov3._url_for_relative(rel)
    assert url.startswith(_dinov3.HF_BASE_URL)
    assert rel in url
    assert url.endswith("?download=true")


def test_download_dinov3_pretrained_weights_writes_file(tmp_path: Path) -> None:
    mock_cm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "11"}
    mock_resp.read.side_effect = [b"fake-bytes", b""]
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = None

    with patch.object(_dinov3, "urlopen", return_value=mock_cm) as mock_urlopen:
        out = _dinov3.download_dinov3_pretrained_weights(
            dest_dir=tmp_path,
            manifest=[("dummy.pth", "dummy.pth")],
        )

    target = out / "dummy.pth"
    assert target.read_bytes() == b"fake-bytes"
    assert mock_urlopen.called


@pytest.mark.skipif(
    os.environ.get("RUN_DINOV3_NETWORK_TEST", "") != "1",
    reason="Set RUN_DINOV3_NETWORK_TEST=1 to verify HuggingFace URL (short read only).",
)
def test_huggingface_dinov3_weight_url_streams_bytes() -> None:
    """Reads only the first chunk from the smallest listed checkpoint URL (no full download)."""
    from urllib.request import Request, urlopen

    _fname, rel = _dinov3.DINOV3_PRETRAINED_MANIFEST[0]
    url = _dinov3._url_for_relative(rel)
    req = Request(url, headers={"User-Agent": "RF-DETR/dinov3_pretrained"})
    with urlopen(req, timeout=120) as resp:
        chunk = resp.read(4096)
    assert len(chunk) >= 16
    assert isinstance(chunk, (bytes, bytearray))


def test_resolve_pretrained_encoder_path_downloads_when_missing(
    tmp_path: Path,
) -> None:
    mock_cm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "4"}
    mock_resp.read.side_effect = [b"data", b""]
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = None

    weights = {"nano": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"}
    fname = weights["nano"]

    with patch.object(_dinov3, "urlopen", return_value=mock_cm):
        resolved = _dinov3.resolve_pretrained_encoder_path(
            tmp_path,
            "nano",
            explicit=None,
            weights_by_size=weights,
        )

    expected = tmp_path / "dinov3_pretrained" / fname
    assert Path(resolved) == expected
    assert expected.read_bytes() == b"data"
