# ------------------------------------------------------------------------
# Load flat / sectioned YAML into a mutable runtime config (SimpleNamespace).
# ------------------------------------------------------------------------

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

_SECTION_KEYS = frozenset(
    {"model", "train", "dataset", "system", "solver", "loss", "export"}
)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "train_default.yaml"


def flatten_run_yaml(raw: Any) -> dict[str, Any]:
    """If YAML uses sections {model, train, system, ...}, merge into one dict."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Run YAML root must be a mapping, got {type(raw)}")
    if _SECTION_KEYS.isdisjoint(raw.keys()):
        return dict(raw)
    out: dict[str, Any] = {}
    for key in _SECTION_KEYS:
        block = raw.get(key)
        if isinstance(block, dict):
            for k, v in block.items():
                out[k] = v
    for k, v in raw.items():
        if k not in _SECTION_KEYS:
            out[k] = v
    return out


def merge_run_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = merge_run_dict(out[k], v)
        else:
            out[k] = v
    return out


def _deep_merge_flat(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two flat dicts; nested dict values merge recursively."""
    return merge_run_dict(a, b)


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return flatten_run_yaml(raw)


def parse_cli_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` tokens; values are passed through yaml.safe_load for typing."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key, _, rest = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override: {item!r}")
        rest = rest.strip()
        try:
            out[key] = yaml.safe_load(rest)
        except yaml.YAMLError:
            out[key] = rest
    return out


def load_run_config(
    config_path: str | Path | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SimpleNamespace:
    """
    Build runtime config: defaults YAML + optional user YAML + ``extra`` + ``kwargs``.

    ``config_path`` may be ``None`` to use packaged ``configs/train_default.yaml`` only
    (then ``kwargs`` / ``extra`` override).
    """
    base = load_yaml_file(default_config_path())
    if config_path is not None:
        user = load_yaml_file(config_path)
        base = _deep_merge_flat(base, user)
    if extra:
        base = _deep_merge_flat(base, dict(extra))
    if kwargs:
        base = _deep_merge_flat(base, dict(kwargs))
    if base.pop("no_use_convnext_projector", None) is True:
        base["use_convnext_projector"] = False
    return dict_to_namespace(base)


def dict_to_namespace(d: Mapping[str, Any]) -> SimpleNamespace:
    """Recursive SimpleNamespace for nested dicts (rare in our flat config)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_namespace(v)
        else:
            out[k] = v
    return SimpleNamespace(**out)


def namespace_to_dict(ns: SimpleNamespace) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = namespace_to_dict(v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def merge_namespace(base: SimpleNamespace, **updates: Any) -> SimpleNamespace:
    d = namespace_to_dict(base)
    d.update(updates)
    return dict_to_namespace(d)
