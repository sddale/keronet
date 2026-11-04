"""
Save and load YAML model configs and rebuild models from them via a name registry.

Config keys: config_version, arch, pde (name, kwargs), hparams (constructor kwargs),
training and metadata. Import registers pi_deeponet, pit, pinto, keronet, mkeronet.
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml

from pdes import get_pde, PDESpec


# Architecture registry


_ARCH_REGISTRY: Dict[str, Type] = {}


def register_architecture(name: str):
    """Decorator that registers a class under name; raises ValueError on a duplicate."""

    def _decorator(cls):
        if name in _ARCH_REGISTRY:
            raise ValueError(f"Architecture {name!r} already registered.")
        _ARCH_REGISTRY[name] = cls
        return cls

    return _decorator


def get_architecture(name: str) -> Type:
    """Return the class registered under name; raises KeyError if absent."""
    if name not in _ARCH_REGISTRY:
        available = sorted(_ARCH_REGISTRY.keys())
        raise KeyError(f"Architecture {name!r} not registered. Available: {available}")
    return _ARCH_REGISTRY[name]


def list_architectures():
    return sorted(_ARCH_REGISTRY.keys())


# Config save / load


CONFIG_VERSION = 1  # bump when schema changes incompatibly


def _coerce_yaml_safe(obj):
    """Recursively convert numpy/JAX values to Python natives and tuples to lists."""
    # Avoid hard imports of jax/numpy at module load; just duck-type.
    if hasattr(obj, "tolist") and hasattr(obj, "shape"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item") and not isinstance(obj, (bool, int, float, str)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _coerce_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_yaml_safe(v) for v in obj]
    return obj


def save_config(
    path: str | os.PathLike,
    arch: str,
    pde_name: str,
    hparams: Dict[str, Any],
    training: Dict[str, Any],
    pde_kwargs: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Validate arch and PDE, then write a YAML config to path (adds generated_at)."""
    if pde_kwargs is None:
        pde_kwargs = {}
    if metadata is None:
        metadata = {}

    # Sanity-check registry membership before writing --- fail fast on typos.
    if arch not in _ARCH_REGISTRY:
        raise KeyError(
            f"Architecture {arch!r} is not registered. Known: {list_architectures()}"
        )
    # Confirm PDE factory exists (this also runs the factory, which is the
    # only way to validate pde_kwargs against the factory signature).
    _ = get_pde(pde_name, **pde_kwargs)

    # Auto-add generated_at if absent
    metadata = dict(metadata)
    metadata.setdefault(
        "generated_at",
        datetime.datetime.now().isoformat(timespec="seconds"),
    )

    cfg = {
        "config_version": CONFIG_VERSION,
        "arch": arch,
        "pde": {
            "name": pde_name,
            "kwargs": _coerce_yaml_safe(pde_kwargs),
        },
        "hparams": _coerce_yaml_safe(hparams),
        "training": _coerce_yaml_safe(training),
        "metadata": _coerce_yaml_safe(metadata),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config, checking its required keys and version; returns a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping.")

    required = ["arch", "pde", "hparams", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config at {path} is missing required key(s): {missing}")

    cv = cfg.get("config_version", None)
    if cv is None:
        # Older configs without versioning are tolerated.
        cfg["config_version"] = 0
    elif cv > CONFIG_VERSION:
        raise ValueError(
            f"Config at {path} has version {cv}, but this code only "
            f"understands up to v{CONFIG_VERSION}. Upgrade the loader."
        )

    return cfg


# Build a model from a config


def build_model_from_config(
    cfg: Dict[str, Any],
) -> Tuple[Any, PDESpec]:
    """Build an untrained model from a loaded config; returns (model, pde_spec).

    Initialization uses cfg["hparams"]["init_seed"] if present, else 0.
    """
    arch_name = cfg["arch"]
    pde_name = cfg["pde"]["name"]
    pde_kwargs = cfg["pde"].get("kwargs", {}) or {}
    hparams = cfg["hparams"]

    # Reconstruct the PDE spec
    pde_spec = get_pde(pde_name, **pde_kwargs)

    # Look up architecture class and construct
    cls = get_architecture(arch_name)
    model = cls(pde_spec=pde_spec, **hparams)

    return model, pde_spec


# Architecture registration  (import side effects)

#
# Importing architecture modules and decorating their classes here keeps
# registry bookkeeping centralized. Add new architectures below.

from architectures.pi_deeponet import PIDeepONet
from architectures.pit import PIT
from architectures.pinto import PINTO
from architectures.keronet import KerONet
from architectures.mkeronet import MKerONet

register_architecture("pi_deeponet")(PIDeepONet)
register_architecture("pit")(PIT)
register_architecture("pinto")(PINTO)
register_architecture("keronet")(KerONet)
register_architecture("mkeronet")(MKerONet)


__all__ = [
    "save_config",
    "load_config",
    "build_model_from_config",
    "register_architecture",
    "get_architecture",
    "list_architectures",
    "CONFIG_VERSION",
]
