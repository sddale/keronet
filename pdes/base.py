"""
PDE specification (PDESpec) and the PDE registry.

Models need only operator_net(params, u, *coords) -> scalar, coords (x, t) or (x, y, t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple, Optional, Protocol
import jax.numpy as jnp


# Type aliases -----------------------------------------------------------------
# An "operator closure" is a thin scalar (x, t) -> u(x, t) callable that has
# already closed over `params` and `u_sensor`. PDE modules use these to compute
# the residual via autodiff without knowing anything about the architecture.
OperatorClosure = Callable[[float, float], jnp.ndarray]  # (x, t) -> scalar

# A "residual function" takes an operator closure and a query point, and
# returns the PDE residual at that point.
ResidualFn = Callable[[OperatorClosure, float, float], jnp.ndarray]

# A "BCS loss function" takes a model and a batch (PDE-specific tuple structure)
# and returns the scalar BCS loss. The function closes over `model` so that it
# can call `model.operator_net(params, ...)` internally.
BCSLossFn = Callable[[Any, Any, Any], jnp.ndarray]  # (model, params, batch) -> scalar


### BCS structure declaration ###
@dataclass(frozen=True)
class BCSStructure:
    """Label and input-array count of a PDE's BCS batch (informational only)."""

    kind: str
    n_inputs: int
    description: str = ""


###  Main PDESpec dataclass ###
@dataclass
class PDESpec:
    """
    Everything training and evaluation need for one PDE.

    coord_range: (min, max) per query coordinate, (x, t) or (x, y, t).
    residual_fn(operator_closure, u, *coords) -> residual, fit by MSE to s_res.
    generate_training_data(key) -> (u, ..., y_res, s_res); all but the last two arrays
    feed default_dataset_classes["bcs"], and (u, y_res, s_res) feeds ["res"].
    """

    name: str
    coord_range: Tuple[Tuple[float, float], Tuple[float, float]]
    params: Dict[str, Any]
    bcs_structure: BCSStructure
    residual_fn: ResidualFn
    bcs_loss_fn: BCSLossFn
    generate_training_data: Callable
    generate_test_data: Callable

    default_training_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_test_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_dataset_classes: Dict[str, Any] = field(default_factory=dict)

    description: str = ""


###  Registry helpers ###
_REGISTRY: Dict[str, Callable[..., PDESpec]] = {}


def register_pde(name: str):
    """Decorator registering a PDESpec factory under a new, unique `name`."""

    def _decorator(factory_fn):
        if name in _REGISTRY:
            raise ValueError(f"PDE name {name!r} already registered.")
        _REGISTRY[name] = factory_fn
        return factory_fn

    return _decorator


def get_pde(name: str, **kwargs) -> PDESpec:
    """Build registered PDE `name` with kwargs, e.g. get_pde("burgers", visc=0.01)."""
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(f"PDE {name!r} not registered. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_pdes():
    return sorted(_REGISTRY.keys())
