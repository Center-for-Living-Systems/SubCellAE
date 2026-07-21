"""
intensity_transform.py
======================
Shifted-log intensity mapping for CIO-RB normalised patch data.

    log_map_forward  :  CIO-RB space  →  log-compressed space  (x_min→0, x_ref→1)
    log_map_inverse  :  log-compressed  →  CIO-RB space

Default parameters chosen from data statistics across all 4 datasets
(vinc, pfak, ppax, nih3t3):
    x_min  = -0.03   safely below the global pixel min of -0.0296
    x_ref  = 10.0    upper reference; max observed ≈ 10.7 in nih3t3
    delta  = 0.5     controls log compression strength

Key landmarks with defaults:
    x = -0.03  →  y = 0.000   (lower bound)
    x =  0.00  →  y ≈ 0.019   (background / outside-cell mean level)
    x =  1.00  →  y ≈ 0.367   (sigmoid ceiling from CIO-RB)
    x = 10.00  →  y = 1.000   (reference / approx upper bound)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# NumPy versions  (use for preprocessing / postprocessing on CPU arrays)
# ---------------------------------------------------------------------------

def log_map_forward(
    x: np.ndarray,
    x_min: float = -0.03,
    x_ref: float = 10.0,
    delta: float = 0.5,
    clip_lower: bool = True,
) -> np.ndarray:
    """
    Invertible shifted-log mapping: x_min→0, x_ref→1.

    Values above x_ref are allowed and map to values above 1.

    Parameters
    ----------
    x          : Input in CIO-RB intensity domain.
    x_min      : Lower reference bound (maps to 0). Default -0.03.
    x_ref      : Upper reference bound (maps to 1). Default 10.0.
    delta      : Log compression strength; larger → more linear. Default 0.5.
    clip_lower : If True, clip values below x_min to x_min before mapping.
    """
    if x_ref <= x_min:
        raise ValueError("x_ref must be greater than x_min.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    x = np.asarray(x, dtype=np.float64)
    if clip_lower:
        x = np.maximum(x, x_min)
    elif np.any(x < x_min):
        raise ValueError("Input contains values below x_min.")

    normalization = np.log1p((x_ref - x_min) / delta)
    return (np.log1p((x - x_min) / delta) / normalization).astype(np.float32)


def log_map_inverse(
    y: np.ndarray,
    x_min: float = -0.03,
    x_ref: float = 10.0,
    delta: float = 0.5,
    clip_lower: bool = False,
) -> np.ndarray:
    """
    Inverse of log_map_forward: y=0→x_min, y=1→x_ref.

    Parameters
    ----------
    y          : Input in the log-compressed domain.
    x_min, x_ref, delta : Must match the forward call.
    clip_lower : If True, clip y below 0 before inverting.
    """
    if x_ref <= x_min:
        raise ValueError("x_ref must be greater than x_min.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    y = np.asarray(y, dtype=np.float64)
    if clip_lower:
        y = np.maximum(y, 0.0)

    normalization = np.log1p((x_ref - x_min) / delta)
    return (x_min + delta * np.expm1(y * normalization)).astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch versions  (use inside training loops, on GPU tensors)
# ---------------------------------------------------------------------------

def log_map_forward_torch(x, x_min=-0.03, x_ref=10.0, delta=0.5, clip_lower=True):
    """
    Torch tensor version of log_map_forward.

    Operates in-place on the value; gradient does not flow through the
    transform (it is treated as a fixed pre-processing step).
    """
    import torch
    x = x.float()
    if clip_lower:
        x = torch.clamp(x, min=x_min)
    norm = float(np.log1p((x_ref - x_min) / delta))
    return torch.log1p((x - x_min) / delta) / norm


def log_map_inverse_torch(y, x_min=-0.03, x_ref=10.0, delta=0.5, clip_lower=False):
    """
    Torch tensor version of log_map_inverse.
    """
    import torch
    y = y.float()
    if clip_lower:
        y = torch.clamp(y, min=0.0)
    norm = float(np.log1p((x_ref - x_min) / delta))
    return x_min + delta * torch.expm1(y * norm)


# ---------------------------------------------------------------------------
# Convenience: build a (forward, inverse) pair bound to fixed params
# ---------------------------------------------------------------------------

def make_log_map(x_min=-0.03, x_ref=10.0, delta=0.5):
    """Return (forward_fn, inverse_fn) torch callables with fixed params."""
    def fwd(x, clip_lower=True):
        return log_map_forward_torch(x, x_min=x_min, x_ref=x_ref,
                                     delta=delta, clip_lower=clip_lower)
    def inv(y, clip_lower=False):
        return log_map_inverse_torch(y, x_min=x_min, x_ref=x_ref,
                                     delta=delta, clip_lower=clip_lower)
    return fwd, inv
