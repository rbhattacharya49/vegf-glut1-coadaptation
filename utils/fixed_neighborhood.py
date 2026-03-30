"""Fixed-neighborhood VEGF-GLUT1 model.

This module implements the two-trait evolutionary dynamics when neighborhood
size ``N`` is treated as fixed.

Implemented components
----------------------
- payoff function ``G_fixedN``
- ESS selection gradients
- Team-optimum gradients
- ODE right-hand sides
- numerical integrators for ESS and Team Optimum

The state vector is ``[u, y]``, where
    u = VEGF investment
    y = GLUT1 investment
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .model_params import FixedNParams


__all__ = [
    "G_fixedN",
    "grads_fixedN",
    "rhs_fixedN",
    "integrate_fixedN",
    "grads_teamopt",
    "rhs_teamopt",
    "integrate_teamopt",
]


def safe_frac_one_minus_exp_over_u(a: float, N: float, u: float, eps: float = 1e-8) -> float:
    """Safely evaluate ``(1 - exp(-a*N*u)) / u``.

    The direct formula becomes numerically unstable near ``u = 0``. In that
    limit, the expression approaches ``a*N``, which we use as a series-limit
    fallback.
    """
    coeff = a * N
    if abs(u) < eps:
        return coeff
    val = (1.0 - np.exp(-coeff * u)) / u
    return val if np.isfinite(val) else coeff


def G_fixedN(N: float, u: float, y: float, q: float, p: FixedNParams) -> float:
    """Payoff function for the fixed-neighborhood model.

    Parameters
    ----------
    N
        Fixed neighborhood size.
    u
        VEGF investment.
    y
        GLUT1 investment.
    q
        Resource-sharing parameter. Present for interface consistency even
        though the payoff itself does not explicitly depend on ``q``.
    p
        Model parameters.
    """
    del q  # kept for API consistency with the gradient functions
    one_m_exp_k = 1.0 - np.exp(-p.k * y)
    one_m_exp_aNu = 1.0 - np.exp(-p.a * N * u)
    return p.b * ((p.R / N) * one_m_exp_k * one_m_exp_aNu - p.c * u - p.h * y - p.f)


def grads_fixedN(u: float, y: float, q: float, N: float, p: FixedNParams) -> Tuple[float, float]:
    """ESS selection gradients for the fixed-neighborhood model.

    Returns the trait dynamics ``(du/dt, dy/dt)`` after applying the
    evolvability scalings ``sigma_u`` and ``sigma_y``.
    """
    exp_k = np.exp(-p.k * y)
    exp_aNu = np.exp(-p.a * N * u)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    term1 = (p.R * p.a / N) * one_m_exp_k * exp_aNu
    assort = (1.0 - q) * (N - 1.0) / (N * N)
    term2 = p.R * one_m_exp_k * assort * safe_frac_one_minus_exp_over_u(p.a, N, u)

    dGdv = p.b * (term1 + term2 - p.c)
    dGdy = p.b * ((p.R * p.k / N) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdv, p.sigma_y * dGdy


def rhs_fixedN(t: float, Y: np.ndarray, q: float, N: float, p: FixedNParams) -> np.ndarray:
    """Right-hand side for ESS dynamics under fixed neighborhood size."""
    del t
    u, y = Y
    du, dy = grads_fixedN(u, y, q, N, p)
    return np.array([du, dy], dtype=float)


def integrate_fixedN(
    q: float,
    N: float,
    p: FixedNParams,
    u0: float = 0.5,
    y0: float = 0.5,
) -> Dict[str, Any]:
    """Integrate ESS dynamics for a fixed neighborhood size ``N``."""
    sol = solve_ivp(
        fun=lambda t, Y: rhs_fixedN(t, Y, q, N, p),
        t_span=(0.0, p.t_max),
        y0=np.array([u0, y0], dtype=float),
        method="LSODA",
        atol=p.atol,
        rtol=p.rtol,
        max_step=p.max_step,
        dense_output=False,
    )
    return {
        "success": sol.success,
        "message": sol.message,
        "t": sol.t,
        "u": sol.y[0],
        "y": sol.y[1],
        "q": q,
        "N": N,
        "params": p,
    }


def grads_teamopt(u: float, y: float, N: float, p: FixedNParams) -> Tuple[float, float]:
    """Team-optimum gradients for the fixed-neighborhood model."""
    exp_k = np.exp(-p.k * y)
    exp_aNu = np.exp(-p.a * N * u)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    dGdu = p.b * (p.R * p.a * one_m_exp_k * exp_aNu - p.c)
    dGdy = p.b * ((p.R * p.k / N) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdu, p.sigma_y * dGdy


def rhs_teamopt(t: float, Y: np.ndarray, N: float, p: FixedNParams) -> np.ndarray:
    """Right-hand side for Team-Optimum dynamics under fixed neighborhood size."""
    del t
    u, y = Y
    du, dy = grads_teamopt(u, y, N, p)
    return np.array([du, dy], dtype=float)


def integrate_teamopt(
    q: float,
    N: float,
    p: FixedNParams,
    u0: float = 0.5,
    y0: float = 0.5,
) -> Dict[str, Any]:
    """Integrate Team-Optimum dynamics for a fixed neighborhood size ``N``."""
    sol = solve_ivp(
        fun=lambda t, Y: rhs_teamopt(t, Y, N, p),
        t_span=(0.0, p.t_max),
        y0=np.array([u0, y0], dtype=float),
        method="LSODA",
        atol=p.atol,
        rtol=p.rtol,
        max_step=p.max_step,
        dense_output=False,
    )
    return {
        "success": sol.success,
        "message": sol.message,
        "t": sol.t,
        "u": sol.y[0],
        "y": sol.y[1],
        "q": q,
        "N": N,
        "params": p,
    }
