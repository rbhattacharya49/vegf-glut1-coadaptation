"""Evolving-neighborhood VEGF-GLUT1 model.

This module implements the model variant in which neighborhood size ``N`` is
itself dynamic and evolves according to

    dN/dt = N * G(N, u, y)

The state vector is ``[N, u, y]``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .model_params import EvolvingParams


__all__ = [
    "G",
    "grads_ess",
    "grads_teamopt",
    "rhs_ess",
    "rhs_teamopt",
    "integrate_ess",
    "integrate_teamopt",
]


def _safe_frac_one_minus_exp_over_u(a: float, N: float, u: float, eps: float) -> float:
    """Safely evaluate ``(1 - exp(-a*N*u)) / u`` near ``u = 0``."""
    coeff = a * N
    if abs(u) < eps:
        return coeff
    val = (1.0 - np.exp(-coeff * u)) / u
    return val if np.isfinite(val) else coeff


def G(N: float, u: float, y: float, p: EvolvingParams) -> float:
    """Payoff function for the evolving-neighborhood model."""
    N_eff = max(N, p.N_floor)
    one_m_exp_k = 1.0 - np.exp(-p.k * y)
    one_m_exp_aNu = 1.0 - np.exp(-p.a * N_eff * u)
    return p.b * ((p.R / N_eff) * one_m_exp_k * one_m_exp_aNu - p.c * u - p.h * y - p.f)


def grads_ess(u: float, y: float, q: float, N: float, p: EvolvingParams) -> Tuple[float, float]:
    """ESS selection gradients for the evolving-neighborhood model."""
    N_eff = max(N, p.N_floor)
    exp_k = np.exp(-p.k * y)
    exp_aNu = np.exp(-p.a * N_eff * u)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    term1 = (p.R * p.a / N_eff) * one_m_exp_k * exp_aNu
    assort = (1.0 - q) * (N_eff - 1.0) / (N_eff * N_eff)
    term2 = p.R * one_m_exp_k * assort * _safe_frac_one_minus_exp_over_u(p.a, N_eff, u, p.u_eps)

    dGdu = p.b * (term1 + term2 - p.c)
    dGdy = p.b * ((p.R * p.k / N_eff) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdu, p.sigma_y * dGdy


def grads_teamopt(u: float, y: float, N: float, p: EvolvingParams) -> Tuple[float, float]:
    """Team-optimum gradients for the evolving-neighborhood model."""
    N_eff = max(N, p.N_floor)
    exp_k = np.exp(-p.k * y)
    exp_aNu = np.exp(-p.a * N_eff * u)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    dGdu = p.b * (p.R * p.a * one_m_exp_k * exp_aNu - p.c)
    dGdy = p.b * ((p.R * p.k / N_eff) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdu, p.sigma_y * dGdy


def rhs_ess(t: float, Y: np.ndarray, q: float, p: EvolvingParams) -> np.ndarray:
    """Right-hand side for ESS dynamics with evolving neighborhood size."""
    del t
    N, u, y = Y
    N_eff = max(N, p.N_floor)
    du_dt, dy_dt = grads_ess(u, y, q, N_eff, p)
    dN_dt = N_eff * G(N_eff, u, y, p)
    return np.array([dN_dt, du_dt, dy_dt], dtype=float)


def rhs_teamopt(t: float, Y: np.ndarray, p: EvolvingParams) -> np.ndarray:
    """Right-hand side for Team-Optimum dynamics with evolving neighborhood size."""
    del t
    N, u, y = Y
    N_eff = max(N, p.N_floor)
    du_dt, dy_dt = grads_teamopt(u, y, N_eff, p)
    dN_dt = N_eff * G(N_eff, u, y, p)
    return np.array([dN_dt, du_dt, dy_dt], dtype=float)


def integrate_ess(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
) -> Dict[str, Any]:
    """Integrate ESS dynamics with evolving neighborhood size."""
    y0_vec = np.array([max(N0, p.N_floor), u0, y0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, Y: rhs_ess(t, Y, q, p),
        t_span=(0.0, p.t_max),
        y0=y0_vec,
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
        "N": sol.y[0],
        "u": sol.y[1],
        "y": sol.y[2],
        "q": q,
        "params": p,
    }


def integrate_teamopt(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
) -> Dict[str, Any]:
    """Integrate Team-Optimum dynamics with evolving neighborhood size."""
    del q
    y0_vec = np.array([max(N0, p.N_floor), u0, y0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, Y: rhs_teamopt(t, Y, p),
        t_span=(0.0, p.t_max),
        y0=y0_vec,
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
        "N": sol.y[0],
        "u": sol.y[1],
        "y": sol.y[2],
        "params": p,
    }
