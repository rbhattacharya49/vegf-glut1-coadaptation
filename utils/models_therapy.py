"""Therapy-enabled evolving-neighborhood VEGF-GLUT1 model.

This module extends the evolving-neighborhood model by allowing time-dependent
therapy to modify the effective VEGF and GLUT1 terms after a treatment onset
``t_drug``.

It also includes an optional positivity-preserving integration scheme, which is
recommended for long simulations because it avoids negative values of ``N``,
``u``, or ``y`` caused by solver overshoot.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .model_params import EvolvingParams


__all__ = [
    "G_time",
    "grads_ess_time",
    "grads_teamopt_time",
    "integrate_ess",
    "integrate_teamopt",
    "integrate_ess_pos",
    "integrate_teamopt_pos",
]

RATE_CLIP = 1e6


def exp_safe(x: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation with clipping to avoid overflow."""
    return np.exp(np.clip(x, -700.0, 700.0))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable softplus transform."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_prime(x: np.ndarray | float) -> np.ndarray | float:
    """Derivative of the softplus transform."""
    return sigmoid(x)


def inv_softplus(u: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """Inverse softplus used to initialize positivity-preserving coordinates."""
    return np.log(np.expm1(u) + eps)


def _heaviside_on(t: float, t_drug: Optional[float]) -> float:
    """Treatment switch: 0 before treatment, 1 after treatment starts."""
    if t_drug is None:
        return 0.0
    return 1.0 if t >= t_drug else 0.0


def _denominators_at_time(t: float, p: EvolvingParams, t_drug: Optional[float]) -> Tuple[float, float]:
    """Return therapy-adjusted denominators for GLUT and VEGF terms."""
    H = _heaviside_on(t, t_drug)
    denom_G = max(1.0 + H * (p.e * p.m_G), 1e-12)
    denom_A = max(1.0 + H * (p.d * p.m_A), 1e-12)
    return denom_G, denom_A


def _safe_frac_one_minus_exp_over_u(
    a: float,
    N: float,
    u: float,
    denom_A: float,
    eps: float,
) -> float:
    """Safely evaluate therapy-adjusted ``(1 - exp(-a*N*u/denom_A))/u``."""
    coeff = (a * N) / max(denom_A, 1e-12)
    if abs(u) < eps:
        return coeff
    val = (1.0 - exp_safe(-(a * N * u) / denom_A)) / u
    return val if np.isfinite(val) else coeff


def G_time(N: float, u: float, y: float, p: EvolvingParams, t: float, t_drug: Optional[float]) -> float:
    """Therapy-adjusted payoff function."""
    N_eff = max(N, p.N_floor)
    denom_G, denom_A = _denominators_at_time(t, p, t_drug)
    one_m_exp_k = 1.0 - exp_safe(-p.k * y / denom_G)
    one_m_exp_aNu = 1.0 - exp_safe(-p.a * N_eff * u / denom_A)
    return p.b * ((p.R / N_eff) * one_m_exp_k * one_m_exp_aNu - p.c * u - p.h * y - p.f)


def grads_ess_time(
    u: float,
    y: float,
    q: float,
    N: float,
    p: EvolvingParams,
    t: float,
    t_drug: Optional[float],
) -> Tuple[float, float]:
    """Therapy-adjusted ESS gradients."""
    N_eff = max(N, p.N_floor)
    denom_G, denom_A = _denominators_at_time(t, p, t_drug)

    exp_k = exp_safe(-p.k * y / denom_G)
    exp_aNu = exp_safe(-p.a * N_eff * u / denom_A)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    term1 = (p.R * p.a / N_eff) * one_m_exp_k * exp_aNu
    assort = (1.0 - q) * (N_eff - 1.0) / (N_eff * N_eff)
    term2 = p.R * one_m_exp_k * assort * _safe_frac_one_minus_exp_over_u(
        p.a, N_eff, u, denom_A, p.u_eps
    )

    dGdu = p.b * (term1 + term2 - p.c)
    dGdy = p.b * ((p.R * p.k / (N_eff * denom_G)) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdu, p.sigma_y * dGdy


def grads_teamopt_time(
    u: float,
    y: float,
    N: float,
    p: EvolvingParams,
    t: float,
    t_drug: Optional[float],
) -> Tuple[float, float]:
    """Therapy-adjusted Team-Optimum gradients."""
    N_eff = max(N, p.N_floor)
    denom_G, denom_A = _denominators_at_time(t, p, t_drug)

    exp_k = exp_safe(-p.k * y / denom_G)
    exp_aNu = exp_safe(-p.a * N_eff * u / denom_A)
    one_m_exp_k = 1.0 - exp_k
    one_m_exp_aNu = 1.0 - exp_aNu

    dGdu = p.b * ((p.R * p.a / denom_A) * one_m_exp_k * exp_aNu - p.c)
    dGdy = p.b * ((p.R * p.k / (N_eff * denom_G)) * one_m_exp_aNu * exp_k - p.h)

    return p.sigma_u * dGdu, p.sigma_y * dGdy


def rhs_ess(t: float, Y: np.ndarray, q: float, p: EvolvingParams, t_drug: Optional[float]) -> np.ndarray:
    """Right-hand side for therapy-adjusted ESS dynamics."""
    N, u, y = Y
    N_eff = max(N, p.N_floor)
    du_dt, dy_dt = grads_ess_time(u, y, q, N_eff, p, t, t_drug)
    dN_dt = N_eff * G_time(N_eff, u, y, p, t, t_drug)
    return np.array([dN_dt, du_dt, dy_dt], dtype=float)


def rhs_teamopt(t: float, Y: np.ndarray, p: EvolvingParams, t_drug: Optional[float]) -> np.ndarray:
    """Right-hand side for therapy-adjusted Team-Optimum dynamics."""
    N, u, y = Y
    N_eff = max(N, p.N_floor)
    du_dt, dy_dt = grads_teamopt_time(u, y, N_eff, p, t, t_drug)
    dN_dt = N_eff * G_time(N_eff, u, y, p, t, t_drug)
    return np.array([dN_dt, du_dt, dy_dt], dtype=float)


def integrate_ess(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
    t_drug: Optional[float] = None,
    positivity: bool = True,
    method_pos: str = "Radau",
    max_step_pos: Optional[float] = 500.0,
) -> Dict[str, Any]:
    """Integrate therapy-adjusted ESS dynamics."""
    if positivity:
        return integrate_ess_pos(
            q=q,
            p=p,
            N0=N0,
            u0=u0,
            y0=y0,
            t_drug=t_drug,
            method=method_pos,
            max_step=max_step_pos,
        )

    y0_vec = np.array([max(N0, p.N_floor), u0, y0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, Y: rhs_ess(t, Y, q, p, t_drug),
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
        "t_drug": t_drug,
        "params": p,
    }


def integrate_teamopt(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
    t_drug: Optional[float] = None,
    positivity: bool = True,
    method_pos: str = "Radau",
    max_step_pos: Optional[float] = 500.0,
) -> Dict[str, Any]:
    """Integrate therapy-adjusted Team-Optimum dynamics."""
    if positivity:
        return integrate_teamopt_pos(
            q=q,
            p=p,
            N0=N0,
            u0=u0,
            y0=y0,
            t_drug=t_drug,
            method=method_pos,
            max_step=max_step_pos,
        )

    y0_vec = np.array([max(N0, p.N_floor), u0, y0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, Y: rhs_teamopt(t, Y, p, t_drug),
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
        "t_drug": t_drug,
        "params": p,
    }


def _map_state_pos(Z: np.ndarray, p: EvolvingParams) -> Tuple[float, float, float]:
    """Map unconstrained coordinates ``Z`` to positive physical coordinates."""
    eta, wu, wy = Z
    eta_clamped = np.clip(eta, -700.0, 700.0)
    N = p.N_floor + exp_safe(eta_clamped)
    u = softplus(wu)
    y = softplus(wy)
    return float(N), float(u), float(y)


def rhs_ess_pos(t: float, Z: np.ndarray, q: float, p: EvolvingParams, t_drug: Optional[float]) -> np.ndarray:
    """Positivity-preserving ESS dynamics in transformed coordinates."""
    N, u, y = _map_state_pos(Z, p)
    du_dt, dy_dt = grads_ess_time(u, y, q, N, p, t, t_drug)
    dN_dt = N * G_time(N, u, y, p, t, t_drug)

    eta, wu, wy = Z
    denomN = max(N - p.N_floor, 1e-12)
    d_eta = dN_dt / denomN
    d_wu = du_dt / max(softplus_prime(wu), 1e-6)
    d_wy = dy_dt / max(softplus_prime(wy), 1e-6)

    d_eta = float(np.clip(np.nan_to_num(d_eta, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))
    d_wu = float(np.clip(np.nan_to_num(d_wu, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))
    d_wy = float(np.clip(np.nan_to_num(d_wy, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))

    return np.array([d_eta, d_wu, d_wy], dtype=float)


def rhs_teamopt_pos(t: float, Z: np.ndarray, p: EvolvingParams, t_drug: Optional[float]) -> np.ndarray:
    """Positivity-preserving Team-Optimum dynamics in transformed coordinates."""
    N, u, y = _map_state_pos(Z, p)
    du_dt, dy_dt = grads_teamopt_time(u, y, N, p, t, t_drug)
    dN_dt = N * G_time(N, u, y, p, t, t_drug)

    eta, wu, wy = Z
    denomN = max(N - p.N_floor, 1e-12)
    d_eta = dN_dt / denomN
    d_wu = du_dt / max(softplus_prime(wu), 1e-6)
    d_wy = dy_dt / max(softplus_prime(wy), 1e-6)

    d_eta = float(np.clip(np.nan_to_num(d_eta, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))
    d_wu = float(np.clip(np.nan_to_num(d_wu, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))
    d_wy = float(np.clip(np.nan_to_num(d_wy, nan=0.0, posinf=RATE_CLIP, neginf=-RATE_CLIP), -RATE_CLIP, RATE_CLIP))

    return np.array([d_eta, d_wu, d_wy], dtype=float)


def integrate_ess_pos(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
    t_drug: Optional[float] = None,
    method: str = "Radau",
    max_step: Optional[float] = 500.0,
) -> Dict[str, Any]:
    """Integrate ESS dynamics in transformed coordinates to preserve positivity."""
    eta0 = np.log(max(N0, p.N_floor) - p.N_floor + 1e-12)
    wu0 = inv_softplus(max(u0, 0.0))
    wy0 = inv_softplus(max(y0, 0.0))
    Z0 = np.array([eta0, wu0, wy0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, Z: rhs_ess_pos(t, Z, q, p, t_drug),
        t_span=(0.0, p.t_max),
        y0=Z0,
        method=method,
        atol=p.atol,
        rtol=p.rtol,
        max_step=max_step,
        dense_output=False,
    )

    eta, wu, wy = sol.y
    N = p.N_floor + np.exp(eta)
    u = softplus(wu)
    y = softplus(wy)

    return {
        "success": sol.success,
        "message": sol.message,
        "t": sol.t,
        "N": N,
        "u": u,
        "y": y,
        "q": q,
        "t_drug": t_drug,
        "params": p,
    }


def integrate_teamopt_pos(
    q: float,
    p: EvolvingParams,
    N0: float = 5.0,
    u0: float = 0.5,
    y0: float = 0.5,
    t_drug: Optional[float] = None,
    method: str = "Radau",
    max_step: Optional[float] = 500.0,
) -> Dict[str, Any]:
    """Integrate Team-Optimum dynamics in transformed coordinates to preserve positivity."""
    eta0 = np.log(max(N0, p.N_floor) - p.N_floor + 1e-12)
    wu0 = inv_softplus(max(u0, 0.0))
    wy0 = inv_softplus(max(y0, 0.0))
    Z0 = np.array([eta0, wu0, wy0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, Z: rhs_teamopt_pos(t, Z, p, t_drug),
        t_span=(0.0, p.t_max),
        y0=Z0,
        method=method,
        atol=p.atol,
        rtol=p.rtol,
        max_step=max_step,
        dense_output=False,
    )

    eta, wu, wy = sol.y
    N = p.N_floor + np.exp(eta)
    u = softplus(wu)
    y = softplus(wy)

    return {
        "success": sol.success,
        "message": sol.message,
        "t": sol.t,
        "N": N,
        "u": u,
        "y": y,
        "q": q,
        "t_drug": t_drug,
        "params": p,
    }
