"""
Extract effective transfer rate Reff from acceptor-population dynamics.

Definition (following Ai et al.):
    Fit P₄(t) to a double-exponential rise:
        P₄(t) = P∞ [1 − A exp(−k₁ t) − (1−A) exp(−k₂ t)]

    Then Reff = amplitude-weighted rate:
        Reff = A k₁ + (1−A) k₂          [cm⁻¹ or fs⁻¹, same units as k]

    If the fit degenerates to a single exponential (k₁ ≈ k₂), Reff = k.

Fallback: if fitting fails, Reff is estimated as the slope of P₄(t) at t=0
(initial transfer rate).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _double_exp_rise(t: np.ndarray, P_inf: float, A: float,
                     k1: float, k2: float) -> np.ndarray:
    """P∞ [1 − A e^{−k₁t} − (1−A) e^{−k₂t}]"""
    A  = np.clip(A, 0.0, 1.0)
    k1 = max(k1, 1e-12)
    k2 = max(k2, 1e-12)
    return P_inf * (1.0 - A * np.exp(-k1 * t) - (1.0 - A) * np.exp(-k2 * t))


def _single_exp_rise(t: np.ndarray, P_inf: float, k: float) -> np.ndarray:
    return P_inf * (1.0 - np.exp(-k * t))


def compute_Reff(times: np.ndarray, P4: np.ndarray) -> float:
    """
    Fit P₄(t) and return Reff (amplitude-weighted transfer rate).

    Parameters
    ----------
    times : 1-D array, time in fs
    P4    : 1-D array, acceptor (site 4) population

    Returns
    -------
    Reff  : effective transfer rate in fs⁻¹
            (multiply by 33356 to convert to cm⁻¹)
    """
    P_inf_guess = float(P4[-1]) if P4[-1] > 1e-4 else 0.5

    # Rough single-exponential initial guess for k
    half_val = 0.5 * P_inf_guess
    idx = np.searchsorted(P4, half_val)
    if idx == 0 or idx >= len(times):
        k_guess = 1.0 / max(times[-1], 1.0)
    else:
        k_guess = np.log(2.0) / max(times[idx], 1.0)

    # Try double-exponential first
    try:
        p0     = [P_inf_guess, 0.5, k_guess * 2, k_guess * 0.5]
        bounds = ([0, 0, 0, 0], [1.0, 1.0, np.inf, np.inf])
        popt, _ = curve_fit(_double_exp_rise, times, P4,
                            p0=p0, bounds=bounds,
                            maxfev=20000, ftol=1e-10)
        P_inf, A, k1, k2 = popt
        A = np.clip(A, 0.0, 1.0)
        Reff = A * k1 + (1.0 - A) * k2
        return float(Reff)
    except RuntimeError:
        pass

    # Fall back to single-exponential
    try:
        popt, _ = curve_fit(_single_exp_rise, times, P4,
                            p0=[P_inf_guess, k_guess],
                            bounds=([0, 0], [1.0, np.inf]),
                            maxfev=10000)
        return float(popt[1])
    except RuntimeError:
        # Last resort: initial slope
        dt = times[1] - times[0]
        return float((P4[1] - P4[0]) / dt) if dt > 0 else 0.0


if __name__ == "__main__":
    # Synthetic test: known double-exponential
    t = np.linspace(0, 5000, 500)
    k1_true, k2_true, A_true = 1e-3, 2e-4, 0.4
    P4_test = _double_exp_rise(t, 0.8, A_true, k1_true, k2_true)
    P4_test += np.random.default_rng(0).normal(0, 0.002, size=len(t))

    R = compute_Reff(t, P4_test)
    R_true = A_true * k1_true + (1.0 - A_true) * k2_true
    print(f"True Reff = {R_true:.4e}  |  Fitted Reff = {R:.4e}")
