"""
Bath spectral densities for the energy-transfer project.

Variant A — Ohmic/Drude (structureless):
    J_Ohmic(ω) = 2 λ γ ω / (ω² + γ²)

    λ : reorganisation energy  (cm⁻¹)
    γ : bath cut-off frequency (cm⁻¹)

    FMO-consistent defaults (Ishizaki & Fleming 2009):
        λ ≈ 35 cm⁻¹,  1/γ ≈ 50 fs  →  γ ≈ 53 cm⁻¹

Units throughout: angular frequency ω in cm⁻¹  (multiply by 2πc to get rad/s).
"""

from __future__ import annotations
import numpy as np


# ── Default Ohmic parameters ──────────────────────────────────────────────────

LAMBDA_CM   = 35.0    # reorganisation energy   (cm⁻¹)
GAMMA_CM    = 53.0    # Drude cut-off frequency  (cm⁻¹)   1/γ ≈ 50 fs
TEMPERATURE_K = 300.0


def J_Ohmic(omega: float | np.ndarray,
            lambda_: float = LAMBDA_CM,
            gamma: float   = GAMMA_CM) -> float | np.ndarray:
    """
    Ohmic (Drude-Lorentz) spectral density.

        J(ω) = 2 λ γ ω / (ω² + γ²)      [cm⁻¹]

    Returns zero for ω ≤ 0 (physical bath only supports ω > 0).
    """
    omega = np.asarray(omega, dtype=float)
    out   = np.where(omega > 0,
                     2.0 * lambda_ * gamma * omega / (omega ** 2 + gamma ** 2),
                     0.0)
    return float(out) if out.ndim == 0 else out


def gamma_Ohmic(omega: float | np.ndarray,
                lambda_: float = LAMBDA_CM,
                gamma: float   = GAMMA_CM,
                temperature: float = TEMPERATURE_K) -> float | np.ndarray:
    """
    One-sided bath correlation spectral function used by QuTiP brmesolve:

        γ(ω) = J(|ω|) / (1 − exp(−β|ω|))   for ω > 0  [emission + absorption]
               J(|ω|) / (exp(β|ω|) − 1)    for ω < 0  [absorption only]

    This avoids the 0×∞ singularity that arises when computing J(0)×n(0)
    directly.  At ω→0 the ratio J(ω)/(1−exp(−βω)) → 2λ/(βγ) (finite).
    """
    hbar_over_kB = 1.4388          # K·cm  (= hc / k_B)
    omega  = np.asarray(omega, dtype=float)
    abs_w  = np.abs(omega)
    beta_w = hbar_over_kB * abs_w / temperature

    J_abs  = J_Ohmic(abs_w, lambda_, gamma)

    # Use L'Hôpital-safe ratio: J/(1−exp(−βω)).
    # For βω ≪ 1 use the Taylor expansion: 1−exp(−βω) ≈ βω → ratio ≈ J/ω × 1/β
    # J_Ohmic(ω)/ω → 2λγ/(ω²+γ²) → 2λ/γ at ω=0
    # Classical (Taylor) limit at ω→0:
    #   J(ω)/(1−exp(−βω)) → J(ω)/(βω) → 2λγ/(γ²·β) = 2λT/(hbar_kB·γ)
    limit_zero = 2.0 * lambda_ * temperature / (hbar_over_kB * gamma)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom_pos = np.where(beta_w > 1e-4, 1.0 - np.exp(-beta_w), beta_w)
        denom_neg = np.where(beta_w > 1e-4, np.expm1(beta_w),       beta_w)
        pos_raw = np.where(J_abs > 0, J_abs / denom_pos, limit_zero)
        neg_raw = np.where(J_abs > 0, J_abs / denom_neg, limit_zero)

    result = np.where(omega >= 0.0, pos_raw, neg_raw)
    return float(result) if result.ndim == 0 else result


# ── Structured (Adolphs–Renger-style) spectral density ────────────────────────
# The realistic FMO bath is not a single Drude peak: fluorescence line-narrowing
# (Wendling 2000; Adolphs & Renger, Biophys. J. 91, 2778, 2006) resolves a
# low-frequency quasi-continuum PLUS discrete intramolecular modes, the most
# important being a ~180 cm⁻¹ vibration that is near-resonant with the FMO
# exciton energy gaps and drives vibronic coherence (PNAS 115, 2018).  We model
# this as a Drude background + one underdamped Brownian-oscillator mode at
# 180 cm⁻¹, which is the standard structured-bath form used in FMO HEOM studies.

OMEGA_RES_CM = 180.0     # resonant intramolecular mode (Adolphs–Renger / FLN)
S_RES        = 0.22      # its Huang–Rhys factor  →  λ_mode = S·ω = 39.6 cm⁻¹
GAMMA_RES_CM = 30.0      # mode damping (cm⁻¹), ~180 fs vibrational lifetime
LAMBDA_RES_CM = S_RES * OMEGA_RES_CM


def J_underdamped(omega, lambda_mode=LAMBDA_RES_CM, w0=OMEGA_RES_CM,
                  gamma_mode=GAMMA_RES_CM):
    """
    Underdamped Brownian-oscillator spectral density for one vibrational mode:

        J(ω) = 2 λ_m ω0² γ_m ω / ((ω² − ω0²)² + γ_m² ω²)      [cm⁻¹]

    Same convention as J_Ohmic: ∫₀^∞ J(ω)/(πω) dω = λ_mode (reorganisation
    energy of the mode); peaks near ω0.
    """
    omega = np.asarray(omega, dtype=float)
    out = np.where(omega > 0,
                   2.0 * lambda_mode * w0 ** 2 * gamma_mode * omega
                   / ((omega ** 2 - w0 ** 2) ** 2 + gamma_mode ** 2 * omega ** 2),
                   0.0)
    return float(out) if out.ndim == 0 else out


def J_structured(omega, lambda_drude=LAMBDA_CM, gamma_drude=GAMMA_CM,
                 lambda_mode=LAMBDA_RES_CM, w0=OMEGA_RES_CM, gamma_mode=GAMMA_RES_CM):
    """Drude background + the 180 cm⁻¹ resonant mode (total λ ≈ 75 cm⁻¹, within
    the 35–120 cm⁻¹ range of reported FMO reorganisation energies)."""
    return (J_Ohmic(omega, lambda_drude, gamma_drude)
            + J_underdamped(omega, lambda_mode, w0, gamma_mode))


def gamma_structured(omega, lambda_drude=LAMBDA_CM, gamma_drude=GAMMA_CM,
                     lambda_mode=LAMBDA_RES_CM, w0=OMEGA_RES_CM,
                     gamma_mode=GAMMA_RES_CM, temperature=TEMPERATURE_K):
    """Bose-weighted rate function for the structured bath (drop-in for
    gamma_Ohmic in the secular-Redfield rate matrix / brmesolve)."""
    hbar_over_kB = 1.4388
    omega = np.asarray(omega, dtype=float)
    abs_w = np.abs(omega)
    beta_w = hbar_over_kB * abs_w / temperature
    J_abs = J_structured(abs_w, lambda_drude, gamma_drude, lambda_mode, w0, gamma_mode)
    # zero-frequency limit is set by the Drude part (the mode vanishes as ω→0)
    limit_zero = 2.0 * lambda_drude * temperature / (hbar_over_kB * gamma_drude)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom_pos = np.where(beta_w > 1e-4, 1.0 - np.exp(-beta_w), beta_w)
        denom_neg = np.where(beta_w > 1e-4, np.expm1(beta_w), beta_w)
        pos_raw = np.where(J_abs > 0, J_abs / denom_pos, limit_zero)
        neg_raw = np.where(J_abs > 0, J_abs / denom_neg, limit_zero)
    result = np.where(omega >= 0.0, pos_raw, neg_raw)
    return float(result) if result.ndim == 0 else result


def reorganization_energy(J_fn, w_max=4000.0, n=20000) -> float:
    """Numerical λ = (1/π) ∫₀^∞ J(ω)/ω dω for any spectral-density function."""
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    w = np.linspace(0, w_max, n)[1:]
    return float(_trapz(J_fn(w) / w, w) / np.pi)


if __name__ == "__main__":
    # λ = (1/π) ∫₀^∞ J(ω)/ω dω  — recovers the input reorganisation energy.
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy 2.0 renamed trapz
    omegas = np.linspace(0, 2000, 4000)[1:]               # skip ω=0 (J/ω limit)
    J_vals = J_Ohmic(omegas)
    lam_num = _trapz(J_vals / omegas, omegas) / np.pi
    print(f"Peak J at ω = γ = {GAMMA_CM} cm⁻¹:  J = {J_Ohmic(GAMMA_CM):.2f} cm⁻¹")
    print(f"Reorganisation energy check (numerical):  λ ≈ {lam_num:.1f} cm⁻¹ "
          f"(input {LAMBDA_CM:.0f})")
    print(f"Structured SD (Drude + {OMEGA_RES_CM:.0f} cm⁻¹ mode):  "
          f"λ_total ≈ {reorganization_energy(J_structured):.1f} cm⁻¹ "
          f"(Drude {reorganization_energy(J_Ohmic):.0f} + mode "
          f"{reorganization_energy(J_underdamped):.0f}); peak at "
          f"{OMEGA_RES_CM:.0f} cm⁻¹ J={J_structured(OMEGA_RES_CM):.1f}")
