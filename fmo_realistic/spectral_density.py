"""
Site-specific structured spectral densities for the 8-site FMO complex.

Total spectral density per site m:
    J_m(ω) = J_m^inter(ω) + J^intra(ω)

Intermolecular (Drude-Lorentz, site-specific reorganisation energy):
    J_m^inter(ω) = 2 λ_m γ_D ω / (ω² + γ_D²)

Intramolecular (10 underdamped Brownian oscillator modes, site-independent):
    J_j^vib(ω) = 4 S_j ω_j³ γ_j ω / [(ω² − ω_j²)² + ω² γ_j²]
    J^intra(ω) = Σ_j J_j^vib(ω)

Bath correlation function accepted by QuTiP brmesolve:
    γ_m(ω) = J_m(|ω|) / (1 − exp(−β|ω|))   ω ≥ 0
    γ_m(ω) = J_m(|ω|) / (exp(β|ω|) − 1)    ω < 0

Performance note
----------------
The 10 UBO modes with γ_j = 5 cm⁻¹ create narrow spectral peaks that make
brmesolve's internal frequency integrator extremely slow if gamma_site() is
called directly. Use build_gamma_tables() + make_gamma_interp() instead —
this pre-tabulates γ_m on a dense grid and returns a fast np.interp lookup.
That cuts Redfield tensor construction from hours to seconds.

Sources:
    Drude parameters: Lorenzoni et al. / Renger et al.
    UBO modes:        Rätsep & Freiberg, J. Lumin. 127, 251 (2007)
    UBO damping:      γ_j = 5 cm⁻¹ (highly underdamped, τ_vib ≈ 1-3 ps)
"""

from __future__ import annotations

import numpy as np

# Optional GPU array backend (CuPy). Falls back to NumPy silently.
try:
    import cupy as cp
    _xp = cp
    _GPU = True
except ImportError:
    _xp = np
    _GPU = False

# ── Drude-Lorentz (intermolecular) parameters ─────────────────────────────────

# Site-specific reorganisation energies λ_m (cm⁻¹), indexed 0..7 for BChl 1..8
LAMBDA_SITES_CM = np.array([21.0, 30.0, 14.0, 12.0, 10.0, 24.0, 14.0, 13.0])

GAMMA_DRUDE_CM = 50.0
TEMPERATURE_K  = 300.0

_HBAR_OVER_KB = 1.4388    # K·cm

# ── Intramolecular UBO modes (Rätsep & Freiberg 2007, top 10) ─────────────────
# Each entry: (ω_j [cm⁻¹], S_j [dimensionless])
MODES = [
    (  46.0, 0.011),
    (  68.0, 0.011),
    ( 117.0, 0.009),
    ( 180.0, 0.010),
    ( 191.0, 0.011),
    ( 202.0, 0.011),
    ( 243.0, 0.012),
    ( 291.0, 0.008),
    ( 366.0, 0.006),
    ( 770.0, 0.018),
]

GAMMA_VIB_CM = 5.0    # UBO damping — highly underdamped (τ_vib ≈ 1–3 ps)

# Pre-tabulation grid
_OMEGA_MIN  = 1e-3    # cm⁻¹
_OMEGA_MAX  = 2000.0  # cm⁻¹  (all modes < 770 cm⁻¹, Drude peaks at 50 cm⁻¹)
_N_TABLE    = 40_000  # 0.05 cm⁻¹ resolution — resolves 5 cm⁻¹ UBO linewidths


# ── Raw spectral density functions ────────────────────────────────────────────

def J_inter(omega, lambda_m, gamma_D=GAMMA_DRUDE_CM):
    omega = np.asarray(omega, dtype=float)
    return np.where(omega > 0,
                    2.0 * lambda_m * gamma_D * omega / (omega**2 + gamma_D**2),
                    0.0)


def J_ubo(omega, omega_j, S_j, gamma_j=GAMMA_VIB_CM):
    """Underdamped Brownian oscillator:  J_j(ω) = 4 S_j ω_j³ γ_j ω / [(ω²−ω_j²)²+ω²γ_j²]"""
    omega = np.asarray(omega, dtype=float)
    denom = (omega**2 - omega_j**2)**2 + omega**2 * gamma_j**2
    return np.where(omega > 0,
                    4.0 * S_j * omega_j**3 * gamma_j * omega / denom,
                    0.0)


def J_intra(omega, gamma_j=GAMMA_VIB_CM):
    omega = np.asarray(omega, dtype=float)
    out = np.zeros_like(omega)
    for omega_j, S_j in MODES:
        out += J_ubo(omega, omega_j, S_j, gamma_j)
    return out


def J_total(omega, site, gamma_D=GAMMA_DRUDE_CM, gamma_j=GAMMA_VIB_CM):
    return J_inter(omega, LAMBDA_SITES_CM[site], gamma_D) + J_intra(omega, gamma_j)


def gamma_site(omega, site, temperature=TEMPERATURE_K,
               gamma_D=GAMMA_DRUDE_CM, gamma_j=GAMMA_VIB_CM):
    """
    Bath correlation spectral function γ_m(ω) for brmesolve.

    WARNING: slow when called repeatedly with sharp UBO peaks.
    Use make_gamma_interp() for production runs.
    """
    omega  = np.asarray(omega, dtype=float)
    abs_w  = np.abs(omega)
    beta_w = _HBAR_OVER_KB * abs_w / temperature

    J_abs = J_total(abs_w, site, gamma_D, gamma_j)

    # Taylor limit at ω → 0 (L'Hôpital)
    lim = (2.0 * LAMBDA_SITES_CM[site] / gamma_D
           + sum(4.0 * S_j * gamma_j / omega_j for omega_j, S_j in MODES)
           ) * temperature / _HBAR_OVER_KB

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom_pos = np.where(beta_w > 1e-4, 1.0 - np.exp(-beta_w), beta_w)
        denom_neg = np.where(beta_w > 1e-4, np.expm1(beta_w),       beta_w)
        pos_raw = np.where(J_abs > 0, J_abs / denom_pos, lim)
        neg_raw = np.where(J_abs > 0, J_abs / denom_neg, lim)

    result = np.where(omega >= 0.0, pos_raw, neg_raw)
    return float(result) if result.ndim == 0 else result


# ── Fast pre-tabulated lookup (use this for brmesolve) ────────────────────────

def build_gamma_tables(
    temperature: float = TEMPERATURE_K,
    n_points: int = _N_TABLE,
    omega_max: float = _OMEGA_MAX,
    gamma_D: float = GAMMA_DRUDE_CM,
    gamma_j: float = GAMMA_VIB_CM,
) -> dict:
    """
    Pre-compute γ_m(ω) for all 8 sites on a dense symmetric frequency grid.

    Uses CuPy if available (GPU), otherwise NumPy.  The returned tables are
    always plain NumPy arrays so they can be passed to QuTiP / brmesolve.

    Returns
    -------
    tables : dict with keys:
        'omega'       : (2*n_points,) sorted frequency grid  [cm⁻¹]
        'gamma_<m>'   : (2*n_points,) γ_m(ω)  for m = 0..7
        'temperature' : float
    """
    print(f"  Building γ_m tables (T={temperature} K, {2*n_points} points, "
          f"{'GPU/CuPy' if _GPU else 'CPU/NumPy'}) …", flush=True)

    xp = _xp   # cupy or numpy

    # Positive half: dense log-spaced near the origin + uniform past 50 cm⁻¹
    # This resolves both the ω→0 Drude peak and the narrow UBO peaks.
    omega_pos = xp.concatenate([
        xp.geomspace(1e-2, 1.0,       500,  dtype=float),
        xp.linspace(  1.0, omega_max, n_points - 500, dtype=float),
    ])
    omega_neg = -omega_pos[::-1]
    omega_full = xp.concatenate([omega_neg, omega_pos])   # sorted, symmetric

    # Taylor limit at ω → 0 (same for all sites regarding UBO contribution)
    ubo_lim_coeff = sum(4.0 * S_j * float(gamma_j) / omega_j for omega_j, S_j in MODES)

    tables = {"omega": np.array(omega_full.get() if _GPU else omega_full),
              "temperature": temperature}

    for site in range(8):
        lim = (2.0 * float(LAMBDA_SITES_CM[site]) / float(gamma_D)
               + ubo_lim_coeff) * temperature / _HBAR_OVER_KB

        abs_w  = xp.abs(omega_full)
        beta_w = _HBAR_OVER_KB * abs_w / temperature

        # Compute J_total on GPU/CPU
        J_abs = (2.0 * float(LAMBDA_SITES_CM[site]) * float(gamma_D)
                 * abs_w / (abs_w**2 + float(gamma_D)**2))
        for omega_j, S_j in MODES:
            denom = (abs_w**2 - omega_j**2)**2 + abs_w**2 * float(gamma_j)**2
            J_abs += 4.0 * S_j * omega_j**3 * float(gamma_j) * abs_w / denom

        with np.errstate(divide="ignore", invalid="ignore"):
            beta_w_np = np.array(beta_w.get() if _GPU else beta_w)
            J_np      = np.array(J_abs.get()  if _GPU else J_abs)
            om_np     = np.array(omega_full.get() if _GPU else omega_full)

        denom_pos = np.where(beta_w_np > 1e-4, 1.0 - np.exp(-beta_w_np), beta_w_np)
        denom_neg = np.where(beta_w_np > 1e-4, np.expm1(beta_w_np),       beta_w_np)
        pos_raw = np.where(J_np > 0, J_np / denom_pos, lim)
        neg_raw = np.where(J_np > 0, J_np / denom_neg, lim)

        gamma_vals = np.where(om_np >= 0.0, pos_raw, neg_raw)
        tables[f"gamma_{site}"] = gamma_vals

    print("    done.", flush=True)
    return tables


def make_gamma_interp(tables: dict, site: int):
    """
    Return a fast interpolating function γ_m(ω) from a pre-built table.

    The returned callable is suitable for passing as the spectral function
    in brmesolve a_ops — it evaluates in microseconds via np.interp.
    """
    omega_grid = tables["omega"]
    gamma_grid = tables[f"gamma_{site}"]

    def _gamma_fast(omega):
        return np.interp(omega, omega_grid, gamma_grid)

    return _gamma_fast


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)

    tables = build_gamma_tables(temperature=300.0)
    omega  = tables["omega"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    mask = omega > 0

    for site, color in [(0, "#1a6e8c"), (2, "#d45f1e"), (5, "#4a9a3f")]:
        axes[0].plot(omega[mask], tables[f"gamma_{site}"][mask],
                     color=color, lw=1.4, label=f"BChl {site+1}")
    axes[0].set_xlabel("ω (cm⁻¹)"); axes[0].set_ylabel("γ_m(ω) (cm⁻¹)")
    axes[0].set_title("Bath correlation γ_m(ω), T=300 K")
    axes[0].set_xlim(0, 900); axes[0].legend(frameon=False)

    # Zoom in on UBO peaks
    mask2 = (omega > 150) & (omega < 350)
    for site, color in [(0, "#1a6e8c"), (2, "#d45f1e")]:
        axes[1].plot(omega[mask2], tables[f"gamma_{site}"][mask2],
                     color=color, lw=1.4, label=f"BChl {site+1}")
    for omega_j, _ in MODES:
        if 150 < omega_j < 350:
            axes[1].axvline(omega_j, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("ω (cm⁻¹)"); axes[1].set_title("UBO mode region (zoom)")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig("results/spectral_density_check.png", dpi=150)
    print("Saved results/spectral_density_check.png")
