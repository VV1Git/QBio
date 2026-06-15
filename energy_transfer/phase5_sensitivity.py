"""
Phase 5: Bath parameter sensitivity analysis at optimal geometry.

Asks: how does Reff change as we vary the bath parameters away from
the FMO defaults?  We fix geometry at r=11.3 Å, θ=0 (optimal from Phase 1)
and sweep three axes:
    (a) Reorganisation energy λ  :  5 – 150 cm⁻¹   (default 35 cm⁻¹)
    (b) Temperature T            : 100 – 500 K       (default 300 K)
    (c) Bath cut-off γ           :  10 – 200 cm⁻¹   (default 53 cm⁻¹)

For each axis we run both Ohmic (secular_reff) and vibronic (vibronic_reff)
so we can see where structured vibrations amplify or suppress the effect.

Output: fig6_bath_sensitivity.png  (3 panels, Ohmic + vibronic on each)
        p5_sensitivity_data.npz   (raw arrays for reuse in Phase 6)

Usage
-----
    python phase5_sensitivity.py
    python phase5_sensitivity.py --quick     # coarser grids
"""

from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import secular_reff
from vibronic import vibronic_reff
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

R_OPT     = 11.3   # Å — optimal geometry from Phase 1
THETA_OPT = 0.0    # rad

# ── Grid defaults ──────────────────────────────────────────────────────────────

N_FULL  = 18
N_QUICK = 9

LAMBDA_RANGE = (5.0,  150.0)
TEMP_RANGE   = (100.0, 500.0)
GAMMA_RANGE  = (10.0, 200.0)


def _make_grid(lo, hi, n):
    return np.linspace(lo, hi, n)


# ── Per-point wrappers ─────────────────────────────────────────────────────────

def _ohm(r, th, lambda_, gamma_bath, temperature, t_end=200_000.0):
    try:
        _, _, reff = secular_reff(r, th,
                                  lambda_=lambda_, gamma_bath=gamma_bath,
                                  temperature=temperature, t_end=t_end, n_steps=400)
        return reff
    except Exception:
        return 0.0


def _vib(r, th, lambda_, gamma_bath, temperature, t_end=5000.0, n_steps=300):
    try:
        _, _, reff = vibronic_reff(r, th, t_end=t_end, n_steps=n_steps,
                                   lambda_=lambda_, gamma_bath=gamma_bath,
                                   temperature=temperature)
        return reff
    except Exception:
        return 0.0


def scan_axis(param_name: str, values: np.ndarray,
              t_end_vib: float, n_steps_vib: int,
              n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep one bath parameter, keeping the others at their defaults.

    Returns (reff_ohm, reff_vib) arrays with shape (len(values),).
    """
    print(f"  {param_name} sweep ({len(values)} points) …", flush=True)
    t0 = time.time()

    def _kwargs(v):
        base = dict(lambda_=LAMBDA_CM, gamma_bath=GAMMA_CM, temperature=TEMPERATURE_K)
        if param_name == "lambda":
            base["lambda_"] = v
        elif param_name == "temperature":
            base["temperature"] = v
        elif param_name == "gamma":
            base["gamma_bath"] = v
        return base

    ohm_vals = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_ohm)(R_OPT, THETA_OPT, **_kwargs(v)) for v in values
    )
    print(f"    Ohmic done in {time.time()-t0:.1f}s")
    t0 = time.time()

    vib_vals = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_vib)(R_OPT, THETA_OPT, t_end=t_end_vib, n_steps=n_steps_vib,
                      **_kwargs(v)) for v in values
    )
    print(f"    Vibronic done in {time.time()-t0:.1f}s")

    return np.array(ohm_vals), np.array(vib_vals)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_sensitivity(
    lambda_grid, ohm_lam, vib_lam,
    temp_grid,   ohm_tmp, vib_tmp,
    gamma_grid,  ohm_gam, vib_gam,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")

    panels = [
        (axes[0], lambda_grid, ohm_lam, vib_lam,
         "Reorganisation energy λ (cm⁻¹)", LAMBDA_CM, "default\n35 cm⁻¹"),
        (axes[1], temp_grid,   ohm_tmp, vib_tmp,
         "Temperature T (K)",               TEMPERATURE_K, "default\n300 K"),
        (axes[2], gamma_grid,  ohm_gam, vib_gam,
         "Bath cut-off γ (cm⁻¹)",          GAMMA_CM, "default\n53 cm⁻¹"),
    ]

    for ax, xvals, ohm, vib, xlabel, default, def_label in panels:
        ax.set_facecolor("#fffdf7")
        ax.plot(xvals, ohm * 1e3, color="#1a6e8c", lw=2.0, label="Ohmic (secular)")
        ax.plot(xvals, vib * 1e3, color="#d45f1e", lw=2.0, ls="--", label="Vibronic (Lindblad)")
        ax.axvline(default, color="gray", ls=":", lw=1.0, label=def_label)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Reff (ps⁻¹)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Fig. 6: Reff sensitivity to bath parameters — r={R_OPT} Å, θ=0°",
        fontsize=11
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig6_bath_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    n        = N_QUICK if args.quick else N_FULL
    t_vib    = 2000.0  if args.quick else 5000.0
    n_s_vib  = 150     if args.quick else 300

    lam_grid = _make_grid(*LAMBDA_RANGE, n)
    tmp_grid = _make_grid(*TEMP_RANGE,   n)
    gam_grid = _make_grid(*GAMMA_RANGE,  n)

    print("=== Phase 5: Bath parameter sensitivity ===")
    print(f"  Geometry: r={R_OPT} Å, θ=0°  |  grid size: {n} per axis")

    print("\n[1/3] λ sweep")
    ohm_lam, vib_lam = scan_axis("lambda",      lam_grid, t_vib, n_s_vib, args.n_jobs)

    print("\n[2/3] T sweep")
    ohm_tmp, vib_tmp = scan_axis("temperature",  tmp_grid, t_vib, n_s_vib, args.n_jobs)

    print("\n[3/3] γ sweep")
    ohm_gam, vib_gam = scan_axis("gamma",        gam_grid, t_vib, n_s_vib, args.n_jobs)

    np.savez(
        RESULTS_DIR / "p5_sensitivity_data.npz",
        lambda_grid=lam_grid,  ohm_lam=ohm_lam, vib_lam=vib_lam,
        temp_grid=tmp_grid,    ohm_tmp=ohm_tmp, vib_tmp=vib_tmp,
        gamma_grid=gam_grid,   ohm_gam=ohm_gam, vib_gam=vib_gam,
    )
    print("  Raw data saved to p5_sensitivity_data.npz")

    plot_sensitivity(lam_grid, ohm_lam, vib_lam,
                     tmp_grid, ohm_tmp, vib_tmp,
                     gam_grid, ohm_gam, vib_gam)
    print("\nDone.")
