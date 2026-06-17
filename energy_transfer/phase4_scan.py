"""
Phase 4: Production Reff(r, θ) scan — Ohmic vs vibronic, higher resolution.

Produces one output figure:
    fig5_reff_comparison.png  — three 2-D heatmap panels (Ohmic | Vibronic | ΔReff)

Grid: 120 r-values × 80 θ-values for Ohmic (secular Redfield, ms/point).
      40 r-values × 28 θ-values for Vibronic (Lindblad mesolve, parallelised
      across all CPU cores via joblib).
The ΔReff difference panel uses the vibronic grid interpolated to the ohmic grid.

Usage
-----
    python phase4_scan.py                 # full run
    python phase4_scan.py --quick         # 12×8 / 10×6 grid, short propagation
    python phase4_scan.py --ohmic-only    # only the secular-Redfield heatmaps
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import secular_reff
from vibronic import vibronic_reff

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Grid defaults ──────────────────────────────────────────────────────────────

# Distance range r ∈ [8, 14] Å — physical BChl Mg–Mg separations (closer than
# ~8 Å is below van der Waals contact and pushes secular Redfield out of its
# weak-coupling validity range).
# Ohmic (secular solver — ms per point, embarrassingly parallel → very fine)
R_GRID_OHM_FULL     = np.linspace(8.0, 14.0, 120)
THETA_GRID_OHM_FULL = np.linspace(0.0, np.pi / 2, 80)

R_GRID_OHM_QUICK     = np.linspace(8.0, 14.0, 16)
THETA_GRID_OHM_QUICK = np.linspace(0.0, np.pi / 2, 10)

# Vibronic (Lindblad mesolve — ~10 s/point on 1 core; 24-core joblib parallel)
R_GRID_VIB_FULL     = np.linspace(8.0, 14.0, 32)
THETA_GRID_VIB_FULL = np.linspace(0.0, np.pi / 2, 22)

R_GRID_VIB_QUICK     = np.linspace(8.0, 14.0, 10)
THETA_GRID_VIB_QUICK = np.linspace(0.0, np.pi / 2, 6)


# ── Per-point helpers (picklable for joblib) ───────────────────────────────────

def _ohmic_point(r, theta):
    try:
        _, _, reff = secular_reff(r, theta, t_end=200_000.0, n_steps=400)
        return reff
    except Exception:
        return 0.0


def _vibronic_point(r, theta, t_end, n_steps):
    try:
        # n_max=3 (dim 64): Reff is insensitive to Fock depth (<0.25 %), ~3× faster
        _, _, reff = vibronic_reff(r, theta, t_end=t_end, n_steps=n_steps, n_max=3)
        return reff
    except Exception:
        return 0.0


# ── Scan functions ─────────────────────────────────────────────────────────────

def scan_ohmic(r_grid, theta_grid, n_jobs=-1) -> np.ndarray:
    jobs = [(r, th) for r in r_grid for th in theta_grid]
    flat = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_ohmic_point)(r, th) for r, th in jobs
        ),
        total=len(jobs), desc="ohmic", unit="pt",
    ))
    return np.array(flat).reshape(len(r_grid), len(theta_grid))


def scan_vibronic(r_grid, theta_grid, t_end, n_steps, n_jobs=-1) -> np.ndarray:
    jobs = [(r, th) for r in r_grid for th in theta_grid]
    flat = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_vibronic_point)(r, th, t_end, n_steps) for r, th in jobs
        ),
        total=len(jobs), desc="vibronic", unit="pt",
    ))
    return np.array(flat).reshape(len(r_grid), len(theta_grid))


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _add_star(ax, r_grid, theta_grid, grid, color="w"):
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    r_opt, th_opt = r_grid[idx[0]], theta_grid[idx[1]]
    ax.plot(np.degrees(th_opt), r_opt, "*", color=color, ms=11,
            label=f"r={r_opt:.1f} Å, θ={np.degrees(th_opt):.0f}°")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# ── 2-D heatmap figure ─────────────────────────────────────────────────────────

def plot_comparison_2d(
    r_ohm: np.ndarray,
    theta_ohm: np.ndarray,
    ohmic_grid: np.ndarray,
    r_vib: np.ndarray | None,
    theta_vib: np.ndarray | None,
    vibronic_grid: np.ndarray | None,
    tag: str = "",
) -> None:
    """Three-panel 2-D heatmap: Ohmic | Vibronic | ΔReff."""
    theta_deg_ohm = np.degrees(theta_ohm)
    n_panels = 3 if vibronic_grid is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), dpi=150)
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f7f2ea")

    ax = axes[0]
    # Cap at the 98th percentile: the sub-100 fs resonance ridge (Redfield
    # breakdown regime) would otherwise saturate the scale and hide the gradient.
    vmax = float(np.percentile(ohmic_grid * 1e3, 98))
    img = ax.pcolormesh(theta_deg_ohm, r_ohm, ohmic_grid * 1e3,
                        cmap="viridis", shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(img, ax=ax, extend="max").set_label("Reff (ps⁻¹)", fontsize=9)
    _add_star(ax, r_ohm, theta_ohm, ohmic_grid)
    ax.set_title("Ohmic (secular Redfield)", fontsize=10)
    ax.set_xlabel("Dipole angle θ (°)")
    ax.set_ylabel("r (Å)")

    if vibronic_grid is not None:
        theta_deg_vib = np.degrees(theta_vib)

        ax2 = axes[1]
        v2max = vibronic_grid.max() * 1e3
        img2 = ax2.pcolormesh(theta_deg_vib, r_vib, vibronic_grid * 1e3,
                               cmap="viridis", shading="auto", vmin=0, vmax=v2max)
        fig.colorbar(img2, ax=ax2).set_label("Reff (ps⁻¹)", fontsize=9)
        _add_star(ax2, r_vib, theta_vib, vibronic_grid)
        ax2.set_title("Vibronic (Lindblad)", fontsize=10)
        ax2.set_xlabel("Dipole angle θ (°)")

        # Interpolate vibronic onto ohmic grid for the difference panel
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (r_vib, theta_vib), vibronic_grid, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        R_ohm_mg, T_ohm_mg = np.meshgrid(r_ohm, theta_ohm, indexing="ij")
        pts = np.column_stack([R_ohm_mg.ravel(), T_ohm_mg.ravel()])
        vib_on_ohm = interp(pts).reshape(len(r_ohm), len(theta_ohm))

        ax3 = axes[2]
        delta = (vib_on_ohm - ohmic_grid) * 1e3
        abs_max = np.abs(delta).max()
        img3 = ax3.pcolormesh(theta_deg_ohm, r_ohm, delta,
                               cmap="RdBu_r", shading="auto",
                               vmin=-abs_max, vmax=abs_max)
        fig.colorbar(img3, ax=ax3).set_label("ΔReff = vib − Ohm (ps⁻¹)", fontsize=9)
        ax3.set_title("Δ Reff (vibronic − Ohmic)", fontsize=10)
        ax3.set_xlabel("Dipole angle θ (°)")

        idx_o = np.unravel_index(np.argmax(ohmic_grid),    ohmic_grid.shape)
        idx_v = np.unravel_index(np.argmax(vibronic_grid), vibronic_grid.shape)
        print(f"  Ohmic optimum:    r={r_ohm[idx_o[0]]:.1f} Å, "
              f"θ={np.degrees(theta_ohm[idx_o[1]]):.0f}°, "
              f"Reff={ohmic_grid[idx_o]*1e3:.3f} ps⁻¹")
        print(f"  Vibronic optimum: r={r_vib[idx_v[0]]:.1f} Å, "
              f"θ={np.degrees(theta_vib[idx_v[1]]):.0f}°, "
              f"Reff={vibronic_grid[idx_v]*1e3:.3f} ps⁻¹")

    fig.suptitle(f"Fig. 5: Production Reff(r,θ) scan{tag}", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "fig5_reff_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",      action="store_true", help="Smaller grid, shorter propagation")
    parser.add_argument("--ohmic-only", action="store_true", help="Skip vibronic scan")
    parser.add_argument("--n-jobs",     type=int, default=-1,
                        help="Parallel workers (default: all cores)")
    args = parser.parse_args()

    n_jobs = args.n_jobs

    if args.quick:
        r_ohm     = R_GRID_OHM_QUICK
        theta_ohm = THETA_GRID_OHM_QUICK
        r_vib     = R_GRID_VIB_QUICK
        theta_vib = THETA_GRID_VIB_QUICK
        t_end_vib, n_steps_vib = 2000.0, 150
    else:
        r_ohm     = R_GRID_OHM_FULL
        theta_ohm = THETA_GRID_OHM_FULL
        r_vib     = R_GRID_VIB_FULL
        theta_vib = THETA_GRID_VIB_FULL
        t_end_vib, n_steps_vib = 5000.0, 300

    print("=== Phase 4: Production Reff(r,θ) scan ===")
    print(f"  Ohmic grid:    {len(r_ohm)}×{len(theta_ohm)} = {len(r_ohm)*len(theta_ohm)} pts")
    print(f"  Vibronic grid: {len(r_vib)}×{len(theta_vib)} = {len(r_vib)*len(theta_vib)} pts")

    print("\n[1/2] Ohmic (secular Redfield)")
    ohmic_grid = scan_ohmic(r_ohm, theta_ohm, n_jobs=n_jobs)
    np.save(RESULTS_DIR / "p4_reff_ohmic.npy",  ohmic_grid)
    np.save(RESULTS_DIR / "p4_r_grid.npy",      r_ohm)
    np.save(RESULTS_DIR / "p4_theta_grid.npy",  theta_ohm)

    vibronic_grid = None
    r_vib_used = theta_vib_used = None
    if not args.ohmic_only:
        print("\n[2/2] Vibronic (Lindblad mesolve)")
        vibronic_grid = scan_vibronic(r_vib, theta_vib, t_end_vib, n_steps_vib, n_jobs=n_jobs)
        np.save(RESULTS_DIR / "p4_reff_vibronic.npy",  vibronic_grid)
        np.save(RESULTS_DIR / "p4_r_grid_vib.npy",     r_vib)
        np.save(RESULTS_DIR / "p4_theta_grid_vib.npy", theta_vib)
        r_vib_used     = r_vib
        theta_vib_used = theta_vib

    tag = " (quick)" if args.quick else f" — {len(r_ohm)}×{len(theta_ohm)} / {len(r_vib)}×{len(theta_vib)}"

    print("\nGenerating 2-D heatmaps …")
    plot_comparison_2d(r_ohm, theta_ohm, ohmic_grid,
                       r_vib_used, theta_vib_used, vibronic_grid, tag=tag)

    print("\nDone.")
