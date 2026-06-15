"""
Phase 4: Production Reff(r, θ) scan — Ohmic vs vibronic, higher resolution.

Produces fig5_reff_comparison.png with three panels:
    Left   : Reff(r,θ) — Ohmic (secular Redfield), ultra-fast
    Centre : Reff(r,θ) — vibronic (Lindblad mesolve), heavier
    Right  : ΔReff = Reff_vib − Reff_ohm  (signed difference)

Grid: 16 r-values × 10 θ-values  (160 points).
The vibronic scan is parallelised with joblib.

Usage
-----
    python phase4_scan.py                 # full run
    python phase4_scan.py --quick         # 8×6 grid, short propagation
    python phase4_scan.py --ohmic-only    # only the secular-Redfield heatmap
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import secular_reff
from vibronic import vibronic_reff

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Grid defaults ──────────────────────────────────────────────────────────────

R_GRID_FULL     = np.linspace(7.0, 15.0, 16)    # Å
THETA_GRID_FULL = np.linspace(0.0, np.pi / 2, 10)  # rad, 0–90°

R_GRID_QUICK     = np.linspace(7.0, 15.0, 8)
THETA_GRID_QUICK = np.linspace(0.0, np.pi / 2, 6)


# ── Per-point helpers (picklable for joblib) ───────────────────────────────────

def _ohmic_point(r, theta):
    try:
        _, _, reff = secular_reff(r, theta, t_end=200_000.0, n_steps=400)
        return reff
    except Exception:
        return 0.0


def _vibronic_point(r, theta, t_end, n_steps):
    try:
        _, _, reff = vibronic_reff(r, theta, t_end=t_end, n_steps=n_steps)
        return reff
    except Exception:
        return 0.0


# ── Scan functions ─────────────────────────────────────────────────────────────

def scan_ohmic(r_grid, theta_grid, n_jobs=-1) -> np.ndarray:
    """Secular Redfield scan — microseconds per point."""
    print(f"  Ohmic scan: {len(r_grid)}×{len(theta_grid)} points …", flush=True)
    t0 = time.time()
    jobs = [(r, th) for r in r_grid for th in theta_grid]
    flat = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_ohmic_point)(r, th) for r, th in jobs
    )
    grid = np.array(flat).reshape(len(r_grid), len(theta_grid))
    print(f"    done in {time.time()-t0:.1f}s")
    return grid


def scan_vibronic(r_grid, theta_grid, t_end, n_steps, n_jobs=-1) -> np.ndarray:
    """Vibronic Lindblad scan — heavier, fully parallelised."""
    total = len(r_grid) * len(theta_grid)
    print(f"  Vibronic scan: {len(r_grid)}×{len(theta_grid)} = {total} points "
          f"(~{total*3//60}–{total*6//60} min on 1 core) …", flush=True)
    t0 = time.time()
    jobs = [(r, th) for r in r_grid for th in theta_grid]
    flat = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_vibronic_point)(r, th, t_end, n_steps) for r, th in jobs
    )
    grid = np.array(flat).reshape(len(r_grid), len(theta_grid))
    print(f"    done in {time.time()-t0:.1f}s")
    return grid


# ── Plotting ───────────────────────────────────────────────────────────────────

def _add_star(ax, r_grid, theta_grid, grid, theta_deg, color="w"):
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    r_opt, th_opt = r_grid[idx[0]], theta_grid[idx[1]]
    ax.plot(np.degrees(th_opt), r_opt, "*", color=color, ms=11,
            label=f"r={r_opt:.1f} Å, θ={np.degrees(th_opt):.0f}°")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_comparison(
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    ohmic_grid: np.ndarray,
    vibronic_grid: np.ndarray | None,
    tag: str = "",
) -> None:
    theta_deg = np.degrees(theta_grid)
    n_panels  = 3 if vibronic_grid is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), dpi=150)
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f7f2ea")

    # Panel 1: Ohmic
    ax = axes[0]
    vmin, vmax = 0.0, ohmic_grid.max() * 1e3
    img = ax.pcolormesh(theta_deg, r_grid, ohmic_grid * 1e3,
                        cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax).set_label("Reff (ps⁻¹)", fontsize=9)
    _add_star(ax, r_grid, theta_grid, ohmic_grid, theta_deg)
    ax.set_title("Ohmic (secular Redfield)", fontsize=10)
    ax.set_xlabel("Dipole angle θ (°)"); ax.set_ylabel("r (Å)")

    if vibronic_grid is not None:
        # Panel 2: Vibronic
        ax2 = axes[1]
        v2max = vibronic_grid.max() * 1e3
        img2 = ax2.pcolormesh(theta_deg, r_grid, vibronic_grid * 1e3,
                               cmap="viridis", shading="auto", vmin=0, vmax=v2max)
        fig.colorbar(img2, ax=ax2).set_label("Reff (ps⁻¹)", fontsize=9)
        _add_star(ax2, r_grid, theta_grid, vibronic_grid, theta_deg)
        ax2.set_title("Vibronic (Lindblad)", fontsize=10)
        ax2.set_xlabel("Dipole angle θ (°)")

        # Panel 3: difference
        ax3 = axes[2]
        delta = (vibronic_grid - ohmic_grid) * 1e3   # ps⁻¹
        abs_max = np.abs(delta).max()
        img3 = ax3.pcolormesh(theta_deg, r_grid, delta,
                               cmap="RdBu_r", shading="auto",
                               vmin=-abs_max, vmax=abs_max)
        cb3 = fig.colorbar(img3, ax=ax3)
        cb3.set_label("ΔReff = vib − Ohm (ps⁻¹)", fontsize=9)
        ax3.set_title("Δ Reff (vibronic − Ohmic)", fontsize=10)
        ax3.set_xlabel("Dipole angle θ (°)")

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
    parser.add_argument("--n-jobs",     type=int, default=-1)
    args = parser.parse_args()

    r_grid     = R_GRID_QUICK     if args.quick else R_GRID_FULL
    theta_grid = THETA_GRID_QUICK if args.quick else THETA_GRID_FULL
    t_end_vib  = 2000.0 if args.quick else 5000.0
    n_steps_vib = 150   if args.quick else 300

    print("=== Phase 4: Production Reff(r,θ) scan ===")

    print("\n[1/2] Ohmic (secular Redfield)")
    ohmic_grid = scan_ohmic(r_grid, theta_grid, n_jobs=args.n_jobs)

    # Save after Ohmic in case vibronic is interrupted
    np.save(RESULTS_DIR / "p4_reff_ohmic.npy",  ohmic_grid)
    np.save(RESULTS_DIR / "p4_r_grid.npy",      r_grid)
    np.save(RESULTS_DIR / "p4_theta_grid.npy",  theta_grid)

    vibronic_grid = None
    if not args.ohmic_only:
        print("\n[2/2] Vibronic (Lindblad mesolve)")
        vibronic_grid = scan_vibronic(r_grid, theta_grid, t_end_vib, n_steps_vib,
                                      n_jobs=args.n_jobs)
        np.save(RESULTS_DIR / "p4_reff_vibronic.npy", vibronic_grid)

        idx_o  = np.unravel_index(np.argmax(ohmic_grid),    ohmic_grid.shape)
        idx_v  = np.unravel_index(np.argmax(vibronic_grid), vibronic_grid.shape)
        print(f"  Ohmic optimum:    r={r_grid[idx_o[0]]:.1f} Å, "
              f"θ={np.degrees(theta_grid[idx_o[1]]):.0f}°, "
              f"Reff={ohmic_grid[idx_o]*1e3:.3f} ps⁻¹")
        print(f"  Vibronic optimum: r={r_grid[idx_v[0]]:.1f} Å, "
              f"θ={np.degrees(theta_grid[idx_v[1]]):.0f}°, "
              f"Reff={vibronic_grid[idx_v]*1e3:.3f} ps⁻¹")

    tag = " (quick)" if args.quick else f" — {len(r_grid)}×{len(theta_grid)}"
    plot_comparison(r_grid, theta_grid, ohmic_grid, vibronic_grid, tag=tag)
    print("\nDone.")
