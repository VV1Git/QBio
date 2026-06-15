"""
Phase 1 validation: reproduce Ai et al. Figure 1 (dynamics) and Figure 2a (Reff heatmap).

Run this script directly to execute the full validation.  Results are saved to results/.

    python validate.py

Qualitative targets
-------------------
Figure 1:  Three P₄(t) traces at θ=0 for r = 8, 11.4, 13.4 Å.
           r=11.4 Å should show the fastest / most efficient transfer.
           r=8 Å  should show reduced transfer (strong coupling → delocalised,
                  less energy-gradient drive).

Figure 2a: Reff(r, θ) heatmap with a peak near r ≈ 11.3 Å, θ ≈ 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed

# Allow running from the energy_transfer/ directory directly
sys.path.insert(0, str(Path(__file__).parent))

from dynamics import run_ohmic, population, secular_reff
from efficiency import compute_Reff
from hamiltonian import build_electronic_H, coupling_matrix


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Unit check ────────────────────────────────────────────────────────────────

def unit_check() -> None:
    """Print coupling values at r=10.8 Å, θ=0 for manual comparison with paper."""
    r, theta = 10.8, 0.0
    J = coupling_matrix(r, theta)
    print("Coupling matrix at r=10.8 Å, θ=0 (cm⁻¹):")
    print(f"  J12={J[0,1]:+.1f}  J23={J[1,2]:+.1f}  J34={J[2,3]:+.1f}")
    print(f"  J13={J[0,2]:+.1f}  J24={J[1,3]:+.1f}  J14={J[0,3]:+.1f}")

    # Also print at r=11.3 (reported optimum)
    J2 = coupling_matrix(11.3, 0.0)
    print("\nCoupling matrix at r=11.3 Å, θ=0 (reported optimum):")
    print(f"  J12={J2[0,1]:+.1f}  J23={J2[1,2]:+.1f}  J34={J2[2,3]:+.1f}")
    print(f"  J13={J2[0,2]:+.1f}  J24={J2[1,3]:+.1f}  J14={J2[0,3]:+.1f}")


# ── Figure 1 reproduction ─────────────────────────────────────────────────────

def reproduce_figure1(t_end: float = 5000.0, n_steps: int = 400) -> None:
    """Plot P₄(t) for three geometries at θ=0, mirroring Ai et al. Fig. 1."""
    configs = [
        (8.0,  "#e05c3a"),
        (11.4, "#1f6f78"),
        (13.4, "#c8a227"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    ax.set_facecolor("#fffdf7")

    for r_ang, color in configs:
        print(f"  Running r={r_ang} Å …", flush=True)
        t0 = time.time()
        times, rhos = run_ohmic(r=r_ang, theta=0.0, t_end=t_end, n_steps=n_steps)
        P4 = population(rhos, site=3)
        print(f"    done in {time.time()-t0:.1f}s  P4(∞) = {P4[-1]:.3f}")
        ax.plot(times, P4, color=color, linewidth=2.2, label=f"r = {r_ang} Å")

    ax.set_xlabel("Time (fs)", fontsize=11)
    ax.set_ylabel("Acceptor population P₄(t)", fontsize=11)
    ax.set_title("Fig. 1 reproduction: site-4 dynamics, Ohmic bath, θ = 0", fontsize=12)
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "fig1_dynamics.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Figure 2a reproduction ────────────────────────────────────────────────────

def _reff_at_point(r: float, theta: float, t_end: float, n_steps: int) -> float:
    """Compute Reff at a single (r, θ) point using secular Redfield (Fix 2)."""
    try:
        _, _, reff = secular_reff(r, theta, t_end=t_end, n_steps=n_steps)
        return reff
    except Exception:
        return 0.0


def reproduce_figure2a(
    r_grid: np.ndarray | None = None,
    theta_grid: np.ndarray | None = None,
    t_end: float = 200_000.0,
    n_steps: int = 400,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute and plot the Reff(r, θ) heatmap, mirroring Ai et al. Fig. 2a.

    Parameters
    ----------
    r_grid     : 1-D array of r values (Å).  Default: 10 points, 8–14 Å.
    theta_grid : 1-D array of θ values (rad).  Default: 8 points, 0–π/2.
    t_end      : propagation length (fs)
    n_steps    : number of time points per simulation
    n_jobs     : joblib parallelism  (-1 = all cores)

    Returns
    -------
    Reff_grid : (len(r_grid), len(theta_grid)) array  [fs⁻¹]
    """
    if r_grid is None:
        r_grid = np.linspace(8.0, 14.0, 10)
    if theta_grid is None:
        theta_grid = np.linspace(0.0, np.pi / 2, 8)

    print(f"  Scanning {len(r_grid)}×{len(theta_grid)} = {len(r_grid)*len(theta_grid)} points "
          f"(secular Redfield, ~instantaneous) …")

    jobs = [(r, th) for r in r_grid for th in theta_grid]
    results_flat = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_reff_at_point)(r, th, t_end, n_steps)
        for r, th in jobs
    )

    Reff_grid = np.array(results_flat).reshape(len(r_grid), len(theta_grid))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")

    theta_deg = np.degrees(theta_grid)
    img = ax.pcolormesh(
        theta_deg, r_grid, Reff_grid * 1e3,   # convert fs⁻¹ → ps⁻¹ for readability
        cmap="viridis", shading="auto",
    )
    cb = fig.colorbar(img, ax=ax)
    cb.set_label("Reff (ps⁻¹)", fontsize=10)

    # Mark the maximum
    idx_max = np.unravel_index(np.argmax(Reff_grid), Reff_grid.shape)
    r_opt, th_opt = r_grid[idx_max[0]], theta_grid[idx_max[1]]
    ax.plot(np.degrees(th_opt), r_opt, "w*", markersize=12,
            label=f"max: r={r_opt:.1f} Å, θ={np.degrees(th_opt):.0f}°")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax.set_xlabel("Dipole angle θ (°)", fontsize=11)
    ax.set_ylabel("Intra-dimer distance r (Å)", fontsize=11)
    ax.set_title("Fig. 2a reproduction: Reff(r, θ), secular Redfield", fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig2a_reff_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")

    # Save raw data
    np.save(RESULTS_DIR / "reff_ohmic.npy", Reff_grid)
    np.save(RESULTS_DIR / "r_grid.npy", r_grid)
    np.save(RESULTS_DIR / "theta_grid.npy", theta_grid)
    print(f"  Optimal geometry: r = {r_opt:.2f} Å, θ = {np.degrees(th_opt):.1f}°")
    print(f"  Reff at optimum:  {Reff_grid[idx_max]:.4e} fs⁻¹")

    return Reff_grid


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 validation against Ai et al.")
    parser.add_argument("--fig1-only",   action="store_true", help="Only run Figure 1")
    parser.add_argument("--fig2-only",   action="store_true", help="Only run Figure 2a")
    parser.add_argument("--quick",       action="store_true",
                        help="Fast low-resolution run (for testing)")
    args = parser.parse_args()

    print("=== Unit check: coupling matrix ===")
    unit_check()

    run_fig1  = not args.fig2_only
    run_fig2  = not args.fig1_only

    t_end_1  = 2000.0  if args.quick else 5000.0
    t_end_2  = 50_000.0 if args.quick else 200_000.0
    r_pts    = 6      if args.quick else 10
    th_pts   = 5      if args.quick else 8

    if run_fig1:
        print("\n=== Figure 1: dynamics traces ===")
        reproduce_figure1(t_end=t_end_1, n_steps=300 if args.quick else 400)

    if run_fig2:
        print("\n=== Figure 2a: Reff heatmap ===")
        reproduce_figure2a(
            r_grid=np.linspace(8.0, 14.0, r_pts),
            theta_grid=np.linspace(0.0, np.pi / 2, th_pts),
            t_end=t_end_2,
            n_steps=200 if args.quick else 300,
        )

    print("\nDone.")
