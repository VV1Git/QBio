"""
Phase 1 validation: reproduce Ai et al. Figure 1 (dynamics) and Figure 2a (Reff heatmap).

Run this script directly to execute the full validation.  Results are saved to results/.

    python validate.py

Qualitative targets
-------------------
Figure 1:  Three Q(t) traces (cumulative trapping yield, 0→1) at θ=0 for
           r = 8, 11.4, 13.4 Å with a reaction-centre trap (κ=2 ps⁻¹) at site 4.
           r=11.4 Å should show the fastest / most efficient transfer.
           r=8 Å  should show reduced transfer (strong coupling → delocalised,
                  less energy-gradient drive).

Figure 2a: Reff(r, θ) heatmap with a peak near r ≈ 11.3 Å, θ ≈ 0,
           shown as both a 2-D heatmap and a 3-D surface.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from joblib import Parallel, delayed
from tqdm import tqdm

# Allow running from the energy_transfer/ directory directly
sys.path.insert(0, str(Path(__file__).parent))

from dynamics import secular_reff, run_ohmic_with_trap
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

    J2 = coupling_matrix(11.3, 0.0)
    print("\nCoupling matrix at r=11.3 Å, θ=0 (reported optimum):")
    print(f"  J12={J2[0,1]:+.1f}  J23={J2[1,2]:+.1f}  J34={J2[2,3]:+.1f}")
    print(f"  J13={J2[0,2]:+.1f}  J24={J2[1,3]:+.1f}  J14={J2[0,3]:+.1f}")


# ── Figure 1 reproduction ─────────────────────────────────────────────────────

def reproduce_figure1(
    t_end: float = 15_000.0,
    n_steps: int = 500,
    kappa_trap_fs: float = 0.001,
) -> None:
    """
    Plot reaction-centre trapping yield Q(t) for three geometries at θ=0.

    A 5-site Bloch-Redfield simulation: sites 1–4 (Frenkel Hamiltonian) plus
    an irreversible trap at site 5 (reaction centre), coupled to site 4 via a
    Lindblad collapse operator at rate κ_trap.
    Q(t) = P_trap(t) rises from 0 → 1, showing complete biological energy harvesting.
    """
    configs = [
        (8.0,  "#e05c3a"),
        (11.4, "#1f6f78"),
        (13.4, "#c8a227"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    ax_q, ax_p = axes

    for r_ang, color in tqdm(configs, desc="Fig 1 geometries", unit="geom"):
        t0 = time.time()
        times, P4, Q = run_ohmic_with_trap(
            r=r_ang, theta=0.0,
            kappa_trap_fs=kappa_trap_fs,
            t_end=t_end, n_steps=n_steps,
        )
        tqdm.write(f"  r={r_ang} Å  done in {time.time()-t0:.1f}s  Q(final)={Q[-1]:.3f}")
        times_ps = times / 1000.0  # convert fs → ps for readability
        ax_q.plot(times_ps, Q,  color=color, linewidth=2.2, label=f"r = {r_ang} Å")
        ax_p.plot(times_ps, P4, color=color, linewidth=2.2, label=f"r = {r_ang} Å",
                  ls="--")

    for ax in axes:
        ax.set_xlabel("Time (ps)", fontsize=11)
        ax.set_xlim(0, t_end / 1000)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_q.set_ylabel("Reaction-centre yield Q(t)", fontsize=11)
    ax_q.set_ylim(0, 1.05)
    ax_q.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax_q.set_title("Cumulative trapping yield (RC trap)", fontsize=10)

    ax_p.set_ylabel("Site-4 population P₄(t)", fontsize=11)
    ax_p.set_ylim(0, None)
    ax_p.set_title("Site-4 transient population", fontsize=10)

    fig.suptitle(
        f"Fig. 1: Energy funnelling to reaction centre — Ohmic bath, θ = 0  "
        f"(κ_trap = {kappa_trap_fs*1e3:.0f} ps⁻¹)",
        fontsize=11,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig1_dynamics.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Figure 2a reproduction ────────────────────────────────────────────────────

def _reff_at_point(r: float, theta: float, t_end: float, n_steps: int) -> float:
    """Compute Reff at a single (r, θ) point using secular Redfield."""
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

    Saves two panels side-by-side:
        Left  — 2-D colour heatmap
        Right — 3-D surface heightmap

    Parameters
    ----------
    r_grid     : 1-D array of r values (Å).  Default: 50 points, 8–14 Å.
    theta_grid : 1-D array of θ values (rad).  Default: 30 points, 0–π/2.
    t_end      : propagation length (fs)
    n_steps    : number of time points per simulation
    n_jobs     : joblib parallelism  (-1 = all cores)

    Returns
    -------
    Reff_grid : (len(r_grid), len(theta_grid)) array  [fs⁻¹]
    """
    if r_grid is None:
        r_grid = np.linspace(8.0, 14.0, 50)
    if theta_grid is None:
        theta_grid = np.linspace(0.0, np.pi / 2, 30)

    jobs = [(r, th) for r in r_grid for th in theta_grid]
    results_flat = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_reff_at_point)(r, th, t_end, n_steps) for r, th in jobs
        ),
        total=len(jobs), desc="Reff scan", unit="pt",
    ))

    Reff_grid = np.array(results_flat).reshape(len(r_grid), len(theta_grid))

    theta_deg = np.degrees(theta_grid)
    idx_max   = np.unravel_index(np.argmax(Reff_grid), Reff_grid.shape)
    r_opt, th_opt = r_grid[idx_max[0]], theta_grid[idx_max[1]]

    # ── Combined 2-D heatmap + 3-D surface ──────────────────────────────────
    fig = plt.figure(figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")

    # Left: 2-D heatmap
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set_facecolor("#fffdf7")
    img = ax2d.pcolormesh(
        theta_deg, r_grid, Reff_grid * 1e3,
        cmap="viridis", shading="auto",
    )
    cb = fig.colorbar(img, ax=ax2d)
    cb.set_label("Reff (ps⁻¹)", fontsize=10)
    ax2d.plot(np.degrees(th_opt), r_opt, "w*", markersize=12,
              label=f"max: r={r_opt:.1f} Å, θ={np.degrees(th_opt):.0f}°")
    ax2d.legend(frameon=False, fontsize=9, loc="upper right")
    ax2d.set_xlabel("Dipole angle θ (°)", fontsize=11)
    ax2d.set_ylabel("Intra-dimer distance r (Å)", fontsize=11)
    ax2d.set_title("2-D heatmap", fontsize=10)

    # Right: 3-D surface
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    T_deg, R_mesh = np.meshgrid(theta_deg, r_grid)
    surf = ax3d.plot_surface(
        T_deg, R_mesh, Reff_grid * 1e3,
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.93,
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.55, pad=0.08).set_label("Reff (ps⁻¹)", fontsize=9)
    ax3d.set_xlabel("θ (°)", labelpad=6, fontsize=9)
    ax3d.set_ylabel("r (Å)", labelpad=6, fontsize=9)
    ax3d.set_zlabel("Reff (ps⁻¹)", labelpad=6, fontsize=9)
    ax3d.set_title("3-D heightmap", fontsize=10)
    ax3d.view_init(elev=30, azim=225)

    fig.suptitle(
        f"Fig. 2a: Reff(r, θ) — secular Redfield  "
        f"({len(r_grid)}×{len(theta_grid)} grid)",
        fontsize=12,
    )
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
    from gpu_utils import setup_gpu
    setup_gpu()

    parser = argparse.ArgumentParser(description="Phase 1 validation against Ai et al.")
    parser.add_argument("--fig1-only",   action="store_true", help="Only run Figure 1")
    parser.add_argument("--fig2-only",   action="store_true", help="Only run Figure 2a")
    parser.add_argument("--quick",       action="store_true",
                        help="Fast low-resolution run (for testing)")
    args = parser.parse_args()

    print("=== Unit check: coupling matrix ===")
    unit_check()

    run_fig1 = not args.fig2_only
    run_fig2 = not args.fig1_only

    t_end_1  = 8_000.0   if args.quick else 15_000.0
    t_end_2  = 50_000.0  if args.quick else 200_000.0
    r_pts    = 12         if args.quick else 50
    th_pts   = 8          if args.quick else 30

    if run_fig1:
        print("\n=== Figure 1: trapping yield Q(t) ===")
        reproduce_figure1(
            t_end=t_end_1,
            n_steps=250 if args.quick else 500,
        )

    if run_fig2:
        print("\n=== Figure 2a: Reff heatmap + 3-D surface ===")
        reproduce_figure2a(
            r_grid=np.linspace(8.0, 14.0, r_pts),
            theta_grid=np.linspace(0.0, np.pi / 2, th_pts),
            t_end=t_end_2,
            n_steps=200 if args.quick else 300,
        )

    print("\nDone.")
