"""
Phase 4: Position-vs-efficiency scan for the 8-site FMO complex.

Replaces the earlier (r, theta) geometry sweep.  Here the real FMO Hamiltonian
is fixed by structure (fmo_data.py), and we ask the biological question:
how does *moving each bacteriochlorophyll* change the excitation-transfer
efficiency (ETE) to the reaction centre, and is the native layout optimal?

Output: fig5_position_scan.png
    * 8 heatmaps (one per BChl): ETE vs in-plane displacement of that pigment,
      with the native position (star) and that pigment's best spot (x) marked.
    * 1 panel: the globally optimised arrangement (all pigments moved together
      to minimise trapping time, subject to a steric floor) as native->optimised
      in-plane displacement arrows, annotated with the ETE / trapping-time gain.

Also saved: p4_position_scan.npz (grids + optimisation result) for Phase 6.

Usage
-----
    python phase4_scan.py
    python phase4_scan.py --quick          # coarse grids, fewer optimiser iters
    python phase4_scan.py --no-optimize    # skip the global optimisation
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from fmo_data import N_SITES, TRAP_SITE, ENTRY_SITES, MG_COORDS_ANG
from geometry_scan import (
    scan_pigment, optimize_arrangement, PLANE_AXES, CENTROID, efficiency,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _role(i: int) -> str:
    if i == TRAP_SITE:
        return " (trap→RC)"
    if i in ENTRY_SITES:
        return " (entry)"
    return ""


def run_scan(n_grid: int, span: float, n_jobs: int = -1
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan all 8 pigments.

    Returns (us, vs, ete[8,n,n], rate[8,n,n]) where rate = 1/tau in ps^-1.
    """
    us = np.linspace(-span, span, n_grid)
    vs = np.linspace(-span, span, n_grid)
    ete = np.empty((N_SITES, n_grid, n_grid))
    rate = np.empty((N_SITES, n_grid, n_grid))
    for p in tqdm(range(N_SITES), desc="per-pigment scan", unit="BChl"):
        e, tau = scan_pigment(p, us, vs, n_jobs=n_jobs)
        ete[p] = e
        rate[p] = 1.0 / (tau / 1000.0)        # ps^-1
    return us, vs, ete, rate


def plot_position_scan(us, vs, ete, rate, opt: dict | None,
                       out_name: str = "fig5_position_scan.png", tag: str = "") -> None:
    """
    3x3 figure: 8 per-pigment trapping-rate heatmaps + 1 global-optimisation
    panel.  Rate = 1/trapping-time (ps^-1); higher = faster, more efficient
    funnelling to the reaction centre.  ETE range is annotated per panel.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    axes = axes.ravel()

    # Shared colour scale across pigment panels (98th pct cap avoids one
    # compact-geometry hot-spot saturating the scale).
    vmin = float(np.nanmin(rate))
    vmax = float(np.nanpercentile(rate, 98))

    ete_native, tau_native = efficiency(np.zeros((N_SITES, 3)))
    rate_native = 1.0 / (tau_native / 1000.0)

    img = None
    for p in range(N_SITES):
        ax = axes[p]
        ax.set_facecolor("#fffdf7")
        img = ax.pcolormesh(vs, us, rate[p], cmap="magma",
                            shading="gouraud", vmin=vmin, vmax=vmax)
        # native position = origin of the displacement grid
        ax.plot(0, 0, "*", color="#39d0ff", ms=15, markeredgecolor="black",
                markeredgewidth=0.7)
        # this pigment's fastest spot
        bi = np.unravel_index(np.nanargmax(rate[p]), rate[p].shape)
        ax.plot(vs[bi[1]], us[bi[0]], "X", color="white", ms=9,
                markeredgecolor="black", markeredgewidth=0.6)
        ax.set_title(f"BChl {p+1}{_role(p)}   "
                     f"ETE {np.nanmin(ete[p]):.3f}–{np.nanmax(ete[p]):.3f}",
                     fontsize=9)
        ax.set_xlabel("in-plane shift e₂ (Å)", fontsize=8)
        ax.set_ylabel("in-plane shift e₁ (Å)", fontsize=8)
        ax.tick_params(labelsize=7)

    cb = fig.colorbar(img, ax=axes[:N_SITES].tolist(), fraction=0.02, pad=0.01,
                      extend="max")
    cb.set_label("Trapping rate 1/τ (ps⁻¹) — higher = faster transfer",
                 fontsize=10)

    # ── Panel 9: global optimisation ──
    ax = axes[8]
    ax.set_facecolor("#fffdf7")
    if opt is not None:
        # Project native + optimised positions onto the in-plane axes for display
        nat2d = (opt["pos_native"] - CENTROID) @ PLANE_AXES.T
        opt2d = (opt["pos_opt"]   - CENTROID) @ PLANE_AXES.T
        for i in range(N_SITES):
            ax.annotate("", xy=opt2d[i], xytext=nat2d[i],
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
        ax.scatter(*nat2d.T, c="#888", s=80, label="native", zorder=3,
                   edgecolor="white")
        colors = ["#ff3b3b" if i == TRAP_SITE else
                  "#1a9e4b" if i in ENTRY_SITES else "#1a6e8c"
                  for i in range(N_SITES)]
        ax.scatter(*opt2d.T, c=colors, s=90, label="optimised", zorder=4,
                   edgecolor="white")
        for i in range(N_SITES):
            ax.text(opt2d[i, 0], opt2d[i, 1], f" {i+1}", fontsize=8, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("e₁ (Å)", fontsize=8)
        ax.set_ylabel("e₂ (Å)", fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_title(
            f"Global optimum (all pigments)\n"
            f"ETE {opt['ete_native']:.4f}→{opt['ete_opt']:.4f}   "
            f"τ {opt['tau_native']/1000:.2f}→{opt['tau_opt']/1000:.2f} ps",
            fontsize=9)
    else:
        ax.text(0.5, 0.5, "global optimisation\nskipped", ha="center",
                va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Global optimum", fontsize=9)

    fig.suptitle(
        "Fig. 5: Position-vs-efficiency scan — 8-site FMO" + tag + "   "
        f"(★ native: ETE={ete_native:.4f}, 1/τ={rate_native:.3f} ps⁻¹;  "
        "✕ per-pigment fastest spot)",
        fontsize=12, y=0.995)
    out = RESULTS_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


if __name__ == "__main__":
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()                       # batched CDC runs on the GPU

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--refine", action="store_true",
                        help="High-fidelity refinement at level 1 (key points only).")
    parser.add_argument("--refine-level", type=int, choices=[1, 2, 3, 4, 5], default=0,
                        help="Refinement precision: 1=key points (<1 min); 2=coarse "
                             "heatmaps (~5 min); 3=medium (~12 min); 4=fine + finer PB "
                             "grid (~45 min); 5=ULTRA: re-optimise under PB objective + "
                             "high-res heatmaps (~5-6 h, use --refine-jobs 16). Needs openmm + apbs.")
    parser.add_argument("--refine-jobs", type=int, default=8,
                        help="Parallel APBS workers for refinement (use 16 for level 5).")
    parser.add_argument("--refine-hours", type=float, default=5.0,
                        help="Wall-clock budget (hours) for the level-5 re-optimisation.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    n_grid = 21 if args.quick else 61
    span   = 6.0

    print("=== Phase 4: Position-vs-efficiency scan (8-site FMO) ===")
    print(f"  Per-pigment grid: {n_grid}×{n_grid} over ±{span:.0f} Å in-plane")
    us, vs, ete, rate = run_scan(n_grid, span, n_jobs=args.n_jobs)

    opt = None
    if not args.no_optimize:
        print("\n  Global optimisation (all pigments, minimise trapping time) …")
        opt = optimize_arrangement(
            bound=span,
            maxiter=25 if args.quick else 80,
            popsize=12 if args.quick else 18,
        )
        print(f"    Native    : ETE={opt['ete_native']:.4f}  "
              f"τ={opt['tau_native']/1000:.2f} ps")
        print(f"    Optimised : ETE={opt['ete_opt']:.4f}  "
              f"τ={opt['tau_opt']/1000:.2f} ps  (min sep {opt['min_sep_opt']:.1f} Å)")

    # Save raw data for Phase 6
    save = dict(us=us, vs=vs, ete=ete, rate=rate)
    if opt is not None:
        save.update(pos_native=opt["pos_native"], pos_opt=opt["pos_opt"],
                    disp_inplane=opt["disp_inplane"], shift_mag=opt["shift_mag"],
                    ete_native=opt["ete_native"], ete_opt=opt["ete_opt"],
                    tau_native=opt["tau_native"], tau_opt=opt["tau_opt"])

    # ── High-fidelity refinement (OpenMM relax + APBS PB), points-only / leveled ──
    level = args.refine_level or (1 if args.refine else 0)
    refined = None
    if level and opt is not None:
        import time as _time
        from refine import (LEVELS, relaxed_protein_xyz, _native_refs, refine_ete,
                            refine_scan_pigment)
        from geometry_scan import disp_from_inplane
        grid_n, dime = LEVELS[level]
        print(f"\n  Refinement level {level} (relax protein + APBS PB, dime={dime}, "
              f"{args.refine_jobs} workers) …")
        t0 = _time.time()
        prot = relaxed_protein_xyz()            # protein relaxed around native pigments
        ps_nat, pp_nat = _native_refs(polarize=True, dime=dime, relax=True)

        # always: refine the fast model's global optimum (full relax + PB)
        disp_opt = disp_from_inplane(opt["disp_inplane"])
        ete_r, tau_r = refine_ete(disp_opt, dime=dime)
        print(f"    Fast optimum             : ETE={opt['ete_opt']:.4f}  τ={opt['tau_opt']/1000:.2f} ps")
        print(f"    Fast optimum, refined    : ETE={ete_r:.4f}  τ={tau_r/1000:.2f} ps")
        save.update(ete_opt_refined=ete_r, tau_opt_refined=tau_r)

        # level 5 ("ultra"): RE-OPTIMISE under the PB-polarized objective (the long part)
        if level >= 5:
            from refine import optimize_refined
            # reserve ~15% of the budget for heatmaps + the final full-relax refine
            opt_budget = args.refine_hours * 3600.0 * 0.85
            print(f"    Re-optimising the arrangement under the PB-polarized objective "
                  f"(dime={dime}, {args.refine_jobs} workers, budget {opt_budget/3600:.1f} h) "
                  "— this is the multi-hour step …")
            inplane_r, disp_r = optimize_refined(bound=span, dime=dime,
                                                 workers=args.refine_jobs,
                                                 time_budget_s=opt_budget)
            ete_ro, tau_ro = refine_ete(disp_r, relax=True, polarize=True, dime=dime)
            print(f"    RE-OPTIMISED optimum     : ETE={ete_ro:.4f}  τ={tau_ro/1000:.2f} ps "
                  f"(searched under high-fidelity physics)")
            save.update(disp_inplane_refined=inplane_r, pos_ropt=MG_COORDS_ANG + disp_r,
                        ete_ropt=ete_ro, tau_ropt=tau_ro)

        if grid_n:   # levels 2-5: refined per-pigment heatmaps
            ur = np.linspace(-span, span, grid_n)
            ete_ref = np.empty((N_SITES, grid_n, grid_n))
            rate_ref = np.empty((N_SITES, grid_n, grid_n))
            for p in tqdm(range(N_SITES), desc="refined scan", unit="BChl"):
                e, t = refine_scan_pigment(p, ur, ur, prot, ps_nat, pp_nat, PLANE_AXES,
                                           dime=dime, n_jobs=args.refine_jobs)
                ete_ref[p] = e
                rate_ref[p] = 1.0 / (t / 1000.0)
            save.update(ur=ur, ete_ref=ete_ref, rate_ref=rate_ref)
            refined = (ur, ete_ref, rate_ref)
        print(f"    refinement done in {(_time.time()-t0)/60:.1f} min")

    np.savez(RESULTS_DIR / "p4_position_scan.npz", **save)

    print("\nGenerating figure …")
    plot_position_scan(us, vs, ete, rate, opt)
    if refined is not None:
        ur, ete_ref, rate_ref = refined
        plot_position_scan(ur, ur, ete_ref, rate_ref, opt,
                           out_name="fig5_position_scan_refined.png",
                           tag=f" — refined L{level} (relax+APBS PB)")
    print("\nDone.")
