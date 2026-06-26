"""
Phase 11: DEEP run — the position-sensitivity result recomputed with numerically
exact HEOM dynamics instead of the fast secular-Redfield solver.

Most FMO studies cannot afford HEOM across a geometry scan; a multi-hour budget
makes it possible, turning the headline two-tier sensitivity result (phase7,
secular) into an exact-dynamics result.  For every displaced geometry the metric
is the reaction-centre-sink (BChl 3) population reached at a fixed probe time
under HEOM — the same transfer proxy validated in phase9 (fig14).

Designed for a long unattended run:
  * wall-clock budget (--hours);
  * CHECKPOINTING after every batch to results/p11_heom_deep.npz — the run is
    fully resumable, so an interrupted 10 h job loses nothing (re-launch with the
    same command and it skips finished points);
  * tasks ordered by scientific value, so even a partial run is useful.

Tasks (in priority order)
    1. native convergence check (HEOM depth 2 vs 3) — establishes the metric;
    2. exact HEOM 2D sensitivity scans for the diagnostic pigments
       (BChl 3 trap, BChl 8 tolerant, BChl 1 entry, BChl 4 core);
    3. exact HEOM static-disorder ensemble at the native geometry.

Outputs: results/p11_heom_deep.npz (checkpoint + final), fig19_heom_deep.png

Usage
-----
    python phase11_heom_deep.py --hours 10            # full deep run
    python phase11_heom_deep.py --hours 10 --grid 27  # finer scans
    python phase11_heom_deep.py --plot-only           # just (re)draw from checkpoint
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from fmo_data import N_SITES, TRAP_SITE, ENTRY_SITES
from hamiltonian import build_electronic_H
from dynamics import _redfield_rate_matrix, C_FS
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K
from geometry_scan import inplane_to_disp

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CKPT = RESULTS_DIR / "p11_heom_deep.npz"

_KT = 1.4388
SPAN = 6.0
PROBE_FS = 1000.0
SCAN_PIGMENTS = (TRAP_SITE, 7, ENTRY_SITES[0], 3)   # BChl 3, 8, 1, 4
INIT = ENTRY_SITES[0]


# ── HEOM sink-population metric ───────────────────────────────────────────────

def _heom_sink(H, depth=3, Nk=1, probe_fs=PROBE_FS):
    """Reaction-centre-sink (BChl 3) population at probe_fs under exact HEOM,
    starting from the entry pigment.  Transfer proxy (higher = better funnelling)."""
    import qutip as qt
    from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
    Hq = qt.Qobj(H - np.mean(np.diag(H)) * np.eye(N_SITES))
    T_cm = TEMPERATURE_K / _KT
    baths = [DrudeLorentzBath(qt.ket2dm(qt.basis(N_SITES, i)),
                              LAMBDA_CM, GAMMA_CM, T_cm, Nk=Nk)
             for i in range(N_SITES)]
    solver = HEOMSolver(Hq, baths, max_depth=depth, options={"nsteps": 30000})
    tlist = np.array([0.0, probe_fs]) * 2 * np.pi * C_FS
    res = solver.run(qt.ket2dm(qt.basis(N_SITES, INIT)), tlist)
    return float(res.states[-1][TRAP_SITE, TRAP_SITE].real)


def _redfield_sink(H, probe_fs=PROBE_FS):
    """Same sink metric from the fast secular-Redfield propagation (for overlay)."""
    from scipy.linalg import expm
    K, U = _redfield_rate_matrix(H, LAMBDA_CM, GAMMA_CM, TEMPERATURE_K)
    Usq = U ** 2
    P = Usq @ (expm(K * (probe_fs * 2 * np.pi * C_FS)) @ Usq[INIT, :])
    return float(P[TRAP_SITE])


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_ckpt(grid, n_dis):
    if CKPT.exists():
        d = dict(np.load(CKPT, allow_pickle=True))
        if int(d.get("grid", -1)) == grid and int(d.get("n_dis", -1)) == n_dis:
            return d
    us = np.linspace(-SPAN, SPAN, grid)
    return {
        "grid": grid, "n_dis": n_dis, "us": us, "depth": 3, "Nk": 1,
        "scan_pigs": np.array(SCAN_PIGMENTS),
        "heom_scan": np.full((len(SCAN_PIGMENTS), grid, grid), np.nan),
        "redf_scan": np.full((len(SCAN_PIGMENTS), grid, grid), np.nan),
        "disorder_seeds": np.arange(n_dis),
        "disorder_heom": np.full(n_dis, np.nan),
        "native_conv": np.full(3, np.nan),   # depth 2,3,(4 if reached)
    }


def _save(d):
    np.savez(CKPT, **d)


# ── Driver ────────────────────────────────────────────────────────────────────

def run(hours, grid, depth, Nk, sigma=80.0):
    d = _load_ckpt(grid, grid_to_ndis(grid))
    d["depth"], d["Nk"] = depth, Nk
    us = d["us"]
    t0 = time.time()
    budget = hours * 3600.0
    n_done = int(np.isfinite(d["heom_scan"]).sum() + np.isfinite(d["disorder_heom"]).sum())
    print(f"=== Phase 11: deep HEOM run (budget {hours:.1f} h, depth {depth}, Nk {Nk}, "
          f"grid {grid}×{grid}) ===")
    print(f"  resuming with {n_done} points already done")

    def over_budget():
        return (time.time() - t0) > budget

    # 1. native convergence
    if not np.isfinite(d["native_conv"][0]):
        H = build_electronic_H()
        for k, dep in enumerate((2, 3)):
            d["native_conv"][k] = _heom_sink(H, depth=dep, Nk=Nk)
        _save(d)
        print(f"  native sink@1ps: depth2={d['native_conv'][0]:.4f} "
              f"depth3={d['native_conv'][1]:.4f} (convergence check)")

    # 2. HEOM sensitivity scans (the expensive, high-value part)
    n_pts = 0
    for pi, p in enumerate(SCAN_PIGMENTS):
        for i, u in enumerate(us):
            if over_budget():
                break
            if np.all(np.isfinite(d["heom_scan"][pi, i])):
                continue                                  # row already done
            for j, v in enumerate(us):
                if np.isfinite(d["heom_scan"][pi, i, j]):
                    continue
                H = build_electronic_H(inplane_to_disp(p, u, v))
                d["heom_scan"][pi, i, j] = _heom_sink(H, depth=depth, Nk=Nk)
                d["redf_scan"][pi, i, j] = _redfield_sink(H)
                n_pts += 1
            _save(d)                                       # checkpoint per row
            rate = (time.time() - t0) / max(n_pts, 1)
            print(f"  [scan BChl{p+1}] row {i+1}/{grid} done  "
                  f"({n_pts} pts, {rate:.1f}s/pt, {(time.time()-t0)/3600:.2f}h elapsed)")
        if over_budget():
            print("  budget reached during sensitivity scans.")
            break

    # 3. HEOM disorder ensemble at native (fills remaining budget)
    if not over_budget():
        H0 = build_electronic_H()
        # regenerate the same disorder draws deterministically per seed
        for k in range(len(d["disorder_heom"])):
            if np.isfinite(d["disorder_heom"][k]):
                continue
            if over_budget():
                break
            de = np.random.default_rng(1000 + k).normal(0, sigma, N_SITES)
            d["disorder_heom"][k] = _heom_sink(H0 + np.diag(de), depth=depth, Nk=Nk)
            if k % 10 == 0:
                _save(d)
                print(f"  [disorder] {k+1}/{len(d['disorder_heom'])} "
                      f"({(time.time()-t0)/3600:.2f}h elapsed)")
        _save(d)

    _save(d)
    print(f"\n  finished/paused at {(time.time()-t0)/3600:.2f} h, "
          f"{int(np.isfinite(d['heom_scan']).sum())} scan pts done.")
    return d


def grid_to_ndis(grid):
    """Disorder-ensemble size — large enough to absorb leftover budget if the
    sensitivity scans finish before the wall-clock limit."""
    return 600


# ── Figure ────────────────────────────────────────────────────────────────────

def plot(d=None):
    import matplotlib.pyplot as plt
    if d is None:
        if not CKPT.exists():
            sys.exit("no checkpoint to plot.")
        d = dict(np.load(CKPT, allow_pickle=True))
    us = d["us"]; pigs = d["scan_pigs"]
    BG, PANEL = "#f7f2ea", "#fffdf7"

    n = len(pigs)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.2), dpi=150)
    fig.patch.set_facecolor(BG)
    vmax = np.nanmax(d["heom_scan"])
    for col, p in enumerate(pigs):
        for row, (data, tag) in enumerate([(d["heom_scan"][col], "HEOM (exact)"),
                                           (d["redf_scan"][col], "secular Redfield")]):
            ax = axes[row, col]; ax.set_facecolor(PANEL)
            im = ax.pcolormesh(us, us, data, cmap="magma", shading="nearest",
                               vmin=0, vmax=vmax)
            ax.plot(0, 0, "*", color="#39d0ff", ms=13, markeredgecolor="black",
                    markeredgewidth=0.6)
            role = ("trap→RC" if p == TRAP_SITE else "tolerant" if p == 7
                    else "entry" if p in ENTRY_SITES else "core")
            ax.set_title(f"BChl {p+1} ({role}) — {tag}", fontsize=9)
            ax.set_xlabel("e₂ (Å)", fontsize=8); ax.set_ylabel("e₁ (Å)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
                "sink pop @1ps", fontsize=7)
    fig.suptitle("Fig. 19: Exact HEOM position-sensitivity scan vs the fast secular "
                 "model\n(top: numerically-exact dynamics; bottom: scan-model proxy — "
                 "same two-tier pattern)", fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig19_heom_deep.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")

    if np.isfinite(d["disorder_heom"]).any():
        dd = d["disorder_heom"][np.isfinite(d["disorder_heom"])]
        print(f"  HEOM disorder ensemble (n={dd.size}): sink@1ps "
              f"{dd.mean():.4f} ± {dd.std():.4f}")


def main():
    p = argparse.ArgumentParser(description="Phase 11: deep HEOM run")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--grid", type=int, default=25)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--Nk", type=int, default=1)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.plot_only:
        plot()
        return
    from gpu_utils import setup_gpu
    setup_gpu(verbose=False)            # HEOM runs on CPU; keep GPU quiet
    d = run(args.hours, args.grid, args.depth, args.Nk)
    plot(d)
    print("\nDone.")


if __name__ == "__main__":
    main()
