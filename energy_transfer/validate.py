"""
Phase 1: the 8-site FMO model — reference Hamiltonian and energy-funnel dynamics.

Two figures:
    fig1_funnel_dynamics.png  — site populations P_i(t) with reaction-centre
                                trapping at BChl 3, plus the cumulative RC yield
                                Q(t) for excitation entering at BChl 1 vs BChl 6.
    fig2_fmo_hamiltonian.png  — the native FMO 8x8 Hamiltonian (heatmap), the
                                exciton energy ladder, and the trapping time per
                                starting pigment.

Validation against the literature is built into fmo_data.validate_hamiltonian()
(point-dipole couplings reproduce the published TrEsp Hamiltonian to ~9 cm^-1
RMS) and the trapping numbers (ETE ~ 0.99, trapping time ~ few ps) match the
known high efficiency of FMO.

Usage
-----
    python validate.py
    python validate.py --quick
    python validate.py --fig1-only / --fig2-only
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dynamics import run_with_trap, compute_ete, K_TRAP_FS
from hamiltonian import build_electronic_H
from fmo_data import (
    SITE_ENERGIES_CM, N_SITES, TRAP_SITE, ENTRY_SITES, validate_hamiltonian,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_SITE_COLORS = plt.cm.viridis(np.linspace(0, 0.92, N_SITES))


def unit_check() -> None:
    """Print the literature self-checks."""
    print(f"  Point-dipole vs published couplings: RMS = {validate_hamiltonian():.2f} cm⁻¹")
    H = build_electronic_H()
    eig = np.linalg.eigvalsh(H)
    print(f"  Exciton band span: {eig[-1]-eig[0]:.0f} cm⁻¹ (lowest exciton {eig[0]:.0f})")
    ete, tau = compute_ete(H)
    print(f"  Native ETE (entry-averaged): {ete:.4f},  trapping time {tau/1000:.2f} ps")


# ── Figure 1: energy-funnel dynamics ──────────────────────────────────────────

def figure1_funnel(t_end: float = 15_000.0, n_steps: int = 500) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    # Left: full site-population trajectory starting at BChl 1
    t0 = time.time()
    times, P_sites, Q, L = run_with_trap(initial_site=ENTRY_SITES[0],
                                         t_end=t_end, n_steps=n_steps)
    print(f"  trajectory (start BChl {ENTRY_SITES[0]+1}): {time.time()-t0:.1f}s "
          f"Q(final)={Q[-1]:.3f}")
    ax = axes[0]
    for i in range(N_SITES):
        lw = 2.4 if i in (ENTRY_SITES[0], TRAP_SITE) else 1.1
        ax.plot(times / 1000, P_sites[:, i], color=_SITE_COLORS[i], lw=lw,
                label=f"BChl {i+1}" + (" (entry)" if i == ENTRY_SITES[0]
                                       else " (→RC)" if i == TRAP_SITE else ""))
    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("Site population", fontsize=11)
    ax.set_title(f"Site populations (start BChl {ENTRY_SITES[0]+1})", fontsize=11)
    ax.set_xlim(0, t_end / 1000)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: cumulative RC yield for each entry pigment
    ax = axes[1]
    for s, color in zip(ENTRY_SITES, ["#1f6f78", "#c8531e"]):
        times, _, Q, L = run_with_trap(initial_site=s, t_end=t_end, n_steps=n_steps)
        ax.plot(times / 1000, Q, color=color, lw=2.4, label=f"RC yield, start BChl {s+1}")
        ax.plot(times / 1000, L, color=color, lw=1.2, ls=":",
                label=f"lost, start BChl {s+1}")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("Population", fontsize=11)
    ax.set_title("Reaction-centre yield Q(t) and loss", fontsize=11)
    ax.set_xlim(0, t_end / 1000)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(True, alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Fig. 1: FMO energy funnel to the reaction centre "
                 f"(trap at BChl {TRAP_SITE+1}, κ={K_TRAP_FS*1e3:.0f} ps⁻¹)",
                 fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig1_funnel_dynamics.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Figure 2: native FMO Hamiltonian & exciton structure ──────────────────────

def figure2_hamiltonian() -> None:
    H = build_electronic_H()
    eigvals, U = np.linalg.eigh(H)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    # (a) Hamiltonian heatmap (couplings; diagonal blanked to show off-diagonal scale)
    J = H - np.diag(np.diag(H))
    ax = axes[0]
    vmax = np.abs(J).max()
    im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("coupling J (cm⁻¹)", fontsize=9)
    ax.set_xticks(range(N_SITES)); ax.set_xticklabels(range(1, N_SITES+1), fontsize=8)
    ax.set_yticks(range(N_SITES)); ax.set_yticklabels(range(1, N_SITES+1), fontsize=8)
    for i in range(N_SITES):
        ax.text(i, i, f"{SITE_ENERGIES_CM[i]-12000:.0f}", ha="center", va="center",
                fontsize=7, color="#333")
    ax.set_title("Coupling matrix (diag = site energy − 12000 cm⁻¹)", fontsize=10)
    ax.set_xlabel("BChl"); ax.set_ylabel("BChl")

    # (b) Exciton energy ladder, coloured by dominant site
    ax = axes[1]
    for a in range(N_SITES):
        dom = int(np.argmax(U[:, a] ** 2))
        ax.hlines(eigvals[a], 0, 1, color=_SITE_COLORS[dom], lw=2.5)
        ax.text(1.02, eigvals[a], f"|{a+1}⟩ ~BChl{dom+1}", va="center", fontsize=7)
    ax.set_xlim(0, 1.6); ax.set_xticks([])
    ax.set_ylabel("Exciton energy (cm⁻¹)", fontsize=10)
    ax.set_title("Exciton energy ladder", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # (c) Trapping time per starting pigment
    ax = axes[2]
    taus, etes = [], []
    for s in range(N_SITES):
        e, tau = compute_ete(H, initial_sites=(s,))
        taus.append(tau / 1000); etes.append(e)
    edge = ["black" if (i == TRAP_SITE or i in ENTRY_SITES) else "none"
            for i in range(N_SITES)]
    bars = ax.bar(range(1, N_SITES+1), taus,
                  color=[_SITE_COLORS[i] for i in range(N_SITES)],
                  edgecolor=edge, linewidth=1.4)
    for b, e in zip(bars, etes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f"{e:.3f}", ha="center", fontsize=6.5)
    ax.set_xlabel("starting BChl  (outlined = entry/trap)", fontsize=10)
    ax.set_ylabel("trapping time (ps)", fontsize=10)
    ax.set_title("Trapping time & ETE per entry site", fontsize=10)
    ax.set_xticks(range(1, N_SITES+1))
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 2: Native 8-site FMO Hamiltonian and exciton structure "
                 "(Klinger et al. 2020 / Schmidt am Busch et al. 2011)", fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig2_fmo_hamiltonian.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")

    np.save(RESULTS_DIR / "fmo_hamiltonian.npy", H)


if __name__ == "__main__":
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()

    parser = argparse.ArgumentParser(description="Phase 1: 8-site FMO model")
    parser.add_argument("--fig1-only", action="store_true")
    parser.add_argument("--fig2-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("=== Unit check: FMO Hamiltonian ===")
    unit_check()

    t_end_1 = 15_000.0 if args.quick else 30_000.0

    if not args.fig2_only:
        print("\n=== Figure 1: energy-funnel dynamics ===")
        figure1_funnel(t_end=t_end_1, n_steps=300 if args.quick else 600)
    if not args.fig1_only:
        print("\n=== Figure 2: FMO Hamiltonian & exciton structure ===")
        figure2_hamiltonian()

    print("\nDone.")
