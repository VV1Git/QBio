"""
Phase 6: Summary figure for the 8-site FMO position-vs-efficiency study.

Six panels:
    [1] Reaction-centre yield Q(t) entering at BChl 1 vs BChl 6
    [2] Native FMO coupling matrix (8x8)
    [3] Trapping time & ETE per starting pigment
    [4] Global position optimisation: native -> optimised arrangement
    [5] Bath sensitivity: ETE & trapping time vs reorganisation energy λ
    [6] Exciton coherence spectrum (FFT of complex coherence) — Ohmic vs vibronic

Cached data from Phases 4–5 is loaded where available; dynamics and coherence
are computed fresh at the native geometry.

Usage
-----
    python phase6_summary.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import run_with_trap, compute_ete
from hamiltonian import build_electronic_H
from fmo_data import N_SITES, TRAP_SITE, ENTRY_SITES
from geometry_scan import PLANE_AXES, CENTROID

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_SITE_COLORS = plt.cm.viridis(np.linspace(0, 0.92, N_SITES))


def _panel_yield(ax):
    for s, color in zip(ENTRY_SITES, ["#1f6f78", "#c8531e"]):
        t, _, Q, L = run_with_trap(initial_site=s, t_end=30000.0, n_steps=400)
        ax.plot(t / 1000, Q, color=color, lw=2.0, label=f"start BChl {s+1}")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
    ete, tau = compute_ete(build_electronic_H())
    ax.set_xlabel("Time (ps)", fontsize=9); ax.set_ylabel("RC yield Q(t)", fontsize=9)
    ax.set_title(f"Energy → reaction centre\n(ETE={ete:.3f}, τ={tau/1000:.1f} ps)", fontsize=9)
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _panel_coupling(ax, fig):
    H = build_electronic_H()
    J = H - np.diag(np.diag(H))
    vmax = np.abs(J).max()
    im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("J (cm⁻¹)", fontsize=8)
    ax.set_xticks(range(N_SITES)); ax.set_xticklabels(range(1, N_SITES+1), fontsize=7)
    ax.set_yticks(range(N_SITES)); ax.set_yticklabels(range(1, N_SITES+1), fontsize=7)
    ax.set_title("Native FMO coupling matrix", fontsize=9)


def _panel_per_site(ax):
    H = build_electronic_H()
    taus, etes = [], []
    for s in range(N_SITES):
        e, t = compute_ete(H, initial_sites=(s,))
        taus.append(t / 1000); etes.append(e)
    edge = ["black" if (i == TRAP_SITE or i in ENTRY_SITES) else "none"
            for i in range(N_SITES)]
    ax.bar(range(1, N_SITES+1), taus, color=[_SITE_COLORS[i] for i in range(N_SITES)],
           edgecolor=edge, linewidth=1.3)
    ax.set_xlabel("starting BChl (outlined = entry/trap)", fontsize=9)
    ax.set_ylabel("trapping time (ps)", fontsize=9)
    ax.set_title("Trapping time per entry site", fontsize=9)
    ax.set_xticks(range(1, N_SITES+1)); ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _panel_optimum(ax):
    path = RESULTS_DIR / "p4_position_scan.npz"
    if not path.exists() or "pos_opt" not in np.load(path).files:
        ax.text(0.5, 0.5, "run phase4_scan.py\nfor optimisation", ha="center",
                va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Global position optimum", fontsize=9); return
    d = np.load(path)
    nat2d = (d["pos_native"] - CENTROID) @ PLANE_AXES.T
    opt2d = (d["pos_opt"] - CENTROID) @ PLANE_AXES.T
    for i in range(N_SITES):
        ax.annotate("", xy=opt2d[i], xytext=nat2d[i],
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.1))
    ax.scatter(*nat2d.T, c="#888", s=55, edgecolor="white", zorder=3, label="native")
    colors = ["#ff3b3b" if i == TRAP_SITE else "#1a9e4b" if i in ENTRY_SITES
              else "#1a6e8c" for i in range(N_SITES)]
    ax.scatter(*opt2d.T, c=colors, s=60, edgecolor="white", zorder=4, label="optimised")
    for i in range(N_SITES):
        ax.text(opt2d[i, 0], opt2d[i, 1], f" {i+1}", fontsize=7, zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("e₁ (Å)", fontsize=9); ax.set_ylabel("e₂ (Å)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(f"Position optimum: ETE {float(d['ete_native']):.3f}→{float(d['ete_opt']):.3f}, "
                 f"τ {float(d['tau_native'])/1000:.1f}→{float(d['tau_opt'])/1000:.1f} ps",
                 fontsize=8)


def _panel_lambda(ax):
    path = RESULTS_DIR / "p5_sensitivity_data.npz"
    if not path.exists():
        ax.text(0.5, 0.5, "run phase5", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("ETE vs λ", fontsize=9); return
    d = np.load(path)
    lam, ete, tau = d["lambda_grid"], d["ete_lam"], d["tau_lam"]
    ax.plot(lam, ete, color="#1a6e8c", lw=2.0)
    ax.set_xlabel("λ (cm⁻¹)", fontsize=9); ax.set_ylabel("ETE", color="#1a6e8c", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#1a6e8c")
    ax.axvline(35.0, color="gray", ls=":", lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(lam, tau, color="#d45f1e", lw=2.0, ls="--")
    ax2.set_ylabel("τ (ps)", color="#d45f1e", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#d45f1e"); ax2.spines["top"].set_visible(False)
    ax.set_title("Efficiency vs reorganisation λ", fontsize=9)
    ax.spines["top"].set_visible(False); ax.grid(True, alpha=0.18, ls="--")


def _panel_coherence(ax):
    from analysis import _exciton_coherences, _coherence_spectrum
    from dynamics import run_ohmic
    from vibronic import run_structured, OMEGA1_CM, OMEGA2_CM
    t_A, rhos_A = run_ohmic(initial_site=ENTRY_SITES[0], t_end=5000.0, n_steps=600)
    eig, ex_A = _exciton_coherences(rhos_A, is_vibronic=False)
    t_B, rhos_B = run_structured(initial_site=ENTRY_SITES[0], t_end=5000.0, n_steps=600)
    _, ex_B = _exciton_coherences(rhos_B, is_vibronic=True)

    for ex, color, label in [(ex_A, "#1a6e8c", "Ohmic"), (ex_B, "#d45f1e", "Vibronic")]:
        f, p = _coherence_spectrum(t_A, ex, 100.0, 3000.0)
        pos = f >= 0; f, p = f[pos], p[pos]
        if p.max() > 0: p = p / p.max()
        m = f <= 1000
        ax.plot(f[m], p[m], color=color, lw=1.4, ls="--" if label == "Vibronic" else "-",
                label=label)
    for omega in (OMEGA1_CM, OMEGA2_CM):
        ax.axvline(omega, color="#a33", ls=":", lw=0.9, alpha=0.6)
        ax.text(omega + 6, 0.85, f"{omega:.0f}", fontsize=7, color="#a33")
    ax.set_xlabel("Frequency (cm⁻¹)", fontsize=9); ax.set_ylabel("Power (norm.)", fontsize=9)
    ax.set_title("Exciton coherence spectrum", fontsize=9)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def build_summary() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.4)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    print("  [1/6] RC yield …");        _panel_yield(axes[0])
    print("  [2/6] coupling matrix …"); _panel_coupling(axes[1], fig)
    print("  [3/6] per-site trapping …"); _panel_per_site(axes[2])
    print("  [4/6] position optimum …"); _panel_optimum(axes[3])
    print("  [5/6] bath sensitivity …"); _panel_lambda(axes[4])
    print("  [6/6] coherence spectrum …"); _panel_coherence(axes[5])

    fig.suptitle("Summary: 8-site FMO — position-dependent excitation transfer to the reaction centre",
                 fontsize=13, y=0.98)
    out = RESULTS_DIR / "fig7_summary.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    from gpu_utils import setup_gpu
    setup_gpu()
    print("=== Phase 6: Summary figure ===")
    build_summary()
    print("\nDone.")
