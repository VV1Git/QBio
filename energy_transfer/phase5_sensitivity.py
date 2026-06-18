"""
Phase 5: Bath-parameter sensitivity of FMO transfer efficiency (native geometry).

How robust is the excitation-transfer efficiency of the real FMO complex to its
environment?  At the fixed native geometry we sweep the three Ohmic-bath
parameters and record both the trapping yield (ETE) and the trapping time:

    (a) reorganisation energy λ :   5 – 150 cm⁻¹   (default 35 cm⁻¹)
    (b) temperature T           : 100 – 500 K       (default 300 K)
    (c) bath cut-off γ          :  10 – 200 cm⁻¹   (default 53 cm⁻¹)

Efficiency comes from the fast secular-Redfield solver (dynamics.compute_ete),
so every curve is computed in well under a second.

Output: fig6_bath_sensitivity.png, p5_sensitivity_data.npz

Usage
-----
    python phase5_sensitivity.py
    python phase5_sensitivity.py --quick
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import compute_ete
from hamiltonian import build_electronic_H
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_FULL, N_QUICK = 60, 15
LAMBDA_RANGE = (5.0, 150.0)
TEMP_RANGE   = (100.0, 500.0)
GAMMA_RANGE  = (10.0, 200.0)

_H_NATIVE = build_electronic_H()


def _sweep(param: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ETE, tau_ps) arrays as one bath parameter varies."""
    ete, tau = [], []
    for v in values:
        kw = dict(lambda_=LAMBDA_CM, gamma_bath=GAMMA_CM, temperature=TEMPERATURE_K)
        kw[{"lambda": "lambda_", "temperature": "temperature",
            "gamma": "gamma_bath"}[param]] = v
        e, t = compute_ete(_H_NATIVE, **kw)
        ete.append(e); tau.append(t / 1000.0)
    return np.array(ete), np.array(tau)


def plot_sensitivity(lam, ete_l, tau_l, tmp, ete_t, tau_t,
                     gam, ete_g, tau_g) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")

    panels = [
        (axes[0], lam, ete_l, tau_l, "Reorganisation energy λ (cm⁻¹)", LAMBDA_CM),
        (axes[1], tmp, ete_t, tau_t, "Temperature T (K)", TEMPERATURE_K),
        (axes[2], gam, ete_g, tau_g, "Bath cut-off γ (cm⁻¹)", GAMMA_CM),
    ]
    for ax, x, ete, tau, xlabel, default in panels:
        ax.set_facecolor("#fffdf7")
        l1, = ax.plot(x, ete, color="#1a6e8c", lw=2.2, label="ETE (yield)")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("ETE", color="#1a6e8c", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#1a6e8c")
        ax.set_ylim(min(0.9, ete.min() - 0.01), 1.002)
        ax.axvline(default, color="gray", ls=":", lw=1.0)

        ax2 = ax.twinx()
        l2, = ax2.plot(x, tau, color="#d45f1e", lw=2.2, ls="--", label="trapping time")
        ax2.set_ylabel("trapping time (ps)", color="#d45f1e", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#d45f1e")
        ax2.spines["top"].set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.grid(True, alpha=0.18, ls="--")
        if ax is axes[0]:
            ax.legend([l1, l2], ["ETE (yield)", "trapping time"],
                      frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Fig. 6: FMO efficiency vs bath parameters — native geometry "
                 "(dotted = physiological default)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "fig6_bath_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n = N_QUICK if args.quick else N_FULL
    lam = np.linspace(*LAMBDA_RANGE, n)
    tmp = np.linspace(*TEMP_RANGE, n)
    gam = np.linspace(*GAMMA_RANGE, n)

    print("=== Phase 5: Bath sensitivity (native FMO geometry) ===")
    ete_l, tau_l = _sweep("lambda", lam)
    ete_t, tau_t = _sweep("temperature", tmp)
    ete_g, tau_g = _sweep("gamma", gam)

    np.savez(RESULTS_DIR / "p5_sensitivity_data.npz",
             lambda_grid=lam, ete_lam=ete_l, tau_lam=tau_l,
             temp_grid=tmp,   ete_tmp=ete_t, tau_tmp=tau_t,
             gamma_grid=gam,  ete_gam=ete_g, tau_gam=tau_g)
    print("  Saved p5_sensitivity_data.npz")

    plot_sensitivity(lam, ete_l, tau_l, tmp, ete_t, tau_t, gam, ete_g, tau_g)
    print(f"\n  ETE range: λ[{ete_l.min():.3f},{ete_l.max():.3f}] "
          f"T[{ete_t.min():.3f},{ete_t.max():.3f}] "
          f"γ[{ete_g.min():.3f},{ete_g.max():.3f}]")
    print("\nDone.")
