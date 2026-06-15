"""
Full FMO realistic simulation: ETE, site populations, coherence spectra.

Two solver modes:
    --secular   (default) : Pauli master equation — ms per case, ETE only
    --full                : Bloch-Redfield brmesolve — minutes per case,
                            full density matrix, coherence spectra, GPU-ready

Four cases:
    T=300 K, BChl 1 init
    T=300 K, BChl 6 init
    T= 77 K, BChl 1 init
    T= 77 K, BChl 6 init

GPU acceleration (for --full mode):
    1. CuPy (spectral density tables):  auto-detected, no changes needed.
    2. JAX/CUDA (ODE integration):
           pip install qutip-jax "jax[cuda12_pip]" -f <jax-cuda-url>
           Then uncomment the JAX activation block below.

Usage
-----
    python run_simulation.py                 # secular, all cases, fast
    python run_simulation.py --full          # brmesolve, full density matrix
    python run_simulation.py --T 77          # only 77 K cases
    python run_simulation.py --full --T 300  # brmesolve, 300 K only
"""

from __future__ import annotations

import sys
import warnings
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eigh as _eigh

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

# ── Optional JAX/CUDA backend for QuTiP (uncomment on GPU machine) ─────────────
# import qutip_jax
# import qutip as qt
# qt.settings.core["default_dtype"] = "jaxdia"
# ──────────────────────────────────────────────────────────────────────────────

from dynamics import (
    run_fmo, run_fmo_secular,
    site_population, ground_population, compute_ete,
    TRAP_SITE, C_FS,
)
from spectral_density import build_gamma_tables
from hamiltonian import build_H_shifted, N_SITES

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

SITE_COLORS = plt.cm.tab10(np.linspace(0, 1, 8))


# ── Population figure (secular mode) ──────────────────────────────────────────

def plot_populations_secular(times, P_site, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea"); ax.set_facecolor("#fffdf7")
    for m in range(N_SITES):
        lw = 2.4 if m == TRAP_SITE - 1 else 1.5
        ax.plot(times, P_site[:, m], color=SITE_COLORS[m], lw=lw,
                label=f"BChl {m+1}{'  ←trap' if m==TRAP_SITE-1 else ''}")
    ax.set_xlabel("Time (fs)", fontsize=10)
    ax.set_ylabel("Site population", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, times[-1]); ax.set_ylim(0, None)
    ax.legend(ncol=4, frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ── Population figure (full brmesolve mode) ───────────────────────────────────

def plot_populations_full(times, rhos, title, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=150, sharex=True)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes: ax.set_facecolor("#fffdf7")

    for m in range(1, N_SITES + 1):
        P = site_population(rhos, m)
        axes[0].plot(times, P, color=SITE_COLORS[m-1], lw=1.6,
                     label=f"BChl {m}{'  ←trap' if m==TRAP_SITE else ''}")
    axes[0].set_ylabel("Site population"); axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[0].set_title(title, fontsize=11)
    axes[0].grid(True, alpha=0.18, ls="--")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].plot(times, ground_population(rhos), "k--", lw=1.8, label="Ground")
    axes[1].plot(times, site_population(rhos, TRAP_SITE),
                 color=SITE_COLORS[TRAP_SITE-1], lw=2.2, label=f"BChl {TRAP_SITE} (trap)")
    axes[1].set_xlabel("Time (fs)"); axes[1].set_ylabel("Population")
    axes[1].legend(frameon=False); axes[1].grid(True, alpha=0.18, ls="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ── ETE bar chart ──────────────────────────────────────────────────────────────

def plot_ete(labels, etes, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#f7f2ea"); ax.set_facecolor("#fffdf7")
    colors = ["#1a6e8c", "#2fa37c", "#d45f1e", "#b5410d"]
    bars = ax.bar(labels, [e * 100 for e in etes], color=colors[:len(etes)],
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, etes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("ETE η (%)"); ax.set_ylim(0, 105)
    ax.set_title("FMO Energy Transfer Efficiency — site-specific structured bath")
    ax.axhline(100, color="gray", ls=":", lw=1.0)
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ── Coherence spectra (full mode only) ────────────────────────────────────────

def plot_coherence_spectra(times, rhos, title, out_path,
                           window_start=100.0, window_end=3000.0):
    H_np = build_H_shifted()
    _, U = _eigh(H_np)
    mask = (times >= window_start) & (times <= window_end)
    t_w  = times[mask]
    if t_w.size < 8:
        return

    coh_sum = np.zeros(t_w.size)
    for k, rho_idx in enumerate(np.where(mask)[0]):
        rho_site = np.array(rhos[rho_idx].full())[1:, 1:]
        rho_ex   = U.conj().T @ rho_site @ U
        for a in range(N_SITES):
            for b in range(a + 1, N_SITES):
                coh_sum[k] += abs(rho_ex[a, b])

    dt   = np.mean(np.diff(t_w))
    freq = np.fft.rfftfreq(t_w.size, d=dt) / C_FS
    power = np.abs(np.fft.rfft(coh_sum))**2
    mask_f = freq <= 900.0

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor("#f7f2ea"); ax.set_facecolor("#fffdf7")
    ax.plot(freq[mask_f], power[mask_f] / max(power[mask_f].max(), 1e-30),
            color="#1a6e8c", lw=1.5)
    for omega_j, _ in [(46,0),(68,0),(180,0),(243,0),(291,0),(366,0),(770,0)]:
        if omega_j <= 900:
            ax.axvline(omega_j, color="gray", ls=":", lw=0.8, alpha=0.6)
            ax.text(omega_j + 5, 0.85, str(omega_j), fontsize=7, color="gray")
    ax.set_xlabel("Frequency (cm⁻¹)"); ax.set_ylabel("Power (norm.)")
    ax.set_title(title); ax.set_xlim(0, 900); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true",
                        help="Use full brmesolve (slow, full density matrix)")
    parser.add_argument("--T",      type=int, choices=[77, 300], default=None)
    parser.add_argument("--t-end",  type=float, default=15_000.0)
    parser.add_argument("--steps",  type=int,   default=600)
    args = parser.parse_args()

    temps = [300] if args.T == 300 else ([77] if args.T == 77 else [300, 77])
    inits = [1, 6]
    cases = [(T, init) for T in temps for init in inits]

    mode = "brmesolve (full)" if args.full else "secular Pauli (fast)"
    print(f"=== FMO Realistic Simulation  [{mode}] ===")

    table_cache: dict[int, dict] = {}
    for T in temps:
        print(f"\nBuilding spectral density tables for T={T} K …")
        table_cache[T] = build_gamma_tables(temperature=float(T))

    labels, etes = [], []

    for T, init in cases:
        label = f"T={T}K / BChl {init}"
        print(f"\n[{label}]")
        t0 = time.time()

        if args.full:
            times, rhos = run_fmo(
                init_site=init, temperature=float(T),
                t_end_fs=args.t_end, n_steps=args.steps,
                gamma_tables=table_cache[T],
            )
            ete = compute_ete(times, rhos)
            print(f"  Solved in {time.time()-t0:.1f}s   ETE = {ete*100:.2f}%")

            tag = f"{T}K_BChl{init}"
            plot_populations_full(
                times, rhos,
                title=f"FMO populations — {label}",
                out_path=RESULTS / f"fig_populations_{tag}.png",
            )
            if T == 300 and init == 1:
                plot_coherence_spectra(
                    times, rhos,
                    title=f"Exciton coherence spectrum — {label}",
                    out_path=RESULTS / "fig_coherence_300K_BChl1.png",
                    window_start=50.0, window_end=min(3000.0, args.t_end * 0.6),
                )
        else:
            times, P_site, ete = run_fmo_secular(
                init_site=init, temperature=float(T),
                t_end_fs=args.t_end, n_steps=args.steps,
                gamma_tables=table_cache[T],
            )
            print(f"  Solved in {time.time()-t0:.3f}s   ETE = {ete*100:.2f}%")

            tag = f"{T}K_BChl{init}"
            plot_populations_secular(
                times, P_site,
                title=f"FMO populations (secular) — {label}",
                out_path=RESULTS / f"fig_populations_{tag}.png",
            )

        labels.append(label)
        etes.append(ete)

    if len(etes) > 0:
        plot_ete(labels, etes, RESULTS / "fig_ete_comparison.png")

    np.savez(RESULTS / "fmo_results.npz", labels=np.array(labels), etes=np.array(etes))
    print(f"\nAll results saved to {RESULTS}/")
    print("\n=== Summary ===")
    for lbl, ete in zip(labels, etes):
        print(f"  {lbl}: ETE = {ete*100:.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
