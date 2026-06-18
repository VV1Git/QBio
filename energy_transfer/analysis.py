"""
Phase 3: Ohmic vs vibronic comparison and coherence spectroscopy (8-site FMO).

Produces two figures at the native FMO geometry:
    fig3_ohmic_vs_vibronic.png — site populations P_i(t) under the structureless
                                 Ohmic bath vs the structured (vibronic) bath,
                                 starting from entry pigment BChl 1.
    fig4_coherence_spectra.png — power spectra of the exciton-basis coherences.

Fig. 4 fix
----------
The spectrum is now the FFT of the *complex* coherence rho_{a,b}(t) (mean
removed), NOT of |rho_{a,b}(t)|.  Taking the modulus rectifies the oscillation
and collapses all power to zero frequency, which is why the old figure showed
only a DC spike.  The complex coherence oscillates at the exciton energy gap
omega_{ab}, so its FFT correctly reveals peaks at the exciton gaps (Ohmic) and
additional vibronic structure near the 770 / 243 cm^-1 modes (vibronic).

Usage
-----
    python analysis.py                 # both figures
    python analysis.py --quick         # shorter propagation
    python analysis.py --fig3-only / --fig4-only
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh as _eigh
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from dynamics import run_ohmic, population
from vibronic import run_structured, population_vibronic, OMEGA1_CM, OMEGA2_CM
from hamiltonian import build_electronic_H
from fmo_data import SITE_ENERGIES_CM, N_SITES, ENTRY_SITES, TRAP_SITE

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_FS = 3e-5   # cm/fs

_SITE_COLORS = plt.cm.viridis(np.linspace(0, 0.92, N_SITES))


# ── Figure 3: population dynamics, Ohmic vs vibronic ──────────────────────────

def figure3_dynamics(t_end: float = 5000.0, n_steps: int = 500,
                     initial_site: int = ENTRY_SITES[0]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150, sharey=True)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    with tqdm(total=2, desc="Fig 3", unit="solve") as bar:
        bar.set_description("Fig 3  Ohmic")
        t0 = time.time()
        t_A, rhos_A = run_ohmic(initial_site=initial_site, t_end=t_end, n_steps=n_steps)
        P_A = np.array([population(rhos_A, i) for i in range(N_SITES)])
        tqdm.write(f"  Ohmic: {time.time()-t0:.1f}s"); bar.update(1)

        bar.set_description("Fig 3  Vibronic")
        t0 = time.time()
        t_B, rhos_B = run_structured(initial_site=initial_site, t_end=t_end, n_steps=n_steps)
        P_B = np.array([population_vibronic(rhos_B, i) for i in range(N_SITES)])
        tqdm.write(f"  Vibronic: {time.time()-t0:.1f}s"); bar.update(1)

    for ax, t, P, title in [(axes[0], t_A, P_A, "Ohmic bath"),
                            (axes[1], t_B, P_B, "Vibronic bath")]:
        for i in range(N_SITES):
            lw = 2.4 if i in (initial_site, TRAP_SITE) else 1.1
            ax.plot(t / 1000, P[i], color=_SITE_COLORS[i], lw=lw,
                    label=f"BChl {i+1}" + (" (entry)" if i == initial_site
                                           else " (→RC)" if i == TRAP_SITE else ""))
        ax.set_xlabel("Time (ps)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, t_end / 1000)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Site population", fontsize=11)
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Fig. 3: FMO site-population dynamics — Ohmic vs vibronic bath "
                 f"(start BChl {initial_site+1}, native geometry)", fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig3_ohmic_vs_vibronic.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Figure 4: exciton-basis coherence spectra (FIXED) ─────────────────────────

def _exciton_coherences(rhos, is_vibronic=False):
    """Return (eigvals, rho_ex[T,8,8]) in the exciton basis at native geometry."""
    H = build_electronic_H()
    H -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    eigvals, U = _eigh(H)
    rhos_np = np.array([np.array((r.ptrace([0]) if is_vibronic else r).full())
                        for r in rhos])
    rho_ex = np.einsum("ij,kjl,lm->kim", U.conj().T, rhos_np, U)
    return eigvals, rho_ex


def _coherence_spectrum(times, rho_ex, window_start, window_end):
    """
    FFT of the *complex* off-diagonal coherences (mean removed), summed in
    power over all exciton pairs.  Peaks appear at the exciton gaps.
    """
    mask = (times >= window_start) & (times <= window_end)
    t_w = times[mask]
    if t_w.size < 8:
        return np.array([0.0]), np.array([0.0])
    dt = np.mean(np.diff(t_w))
    n = t_w.size
    freqs = np.fft.fftfreq(n, d=dt) / C_FS          # cm^-1 (signed)
    power = np.zeros(n)
    for a in range(N_SITES):
        for b in range(a + 1, N_SITES):
            c = rho_ex[mask, a, b]
            c = c - c.mean()
            power += np.abs(np.fft.fft(c)) ** 2
    order = np.argsort(freqs)
    return freqs[order], power[order]


def figure4_coherence_spectra(t_end: float = 5000.0, n_steps: int = 600,
                              initial_site: int = ENTRY_SITES[0],
                              window_start_fs: float = 100.0,
                              window_end_fs: float = 3000.0) -> None:
    print(f"\n  Coherence spectra at native geometry (start BChl {initial_site+1})")
    with tqdm(total=2, desc="Fig 4", unit="solve") as bar:
        bar.set_description("Fig 4  Ohmic")
        t_A, rhos_A = run_ohmic(initial_site=initial_site, t_end=t_end, n_steps=n_steps)
        eig, rho_ex_A = _exciton_coherences(rhos_A, is_vibronic=False)
        bar.update(1)
        bar.set_description("Fig 4  Vibronic")
        t_B, rhos_B = run_structured(initial_site=initial_site, t_end=t_end, n_steps=n_steps)
        _, rho_ex_B = _exciton_coherences(rhos_B, is_vibronic=True)
        bar.update(1)

    window_end_fs = min(window_end_fs, t_end)
    gaps = sorted({abs(eig[b] - eig[a]) for a in range(N_SITES)
                   for b in range(a + 1, N_SITES) if abs(eig[b] - eig[a]) <= 1000})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")

    for ax, (times, rho_ex, variant, color) in zip(axes, [
            (t_A, rho_ex_A, "Ohmic", "#1a6e8c"),
            (t_B, rho_ex_B, "Vibronic", "#d45f1e")]):
        ax.set_facecolor("#fffdf7")
        f, p = _coherence_spectrum(times, rho_ex, window_start_fs, window_end_fs)
        pos = f >= 0
        f, p = f[pos], p[pos]
        if p.max() > 0:
            p = p / p.max()
        m = f <= 1000.0
        ax.plot(f[m], p[m], color=color, lw=1.6)

        for g in gaps:
            ax.axvline(g, color="#555", ls="--", lw=0.7, alpha=0.35)
        for omega, name in [(OMEGA1_CM, f"{OMEGA1_CM:.0f}"),
                            (OMEGA2_CM, f"{OMEGA2_CM:.0f}")]:
            ax.axvline(omega, color="#a33", ls=":", lw=1.1, alpha=0.7)
            ax.text(omega + 6, 0.88, name, fontsize=8, color="#a33")

        ax.set_xlabel("Frequency (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Normalised power", fontsize=11)
        ax.set_title(f"Coherence spectrum — {variant}", fontsize=11)
        ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.15, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].plot([], [], color="#555", ls="--", lw=0.7, label="exciton gaps")
    axes[0].plot([], [], color="#a33", ls=":", lw=1.1, label="vib. modes")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Fig. 4: Exciton coherence spectra (FFT of complex coherence) — "
                 f"native FMO, window {window_start_fs:.0f}–{window_end_fs:.0f} fs",
                 fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "fig4_coherence_spectra.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


if __name__ == "__main__":
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()

    parser = argparse.ArgumentParser(description="Phase 3: comparison + coherence")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fig3-only", action="store_true")
    parser.add_argument("--fig4-only", action="store_true")
    args = parser.parse_args()

    t_end = 2000.0 if args.quick else 5000.0
    n_steps = 300 if args.quick else 700

    if not args.fig4_only:
        print("=== Figure 3: Ohmic vs vibronic dynamics ===")
        figure3_dynamics(t_end=t_end, n_steps=n_steps)
    if not args.fig3_only:
        print("\n=== Figure 4: Coherence spectra ===")
        figure4_coherence_spectra(t_end=t_end, n_steps=n_steps,
                                  window_start_fs=100.0,
                                  window_end_fs=min(t_end, 3000.0))
    print("\nDone.")
