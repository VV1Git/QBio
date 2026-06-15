"""
Phase 3: Comparison and coherence spectroscopy.

Produces two figures:
    fig3_p4_comparison.png   — P₄(t) for Ohmic (Variant A) vs vibronic (Variant B)
                               at two geometries: r=11.3 Å and r=8 Å, both θ=0.
    fig4_coherence_spectra.png — FFT power spectra of exciton-basis coherences
                               |ρ_{α,β}(t)| for both variants at r=11.3 Å.

Usage
-----
    python analysis.py                 # full run (both figures)
    python analysis.py --quick         # shorter propagation for testing
"""

from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh as _eigh

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

from dynamics import run_ohmic, population
from vibronic import run_structured, population_vibronic
from hamiltonian import build_electronic_H, SITE_ENERGIES_CM

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_FS = 3e-5   # cm/fs


# ── Figure 3: P₄(t) comparison ───────────────────────────────────────────────

def figure3_p4_comparison(t_end: float = 5000.0, n_steps: int = 400) -> None:
    """Plot P₄(t) for Variant A (Ohmic) and Variant B (vibronic) side-by-side."""
    geometries = [
        (11.3, "r = 11.3 Å (optimal)"),
        (8.0,  "r = 8 Å  (compact)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
    fig.patch.set_facecolor("#f7f2ea")
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    for ax, (r_val, title) in zip(axes, geometries):
        print(f"\n  {title}")

        print("    Variant A (Ohmic/Redfield) …", end="", flush=True)
        t0 = time.time()
        t_A, rhos_A = run_ohmic(r=r_val, theta=0.0, t_end=t_end, n_steps=n_steps)
        P4_A = population(rhos_A, site=3)
        print(f" {time.time()-t0:.1f}s  P4(∞)={P4_A[-1]:.3f}")

        print("    Variant B (vibronic/Lindblad) …", end="", flush=True)
        t0 = time.time()
        t_B, rhos_el_B = run_structured(r=r_val, theta=0.0, t_end=t_end, n_steps=n_steps)
        P4_B = population_vibronic(rhos_el_B, site=3)
        print(f" {time.time()-t0:.1f}s  P4(∞)={P4_B[-1]:.3f}")

        ax.plot(t_A, P4_A, color="#1a6e8c", lw=2.0, label="Variant A (Ohmic)")
        ax.plot(t_B, P4_B, color="#d45f1e", lw=2.0, ls="--", label="Variant B (vibronic)")
        ax.set_xlabel("Time (fs)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, t_end)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Acceptor population P₄(t)", fontsize=11)
    fig.suptitle("Fig. 3: P₄(t) — Ohmic vs vibronic model, θ = 0", fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig3_p4_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\n  Saved {out}")


# ── Figure 4: exciton-basis coherence spectra ─────────────────────────────────

def _exciton_basis_rhos(rhos_dm: list, r: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform electronic density matrices into the exciton basis.

    Returns
    -------
    eigvals : (4,) array of exciton energies (cm⁻¹)
    rho_ex  : (T, 4, 4) complex array — exciton-basis density matrix vs time
    """
    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(4)
    eigvals, U = _eigh(H_np)

    T = len(rhos_dm)
    rho_ex = np.zeros((T, 4, 4), dtype=complex)
    for k, rho in enumerate(rhos_dm):
        if hasattr(rho, "full"):
            rho_np = rho.full()
        else:
            rho_np = np.array(rho)
        rho_ex[k] = U.conj().T @ rho_np @ U

    return eigvals, rho_ex


def figure4_coherence_spectra(
    r: float = 11.3,
    theta: float = 0.0,
    t_end: float = 5000.0,
    n_steps: int = 500,
    window_start_fs: float = 300.0,
    window_end_fs: float = 2000.0,
) -> None:
    """
    Compute and plot exciton-basis coherence power spectra for Variant A and B.

    The FFT is applied to |⟨εᵢ|ρ_el(t)|εⱼ⟩| for i≠j over a time window
    that avoids the initial transient and cuts off before full relaxation.
    """
    print(f"\n  Coherence spectra at r={r} Å, θ={np.degrees(theta):.0f}°")

    print("    Variant A …", end="", flush=True)
    t0 = time.time()
    t_A, rhos_A = run_ohmic(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
    _, rho_ex_A = _exciton_basis_rhos(rhos_A, r, theta)
    print(f" {time.time()-t0:.1f}s")

    print("    Variant B …", end="", flush=True)
    t0 = time.time()
    t_B, rhos_el_B = run_structured(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
    _, rho_ex_B = _exciton_basis_rhos(rhos_el_B, r, theta)
    eigvals, _ = _exciton_basis_rhos(rhos_el_B[:1], r, theta)
    print(f" {time.time()-t0:.1f}s")

    def _spectrum(times, rho_ex, label):
        """FFT of the sum of |off-diagonal| exciton coherences."""
        mask = (times >= window_start_fs) & (times <= window_end_fs)
        t_win = times[mask]
        if t_win.size < 4:
            return np.array([0.0]), np.array([0.0])

        # Sum |ρ_{αβ}| over all α<β pairs
        coh_sum = np.zeros(t_win.size, dtype=float)
        for a in range(4):
            for b in range(a + 1, 4):
                coh_sum += np.abs(rho_ex[mask, a, b])

        dt_fs   = np.mean(np.diff(t_win))
        n       = t_win.size
        freqs   = np.fft.rfftfreq(n, d=dt_fs)     # fs⁻¹
        freqs_cm = freqs / C_FS                    # cm⁻¹  (ν = f / c)
        power   = np.abs(np.fft.rfft(coh_sum)) ** 2
        return freqs_cm, power

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=False)
    fig.patch.set_facecolor("#f7f2ea")

    for ax, (times, rho_ex, variant, color) in zip(
        axes,
        [
            (t_A, rho_ex_A, "A (Ohmic)", "#1a6e8c"),
            (t_B, rho_ex_B, "B (vibronic)", "#d45f1e"),
        ],
    ):
        ax.set_facecolor("#fffdf7")
        freqs_cm, power = _spectrum(times, rho_ex, variant)

        # Normalise to max = 1 for comparison
        pmax = power.max()
        if pmax > 0:
            power = power / pmax

        # Only show up to 1000 cm⁻¹
        mask_f = freqs_cm <= 1000.0
        ax.plot(freqs_cm[mask_f], power[mask_f], color=color, lw=1.5)

        # Annotate mode frequencies
        for omega, name in [(OMEGA1_CM_ANNOT, "Mode 1\n726"), (OMEGA2_CM_ANNOT, "Mode 2\n243")]:
            if omega <= 1000.0:
                ax.axvline(omega, color="gray", ls=":", lw=1.0, alpha=0.6)
                ax.text(omega + 8, 0.85, name, fontsize=7, color="gray")

        # Annotate exciton gaps
        if eigvals is not None:
            for a in range(4):
                for b in range(a + 1, 4):
                    gap = abs(eigvals[b] - eigvals[a])
                    if gap <= 1000.0:
                        ax.axvline(gap, color="#444", ls="--", lw=0.8, alpha=0.4)

        ax.set_xlabel("Frequency (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Normalised power", fontsize=11)
        ax.set_title(f"Coherence spectrum — Variant {variant}", fontsize=11)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Fig. 4: Exciton coherence spectra — r={r} Å, θ={np.degrees(theta):.0f}°  "
        f"(window {window_start_fs:.0f}–{window_end_fs:.0f} fs)",
        fontsize=11,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig4_coherence_spectra.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# Mode frequencies for annotations
OMEGA1_CM_ANNOT = 726.0
OMEGA2_CM_ANNOT = 243.0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 analysis: comparison + coherence spectra")
    parser.add_argument("--quick", action="store_true", help="Faster low-res run")
    parser.add_argument("--fig3-only", action="store_true")
    parser.add_argument("--fig4-only", action="store_true")
    args = parser.parse_args()

    t_end  = 2000.0  if args.quick else 5000.0
    n_steps = 200    if args.quick else 400

    run_3 = not args.fig4_only
    run_4 = not args.fig3_only

    if run_3:
        print("=== Figure 3: P₄(t) comparison ===")
        figure3_p4_comparison(t_end=t_end, n_steps=n_steps)

    if run_4:
        print("\n=== Figure 4: Coherence spectra ===")
        figure4_coherence_spectra(
            r=11.3, theta=0.0, t_end=t_end, n_steps=n_steps,
            window_start_fs=300.0, window_end_fs=min(t_end, 2000.0),
        )

    print("\nDone.")
