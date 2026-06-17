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
from scipy.linalg import eigh as _eigh
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

from dynamics import run_ohmic, population
from vibronic import run_structured, population_vibronic
from hamiltonian import build_electronic_H, SITE_ENERGIES_CM

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_FS = 3e-5   # cm/fs

# Mode frequencies for annotations (match vibronic.py: Rätsep 2007)
OMEGA1_CM_ANNOT = 770.0
OMEGA2_CM_ANNOT = 243.0


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

    # Run all 4 solves with a progress bar, then plot
    tasks = [(ax, r_val, title, variant)
             for (ax, (r_val, title)) in zip(axes, geometries)
             for variant in ("ohmic", "vibronic")]
    results: dict = {}

    with tqdm(tasks, desc="Fig 3", unit="solve") as bar:
        for ax, r_val, title, variant in bar:
            bar.set_description(f"Fig 3  r={r_val} Å  {variant}")
            t0 = time.time()
            if variant == "ohmic":
                t_, rhos_ = run_ohmic(r=r_val, theta=0.0, t_end=t_end, n_steps=n_steps)
                P4_ = population(rhos_, site=3)
            else:
                t_, rhos_el_ = run_structured(r=r_val, theta=0.0, t_end=t_end, n_steps=n_steps)
                P4_ = population_vibronic(rhos_el_, site=3)
            results[(r_val, variant)] = (t_, P4_)
            tqdm.write(f"  r={r_val} Å  {variant}: {time.time()-t0:.1f}s  P4(∞)={P4_[-1]:.3f}")

    for ax, (r_val, title) in zip(axes, geometries):
        t_A, P4_A = results[(r_val, "ohmic")]
        t_B, P4_B = results[(r_val, "vibronic")]
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

    # Batch transform all timesteps at once: rho_ex[k] = U† @ rho[k] @ U
    rhos_np = np.array([
        rho.full() if hasattr(rho, "full") else np.array(rho)
        for rho in rhos_dm
    ])  # (T, 4, 4)
    Uc = U.conj().T
    rho_ex = np.einsum("ij,kjl,lm->kim", Uc, rhos_np, U)

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

    with tqdm(total=2, desc="Fig 4", unit="solve") as bar:
        bar.set_description("Fig 4  Variant A")
        t0 = time.time()
        t_A, rhos_A = run_ohmic(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
        _, rho_ex_A = _exciton_basis_rhos(rhos_A, r, theta)
        tqdm.write(f"  Variant A: {time.time()-t0:.1f}s")
        bar.update(1)

        bar.set_description("Fig 4  Variant B")
        t0 = time.time()
        t_B, rhos_el_B = run_structured(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
        _, rho_ex_B = _exciton_basis_rhos(rhos_el_B, r, theta)
        eigvals, _ = _exciton_basis_rhos(rhos_el_B[:1], r, theta)
        tqdm.write(f"  Variant B: {time.time()-t0:.1f}s")
        bar.update(1)

    def _spectrum(times, rho_ex):
        """FFT of the sum of |off-diagonal| exciton coherences."""
        mask = (times >= window_start_fs) & (times <= window_end_fs)
        t_win = times[mask]
        if t_win.size < 4:
            return np.array([0.0]), np.array([0.0])

        # Sum |ρ_{αβ}| over all α<β pairs — vectorised with upper-triangle mask
        triu = np.triu(np.ones((4, 4), dtype=bool), k=1)
        coh_sum = np.sum(np.abs(rho_ex[mask][:, triu]), axis=1)

        dt_fs    = np.mean(np.diff(t_win))
        freqs_cm = np.fft.rfftfreq(t_win.size, d=dt_fs) / C_FS   # cm⁻¹
        power    = np.abs(np.fft.rfft(coh_sum)) ** 2
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
        freqs_cm, power = _spectrum(times, rho_ex)

        # Normalise to max = 1 for comparison
        pmax = power.max()
        if pmax > 0:
            power = power / pmax

        # Only show up to 1000 cm⁻¹
        mask_f = freqs_cm <= 1000.0
        ax.plot(freqs_cm[mask_f], power[mask_f], color=color, lw=1.5)

        # Annotate mode frequencies
        for omega, name in [(OMEGA1_CM_ANNOT, "Mode 1\n770"), (OMEGA2_CM_ANNOT, "Mode 2\n243")]:
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()

    parser = argparse.ArgumentParser(description="Phase 3 analysis: comparison + coherence spectra")
    parser.add_argument("--quick", action="store_true", help="Faster low-res run")
    parser.add_argument("--fig3-only", action="store_true")
    parser.add_argument("--fig4-only", action="store_true")
    args = parser.parse_args()

    t_end  = 2000.0  if args.quick else 5000.0
    n_steps = 300    if args.quick else 700

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
