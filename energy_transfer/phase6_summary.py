"""
Phase 6: Summary figure and quantitative conclusions.

Assembles a publication-style 6-panel figure that tells the full story:

    [1] P₄(t) at optimal geometry — Ohmic vs vibronic
    [2] Reff(r,θ) heatmap — Ohmic (secular Redfield)
    [3] Reff(r,θ) heatmap — Vibronic (Lindblad)
    [4] ΔReff = vibronic − Ohmic  (signed enhancement map)
    [5] Bath λ sensitivity at r=11.3 Å
    [6] Coherence spectrum at r=11.3 Å — Ohmic vs vibronic (overlaid)

All data is loaded from previously saved .npy / .npz files where possible,
so this script is fast to re-run after all prior phases have completed.

Usage
-----
    python phase6_summary.py
    python phase6_summary.py --recompute   # ignore cached data and rerun
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eigh as _eigh

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_FS = 3e-5


# ── Data loaders (with fallback recomputation) ─────────────────────────────────

def _load_or(path: Path, recompute: bool, compute_fn):
    """Return loaded array or call compute_fn() to generate it."""
    if not recompute and path.exists():
        return np.load(path, allow_pickle=True)
    print(f"  Recomputing {path.name} …")
    result = compute_fn()
    np.save(path, result)
    return result


def _load_npz_or(path: Path, recompute: bool, compute_fn):
    if not recompute and path.exists():
        return dict(np.load(path, allow_pickle=True))
    print(f"  Recomputing {path.name} …")
    return compute_fn()


def _compute_p4_dynamics(r=11.3, theta=0.0, t_end=5000.0, n_steps=400):
    from dynamics import run_ohmic, population
    from vibronic import run_structured, population_vibronic
    t_A, rhos_A = run_ohmic(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
    P4_A = population(rhos_A, site=3)
    t_B, rhos_el_B = run_structured(r=r, theta=theta, t_end=t_end, n_steps=n_steps)
    P4_B = population_vibronic(rhos_el_B, site=3)
    return t_A, P4_A, t_B, P4_B, rhos_A, rhos_el_B


def _compute_coherence(rhos, r, theta, is_vibronic=False):
    from hamiltonian import build_electronic_H, SITE_ENERGIES_CM
    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(4)
    _, U = _eigh(H_np)
    T = len(rhos)
    coh_sum = np.zeros(T)
    for k, rho in enumerate(rhos):
        rho_el = rho.ptrace([0]) if is_vibronic else rho
        rho_np = np.array(rho_el.full())
        rho_ex = U.conj().T @ rho_np @ U
        for a in range(4):
            for b in range(a + 1, 4):
                coh_sum[k] += abs(rho_ex[a, b])
    return coh_sum


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _panel_dynamics(ax, t_A, P4_A, t_B, P4_B):
    ax.plot(t_A, P4_A, color="#1a6e8c", lw=1.8, label="Ohmic")
    ax.plot(t_B, P4_B, color="#d45f1e", lw=1.8, ls="--", label="Vibronic")
    ax.set_xlabel("Time (fs)", fontsize=9)
    ax.set_ylabel("P₄(t)", fontsize=9)
    ax.set_title("P₄(t) at optimal geometry\n(r=11.3 Å, θ=0°)", fontsize=9)
    ax.set_xlim(0, t_A[-1])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _panel_heatmap(ax, fig, r_grid, theta_grid, reff_grid, title, cmap="viridis",
                   vmin=None, vmax=None, center=None):
    theta_deg = np.degrees(theta_grid)
    kwargs = dict(cmap=cmap, shading="auto")
    if center is not None:
        abs_max = np.abs(reff_grid * 1e3).max()
        kwargs.update(vmin=-abs_max, vmax=abs_max)
    else:
        kwargs.update(vmin=vmin or 0, vmax=vmax or reff_grid.max() * 1e3)
    img = ax.pcolormesh(theta_deg, r_grid, reff_grid * 1e3, **kwargs)
    cb  = fig.colorbar(img, ax=ax, pad=0.02)
    cb.set_label("Reff (ps⁻¹)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    idx = np.unravel_index(np.argmax(reff_grid), reff_grid.shape)
    r_opt, th_opt = r_grid[idx[0]], theta_grid[idx[1]]
    ax.plot(np.degrees(th_opt), r_opt, "w*", ms=9, label=f"{r_opt:.1f} Å, {np.degrees(th_opt):.0f}°")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlabel("θ (°)", fontsize=9); ax.set_ylabel("r (Å)", fontsize=9)
    ax.set_title(title, fontsize=9)


def _panel_sensitivity(ax, lam_grid, ohm_lam, vib_lam, lambda_default=35.0):
    ax.plot(lam_grid, ohm_lam * 1e3, color="#1a6e8c", lw=1.8, label="Ohmic")
    ax.plot(lam_grid, vib_lam * 1e3, color="#d45f1e", lw=1.8, ls="--", label="Vibronic")
    ax.axvline(lambda_default, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("λ (cm⁻¹)", fontsize=9); ax.set_ylabel("Reff (ps⁻¹)", fontsize=9)
    ax.set_title("Reff vs reorganisation energy\n(r=11.3 Å, θ=0°)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _panel_coherence(ax, t_A, coh_A, t_B, coh_B,
                     window_start=300.0, window_end=2000.0):
    def _fft(times, coh):
        mask = (times >= window_start) & (times <= window_end)
        t_w  = times[mask]; sig = coh[mask]
        if t_w.size < 4:
            return np.array([0.0]), np.array([0.0])
        dt   = np.mean(np.diff(t_w))
        freq = np.fft.rfftfreq(t_w.size, d=dt) / C_FS   # cm⁻¹
        power = np.abs(np.fft.rfft(sig)) ** 2
        return freq, power / max(power.max(), 1e-30)

    f_A, p_A = _fft(t_A, coh_A)
    f_B, p_B = _fft(t_B, coh_B)
    mask_f = f_A <= 1000.0
    ax.plot(f_A[mask_f], p_A[mask_f], color="#1a6e8c", lw=1.5, label="Ohmic")
    mask_f2 = f_B <= 1000.0
    ax.plot(f_B[mask_f2], p_B[mask_f2], color="#d45f1e", lw=1.5, ls="--", label="Vibronic")
    for omega, name in [(726, "726"), (243, "243")]:
        ax.axvline(omega, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.text(omega + 8, 0.82, name, fontsize=7, color="gray")
    ax.set_xlabel("Frequency (cm⁻¹)", fontsize=9); ax.set_ylabel("Power (norm.)", fontsize=9)
    ax.set_title("Exciton coherence spectrum\n(r=11.3 Å, θ=0°)", fontsize=9)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_summary(recompute: bool = False) -> None:
    # ── load / compute all data ──
    print("Loading Phase 4 heatmap data …")
    try:
        r_grid    = np.load(RESULTS_DIR / "p4_r_grid.npy")
        th_grid   = np.load(RESULTS_DIR / "p4_theta_grid.npy")
        ohm_grid  = np.load(RESULTS_DIR / "p4_reff_ohmic.npy")
        vib_grid  = np.load(RESULTS_DIR / "p4_reff_vibronic.npy")
        have_p4   = True
    except FileNotFoundError:
        print("  Phase 4 data missing — will use Phase 1 heatmap data")
        try:
            r_grid   = np.load(RESULTS_DIR / "r_grid.npy")
            th_grid  = np.load(RESULTS_DIR / "theta_grid.npy")
            ohm_grid = np.load(RESULTS_DIR / "reff_ohmic.npy")
            vib_grid = None
            have_p4  = False
        except FileNotFoundError:
            r_grid = th_grid = ohm_grid = vib_grid = None
            have_p4 = False

    print("Loading Phase 5 sensitivity data …")
    try:
        p5 = dict(np.load(RESULTS_DIR / "p5_sensitivity_data.npz"))
        have_p5 = True
    except FileNotFoundError:
        print("  Phase 5 data missing — will skip sensitivity panel")
        have_p5 = False
        p5 = {}

    print("Computing/loading dynamics at optimal geometry …")
    from dynamics import run_ohmic, population
    from vibronic import run_structured, population_vibronic

    t_A, rhos_A = run_ohmic(r=11.3, theta=0.0, t_end=5000.0, n_steps=300)
    P4_A = population(rhos_A, site=3)
    coh_A = _compute_coherence(rhos_A, 11.3, 0.0, is_vibronic=False)

    t_B, rhos_el_B = run_structured(r=11.3, theta=0.0, t_end=5000.0, n_steps=300)
    P4_B = population_vibronic(rhos_el_B, site=3)
    coh_B = _compute_coherence(rhos_el_B, 11.3, 0.0, is_vibronic=True)

    # ── build figure ──
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f7f2ea")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#fffdf7")

    # Panel 1: dynamics
    _panel_dynamics(axes[0], t_A, P4_A, t_B, P4_B)

    # Panel 2: Ohmic heatmap
    if ohm_grid is not None:
        _panel_heatmap(axes[1], fig, r_grid, th_grid, ohm_grid,
                       "Reff(r,θ) — Ohmic")
    else:
        axes[1].text(0.5, 0.5, "Phase 4 data\nnot found", ha="center",
                     va="center", transform=axes[1].transAxes, fontsize=10)
        axes[1].set_title("Reff(r,θ) — Ohmic", fontsize=9)

    # Panel 3: Vibronic heatmap
    if vib_grid is not None:
        _panel_heatmap(axes[2], fig, r_grid, th_grid, vib_grid,
                       "Reff(r,θ) — Vibronic")
    elif ohm_grid is not None:
        axes[2].text(0.5, 0.5, "Vibronic scan\nnot available", ha="center",
                     va="center", transform=axes[2].transAxes, fontsize=10)
        axes[2].set_title("Reff(r,θ) — Vibronic", fontsize=9)

    # Panel 4: ΔReff
    if ohm_grid is not None and vib_grid is not None:
        delta = vib_grid - ohm_grid
        _panel_heatmap(axes[3], fig, r_grid, th_grid, delta,
                       "ΔReff (vibronic − Ohmic)", cmap="RdBu_r", center=True)
    elif ohm_grid is not None:
        axes[3].text(0.5, 0.5, "ΔReff requires\nPhase 4 vibronic data",
                     ha="center", va="center", transform=axes[3].transAxes, fontsize=10)
        axes[3].set_title("ΔReff (vibronic − Ohmic)", fontsize=9)

    # Panel 5: λ sensitivity
    if have_p5:
        _panel_sensitivity(axes[4],
                           p5["lambda_grid"], p5["ohm_lam"], p5["vib_lam"])
    else:
        axes[4].text(0.5, 0.5, "Phase 5 data\nnot found", ha="center",
                     va="center", transform=axes[4].transAxes, fontsize=10)
        axes[4].set_title("Reff vs λ", fontsize=9)

    # Panel 6: coherence spectrum
    _panel_coherence(axes[5], t_A, coh_A, t_B, coh_B,
                     window_start=300.0, window_end=2000.0)

    fig.suptitle(
        "Summary: 4-site dimerized Frenkel exciton — Ohmic vs vibronic bath",
        fontsize=13, y=0.98
    )
    out = RESULTS_DIR / "fig7_summary.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")

    # ── print key numbers ──
    print("\n=== Key results ===")
    print(f"  Ohmic   Reff at r=11.3 Å: {_ohm_reff_opt()*1e3:.4f} ps⁻¹")
    if ohm_grid is not None:
        idx = np.unravel_index(np.argmax(ohm_grid), ohm_grid.shape)
        print(f"  Ohmic   peak Reff: {ohm_grid[idx]*1e3:.3f} ps⁻¹ at "
              f"r={r_grid[idx[0]]:.1f} Å, θ={np.degrees(th_grid[idx[1]]):.0f}°")
    if vib_grid is not None:
        idx2 = np.unravel_index(np.argmax(vib_grid), vib_grid.shape)
        print(f"  Vibronic peak Reff: {vib_grid[idx2]*1e3:.3f} ps⁻¹ at "
              f"r={r_grid[idx2[0]]:.1f} Å, θ={np.degrees(th_grid[idx2[1]]):.0f}°")
        # Shift in optimal geometry
        if r_grid[idx[0]] != r_grid[idx2[0]] or th_grid[idx[1]] != th_grid[idx2[1]]:
            print("  → Vibronic modes shift the optimal geometry")
        enhancement = (vib_grid[idx2] - ohm_grid[idx2]) / ohm_grid[idx2] * 100
        print(f"  → Vibronic enhancement at vibronic optimum: {enhancement:+.1f}%")


def _ohm_reff_opt():
    from dynamics import secular_reff
    _, _, reff = secular_reff(11.3, 0.0)
    return reff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute dynamics rather than loading cached data")
    args = parser.parse_args()

    print("=== Phase 6: Summary figure ===")
    build_summary(recompute=args.recompute)
    print("\nDone.")
