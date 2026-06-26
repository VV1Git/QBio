"""
Phase 7: Impact analyses for the 8-site FMO position-vs-efficiency study.

Six self-contained analyses that turn the raw position scan into specific,
biologically interpretable claims:

  (1) BChl 8 trimer-entry role  — does BChl 8's position matter when it is the
      excitation entry point (its inter-monomer role) rather than a bystander of
      the intra-monomer funnel?
  (2) Geometric sensitivity fingerprint — the displacement radius at which each
      pigment's ETE first drops below a series of thresholds (0.995 … 0.90).
  (3) Out-of-plane tolerance — 1-D scans along the FMO disk normal for the most
      tightly constrained pigments, completing the 2-D in-plane picture.
  (4) Anatomy of the optimum — the site-energy / coupling changes the global
      optimiser makes, showing it *decouples* peripheral pigments rather than
      fine-tuning the funnel.
  (5) Pocket-tightness proxy — number of protein heavy atoms lining each
      pigment, a structural correlate of positional sensitivity (a computable
      stand-in for cross-species sequence conservation).
  (6) Noise robustness — ETE of the native vs optimised arrangement under
      Gaussian positional jitter, testing whether each is an evolutionarily
      accessible plateau or a sharp, unrealisable peak.

Outputs (results/):
    fig8_sensitivity_fingerprint.png   (analyses 2 + 5)
    fig9_bchl8_trimer.png              (analysis 1)
    fig10_out_of_plane.png             (analysis 3)
    fig11_optimum_anatomy.png          (analysis 4)
    fig12_noise_robustness.png         (analysis 6)
    p7_impact_data.npz

Usage
-----
    python phase7_impact.py
    python phase7_impact.py --quick
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from fmo_data import N_SITES, TRAP_SITE, ENTRY_SITES
from hamiltonian import build_electronic_H
from dynamics import compute_ete
from geometry_scan import PLANE_AXES, NORMAL_AXIS, disp_from_inplane, efficiency
from electrostatics import (
    site_energies_batch, scan_pigment_grid, PROT_XYZ, PIG_XYZ,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_SITE_COLORS = plt.cm.viridis(np.linspace(0, 0.92, N_SITES))
BG, PANEL = "#f7f2ea", "#fffdf7"
SPAN = 6.0

# ETE thresholds for the sensitivity fingerprint (analysis 2)
THRESHOLDS = (0.995, 0.99, 0.95, 0.90)


def _role(i: int) -> str:
    if i == TRAP_SITE:
        return "trap→RC"
    if i in ENTRY_SITES:
        return "entry"
    return "core"


# ── Shared: ETE over a single-pigment in-plane grid, arbitrary entry sites ─────

def _scan_pigment_entry(p: int, us: np.ndarray, vs: np.ndarray,
                        initial_sites: tuple[int, ...]) -> np.ndarray:
    """ETE grid (len(us), len(vs)) for moving pigment p, excitation entering at
    `initial_sites`.  CDC shifts are batched; the ETE solve is per point."""
    U, V = np.meshgrid(us, vs, indexing="ij")
    disps2d = np.stack([U.ravel(), V.ravel()], axis=1)        # (G,2)
    disps3d = disps2d @ PLANE_AXES                            # (G,3)
    cdc = scan_pigment_grid(p, disps3d)                       # (G,8)
    ete = np.empty(disps3d.shape[0])
    full = np.zeros((N_SITES, 3))
    for g in range(disps3d.shape[0]):
        full[:] = 0.0
        full[p] = disps3d[g]
        ete[g], _ = compute_ete(build_electronic_H(full, site_shift=cdc[g]),
                                initial_sites=initial_sites)
    return ete.reshape(len(us), len(vs))


# ── Analysis 2: geometric sensitivity fingerprint ─────────────────────────────

def tolerance_radii(ete_grid: np.ndarray, us: np.ndarray, vs: np.ndarray,
                    thresholds=THRESHOLDS) -> dict:
    """For each threshold, the smallest displacement radius (Å) at which ETE
    drops below it (within the inscribed disk R ≤ span).  np.inf if never."""
    U, V = np.meshgrid(us, vs, indexing="ij")
    R = np.sqrt(U ** 2 + V ** 2)
    inside = R <= (us.max())
    out = {}
    for thr in thresholds:
        below = inside & (ete_grid < thr)
        out[thr] = float(R[below].min()) if below.any() else np.inf
    return out


def analysis_fingerprint(p4: dict) -> dict:
    us, vs, ete = p4["us"], p4["vs"], p4["ete"]
    radii = {p: tolerance_radii(ete[p], us, vs) for p in range(N_SITES)}
    frac = {p: float((ete[p] > 0.99).mean()) for p in range(N_SITES)}
    return {"radii": radii, "frac_high": frac, "us": us}


# ── Analysis 5: structural pocket-tightness proxy ─────────────────────────────

def analysis_pocket(cutoff: float = 5.0) -> np.ndarray:
    """Number of protein heavy atoms within `cutoff` Å of each pigment's
    macrocycle — a structural correlate of how tightly the scaffold encloses it."""
    counts = np.empty(N_SITES, dtype=int)
    for p in range(N_SITES):
        d = np.linalg.norm(PIG_XYZ[p][:, None, :] - PROT_XYZ[None, :, :], axis=-1)
        counts[p] = int((d.min(axis=0) <= cutoff).sum())
    return counts


def plot_fingerprint(fp: dict, pocket: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    # Panel A: tolerance radius vs threshold, one line per pigment
    ax = axes[0]
    radii = fp["radii"]
    cap = SPAN * 1.08
    xs = np.arange(len(THRESHOLDS))
    for p in range(N_SITES):
        ys = [min(radii[p][t], cap) if np.isfinite(radii[p][t]) else cap
              for t in THRESHOLDS]
        ax.plot(xs, ys, "-o", color=_SITE_COLORS[p], lw=2 if p in (0, 2, 5, 7) else 1,
                ms=6 if p in (0, 2, 5, 7) else 4,
                label=f"BChl {p+1} ({_role(p)})")
    ax.axhline(SPAN, color="#999", ls=":", lw=1)
    ax.text(0.02, SPAN + 0.1, "scan limit (tolerant beyond)", fontsize=7, color="#777")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t:.3f}".rstrip("0") for t in THRESHOLDS])
    ax.set_xlabel("ETE threshold", fontsize=10)
    ax.set_ylabel("tolerance radius (Å)  — larger = more position-tolerant", fontsize=10)
    ax.set_title("Geometric sensitivity fingerprint", fontsize=11)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.18, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B: pocket tightness vs tolerance radius (at ETE 0.99)
    ax = axes[1]
    tol99 = np.array([min(radii[p][0.99], cap) if np.isfinite(radii[p][0.99]) else cap
                      for p in range(N_SITES)])
    for p in range(N_SITES):
        ax.scatter(pocket[p], tol99[p], s=140, color=_SITE_COLORS[p],
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(f"{p+1}", (pocket[p], tol99[p]), fontsize=8,
                    ha="center", va="center", zorder=4,
                    color="white" if p not in (7,) else "black")
    # correlation
    r = np.corrcoef(pocket, tol99)[0, 1]
    ax.set_xlabel("protein heavy atoms within 5 Å of pigment\n(structural pocket tightness)",
                  fontsize=10)
    ax.set_ylabel("tolerance radius at ETE 0.99 (Å)", fontsize=10)
    ax.set_title(f"Pocket tightness vs positional tolerance  (Pearson r = {r:+.2f})",
                 fontsize=11)
    ax.grid(True, alpha=0.18, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 8: Per-pigment positional sensitivity and its structural basis",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig8_sensitivity_fingerprint.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    return tol99, r


# ── Analysis 1: BChl 8 trimer-entry role ──────────────────────────────────────

def analysis_bchl8(n_grid: int) -> dict:
    """Scan BChl 8's position under (a) standard entry (BChl 1 & 6) and (b) entry
    at BChl 8 itself (its inter-monomer role), plus the native funnel under each."""
    us = np.linspace(-SPAN, SPAN, n_grid)
    p = 7
    ete_std = _scan_pigment_entry(p, us, us, ENTRY_SITES)        # bystander
    ete_b8 = _scan_pigment_entry(p, us, us, (7,))               # BChl8 is entry

    H = build_electronic_H()
    nat_std = compute_ete(H, initial_sites=ENTRY_SITES)
    nat_b8 = compute_ete(H, initial_sites=(7,))
    # per-pigment trapping time when BChl8 is the entry vs standard
    return {"us": us, "ete_std": ete_std, "ete_b8": ete_b8,
            "nat_std": nat_std, "nat_b8": nat_b8,
            "range_std": float(ete_std.max() - ete_std.min()),
            "range_b8": float(ete_b8.max() - ete_b8.min())}


def plot_bchl8(b8: dict) -> None:
    us = b8["us"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)
    fig.patch.set_facecolor(BG)
    vmin = min(b8["ete_std"].min(), b8["ete_b8"].min())
    for ax, key, title, nat in [
            (axes[0], "ete_std", "BChl 8 a bystander\n(entry at BChl 1 & 6)", b8["nat_std"]),
            (axes[1], "ete_b8", "BChl 8 IS the entry point\n(inter-monomer role)", b8["nat_b8"])]:
        ax.set_facecolor(PANEL)
        im = ax.pcolormesh(us, us, b8[key], cmap="magma", shading="gouraud",
                           vmin=vmin, vmax=1.0)
        ax.plot(0, 0, "*", color="#39d0ff", ms=16, markeredgecolor="black",
                markeredgewidth=0.7, label="native")
        bi = np.unravel_index(np.argmax(b8[key]), b8[key].shape)
        ax.plot(us[bi[1]], us[bi[0]], "X", color="white", ms=10,
                markeredgecolor="black", markeredgewidth=0.6, label="best spot")
        rng = b8[key].max() - b8[key].min()
        ax.set_title(f"{title}\nETE range {rng:.3f}  (native {nat[0]:.3f})", fontsize=10)
        ax.set_xlabel("in-plane shift e₂ (Å)", fontsize=9)
        ax.set_ylabel("in-plane shift e₁ (Å)", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("ETE", fontsize=9)

    fig.suptitle("Fig. 9: BChl 8 becomes position-sensitive only in its inter-monomer "
                 "entry role", fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig9_bchl8_trimer.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Analysis 3: out-of-plane tolerance ────────────────────────────────────────

def analysis_out_of_plane(pigments=(0, 2, 5, 7), n=41) -> dict:
    """1-D ETE scan along the disk-normal axis for selected pigments, compared to
    the in-plane principal axis (e1).  Reveals whether the scaffold constrains a
    pigment in 2-D (disk only) or 3-D."""
    shifts = np.linspace(-SPAN, SPAN, n)
    res = {"shifts": shifts, "normal": {}, "inplane": {}}
    for p in pigments:
        e_norm, e_plane = np.empty(n), np.empty(n)
        for i, s in enumerate(shifts):
            d_n = np.zeros((N_SITES, 3)); d_n[p] = s * NORMAL_AXIS
            d_p = np.zeros((N_SITES, 3)); d_p[p] = s * PLANE_AXES[0]
            e_norm[i] = efficiency(d_n)[0]
            e_plane[i] = efficiency(d_p)[0]
        res["normal"][p] = e_norm
        res["inplane"][p] = e_plane
    return res


def plot_out_of_plane(oop: dict) -> None:
    shifts = oop["shifts"]
    pigs = list(oop["normal"].keys())
    fig, axes = plt.subplots(1, len(pigs), figsize=(3.6 * len(pigs), 4.6),
                             dpi=150, sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, p in zip(np.atleast_1d(axes), pigs):
        ax.set_facecolor(PANEL)
        ax.plot(shifts, oop["inplane"][p], color="#1a6e8c", lw=2.2,
                label="in-plane (e₁)")
        ax.plot(shifts, oop["normal"][p], color="#d45f1e", lw=2.2, ls="--",
                label="out-of-plane (normal)")
        ax.axvline(0, color="#999", ls=":", lw=1)
        ax.set_title(f"BChl {p+1} ({_role(p)})", fontsize=10)
        ax.set_xlabel("displacement (Å)", fontsize=9)
        ax.grid(True, alpha=0.18, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("ETE", fontsize=10)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower center")
    fig.suptitle("Fig. 10: In-plane vs out-of-plane positional tolerance "
                 "(does the scaffold constrain in 2-D or 3-D?)", fontsize=11, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig10_out_of_plane.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Analysis 4: anatomy of the optimum ────────────────────────────────────────

def analysis_optimum(p4: dict) -> dict:
    disp_opt = disp_from_inplane(p4["disp_inplane"])
    H_nat = build_electronic_H()
    H_opt = build_electronic_H(disp_opt)
    dE = np.diag(H_opt) - np.diag(H_nat)
    eig_nat = np.linalg.eigvalsh(H_nat)
    eig_opt = np.linalg.eigvalsh(H_opt)
    return {"H_nat": H_nat, "H_opt": H_opt, "dE": dE,
            "band_nat": float(eig_nat[-1] - eig_nat[0]),
            "band_opt": float(eig_opt[-1] - eig_opt[0])}


def plot_optimum(opt: dict, p4: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for ax, H, title in [(axes[0], opt["H_nat"], "Native"),
                         (axes[1], opt["H_opt"], "Optimised")]:
        J = H - np.diag(np.diag(H))
        vmax = np.abs(opt["H_opt"] - np.diag(np.diag(opt["H_opt"]))).max()
        im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(N_SITES)); ax.set_xticklabels(range(1, N_SITES + 1), fontsize=7)
        ax.set_yticks(range(N_SITES)); ax.set_yticklabels(range(1, N_SITES + 1), fontsize=7)
        ax.set_title(f"{title} coupling matrix", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("J (cm⁻¹)", fontsize=8)

    ax = axes[2]
    colors = [_SITE_COLORS[i] for i in range(N_SITES)]
    ax.bar(range(1, N_SITES + 1), opt["dE"], color=colors, edgecolor="black", linewidth=0.6)
    for i, d in enumerate(opt["dE"]):
        ax.text(i + 1, d + (120 if d >= 0 else -120), f"{d:+.0f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=7)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("BChl", fontsize=10)
    ax.set_ylabel("site-energy shift at optimum (cm⁻¹)", fontsize=10)
    ax.set_title(f"Optimiser detunes the periphery\nexciton band {opt['band_nat']:.0f}→"
                 f"{opt['band_opt']:.0f} cm⁻¹", fontsize=10)
    ax.set_xticks(range(1, N_SITES + 1))
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 11: Anatomy of the global optimum — decoupling, not funnel "
                 f"fine-tuning  (ETE {float(p4['ete_native']):.4f}→{float(p4['ete_opt']):.4f}, "
                 f"τ {float(p4['tau_native'])/1000:.2f}→{float(p4['tau_opt'])/1000:.2f} ps)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig11_optimum_anatomy.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Analysis 6: noise robustness of native vs optimum ─────────────────────────

def _ete_batch_inplane(inplane_batch: np.ndarray) -> np.ndarray:
    """ETE for a batch of (B,8,2) in-plane arrangements (batched CDC)."""
    B = inplane_batch.shape[0]
    disp = inplane_batch @ PLANE_AXES                        # (B,8,3)
    shifts = site_energies_batch(disp)                       # (B,8)
    ete = np.empty(B)
    for b in range(B):
        ete[b], _ = compute_ete(build_electronic_H(disp[b], site_shift=shifts[b]))
    return ete


def analysis_noise(p4: dict, sigmas=(0.1, 0.25, 0.5, 1.0, 1.5),
                   n_samples: int = 200, seed: int = 0) -> dict:
    """ETE distribution of the native and optimised arrangements under Gaussian
    positional jitter of width σ (Å) on each in-plane coordinate."""
    rng = np.random.default_rng(seed)
    nat = np.zeros((N_SITES, 2))
    opt = np.asarray(p4["disp_inplane"], float)
    res = {"sigmas": np.array(sigmas)}
    for name, base in [("native", nat), ("optimum", opt)]:
        means, stds, p10, p50, p90 = [], [], [], [], []
        for s in sigmas:
            jitter = base[None] + rng.normal(0, s, (n_samples, N_SITES, 2))
            ete = _ete_batch_inplane(jitter)
            means.append(ete.mean()); stds.append(ete.std())
            # percentiles respect the hard ETE ≤ 1 bound (the distribution is
            # left-skewed against the ceiling, so mean±σ would overshoot 1).
            p10.append(np.percentile(ete, 10))
            p50.append(np.percentile(ete, 50))
            p90.append(np.percentile(ete, 90))
        res[name] = {"mean": np.array(means), "std": np.array(stds),
                     "p10": np.array(p10), "p50": np.array(p50),
                     "p90": np.array(p90)}
    res["ete_native0"] = float(p4["ete_native"])
    res["ete_opt0"] = float(p4["ete_opt"])
    return res


def plot_noise(nz: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    s = nz["sigmas"]
    for name, color, e0 in [("native", "#1a6e8c", nz["ete_native0"]),
                            ("optimum", "#d45f1e", nz["ete_opt0"])]:
        p10, p50, p90 = nz[name]["p10"], nz[name]["p50"], nz[name]["p90"]
        ax.plot(s, p50, "-o", color=color, lw=2.2, label=f"{name} (median)")
        ax.fill_between(s, p10, p90, color=color, alpha=0.15)
        ax.axhline(e0, color=color, ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel("positional jitter σ per coordinate (Å)", fontsize=10)
    ax.set_ylabel("ETE", fontsize=10)
    ax.set_ylim(top=1.005)
    ax.set_title("Fig. 12: Robustness of the native vs optimised arrangement\n"
                 "(dotted = unperturbed; band = 10th–90th percentile over 200 samples)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.18, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "fig12_noise_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()

    parser = argparse.ArgumentParser(description="Phase 7: impact analyses")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_grid = 31 if args.quick else 61
    n_samples = 80 if args.quick else 200

    p4_path = RESULTS_DIR / "p4_position_scan.npz"
    if not p4_path.exists():
        sys.exit("p4_position_scan.npz not found — run phase4_scan.py first.")
    p4 = dict(np.load(p4_path))

    print("=== Phase 7: Impact analyses (8-site FMO) ===\n")
    save = {}

    print("[2] Geometric sensitivity fingerprint …")
    fp = analysis_fingerprint(p4)
    print("[5] Structural pocket tightness …")
    pocket = analysis_pocket()
    tol99, r_pocket = plot_fingerprint(fp, pocket)
    for p in range(N_SITES):
        rr = fp["radii"][p]
        print(f"    BChl{p+1:<2}({_role(p):<7}): tol@0.99={rr[0.99]:.2f}Å  "
              f"high-eff disk={100*fp['frac_high'][p]:.0f}%  pocket={pocket[p]} atoms")
    print(f"    pocket-vs-tolerance correlation r = {r_pocket:+.2f}")
    save.update(tol_radii_099=tol99, pocket_atoms=pocket, pocket_corr=r_pocket,
                frac_high=np.array([fp["frac_high"][p] for p in range(N_SITES)]))

    print("\n[1] BChl 8 trimer-entry role …")
    b8 = analysis_bchl8(n_grid)
    plot_bchl8(b8)
    print(f"    BChl8 ETE range — bystander: {b8['range_std']:.3f}  "
          f"as entry: {b8['range_b8']:.3f}")
    print(f"    native ETE — standard entry: {b8['nat_std'][0]:.4f}  "
          f"BChl8 entry: {b8['nat_b8'][0]:.4f} (τ {b8['nat_b8'][1]/1000:.2f} ps)")
    save.update(b8_range_std=b8["range_std"], b8_range_entry=b8["range_b8"],
                b8_ete_std=b8["ete_std"], b8_ete_entry=b8["ete_b8"], b8_us=b8["us"])

    print("\n[3] Out-of-plane tolerance …")
    oop = analysis_out_of_plane(n=31 if args.quick else 41)
    plot_out_of_plane(oop)
    def _tol1d(e, shifts):
        below = np.abs(shifts)[e < 0.99]
        return float(below.min()) if below.size else np.inf
    for p in oop["normal"]:
        e_n, e_p = oop["normal"][p], oop["inplane"][p]
        print(f"    BChl{p+1}: tol@0.99 in-plane={_tol1d(e_p, oop['shifts']):.1f}Å  "
              f"normal={_tol1d(e_n, oop['shifts']):.1f}Å  "
              f"(min ETE in-plane={e_p.min():.3f} normal={e_n.min():.3f})")

    print("\n[4] Anatomy of the optimum …")
    opt = analysis_optimum(p4)
    plot_optimum(opt, p4)
    print(f"    site-energy shifts (cm⁻¹): "
          f"{np.array2string(opt['dE'], precision=0, separator=',')}")
    print(f"    exciton band {opt['band_nat']:.0f} → {opt['band_opt']:.0f} cm⁻¹")
    save.update(opt_dE=opt["dE"], band_nat=opt["band_nat"], band_opt=opt["band_opt"])

    print("\n[6] Noise robustness …")
    nz = analysis_noise(p4, n_samples=n_samples)
    plot_noise(nz)
    i50 = int(np.where(nz["sigmas"] == 0.5)[0][0])
    for name in ("native", "optimum"):
        q = nz[name]
        print(f"    {name:<8}: σ=0.5Å median ETE = {q['p50'][i50]:.4f} "
              f"[p10 {q['p10'][i50]:.4f}, p90 {q['p90'][i50]:.4f}]")
    save.update(noise_sigmas=nz["sigmas"],
                noise_native_mean=nz["native"]["mean"], noise_native_std=nz["native"]["std"],
                noise_native_p10=nz["native"]["p10"], noise_native_p50=nz["native"]["p50"],
                noise_native_p90=nz["native"]["p90"],
                noise_opt_mean=nz["optimum"]["mean"], noise_opt_std=nz["optimum"]["std"],
                noise_opt_p10=nz["optimum"]["p10"], noise_opt_p50=nz["optimum"]["p50"],
                noise_opt_p90=nz["optimum"]["p90"])

    np.savez(RESULTS_DIR / "p7_impact_data.npz", **save)
    print(f"\nSaved {RESULTS_DIR / 'p7_impact_data.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
