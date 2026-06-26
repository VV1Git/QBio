"""
Phase 10: literature-driven refinements and robustness checks.

Addresses the modelling points raised by the FMO literature review:

  (#8) Transport mechanism — compare the coherent secular-Redfield transfer used
       throughout to an incoherent Förster/Marcus site-hopping model.  HEOM
       (phase9) is the exact result that lies between them; if both limits agree
       FMO is highly efficient, the conclusions don't depend on the transport
       picture (an ENAQT-style robustness statement).

  (#5) Correlated disorder — real protein fluctuations are spatially correlated,
       not independent per site.  We draw site-energy disorder with covariance
       Σ_ij = σ² exp(−d_ij/ℓ) and test whether efficiency and the two-tier
       position sensitivity survive a range of correlation lengths ℓ.

  (#4) Reorganisation-energy range — reported FMO λ spans ~35–120 cm⁻¹.  We show
       the sensitive-vs-tolerant (BChl 3 vs BChl 8) distinction holds across it.

  (#3) Structured bath — the Adolphs–Renger spectral density (Drude background +
       resonant 180 cm⁻¹ mode, spectral_density.J_structured) is compared to the
       plain Drude bath with exact HEOM dynamics.

Outputs: fig16_transport_mechanism.png, fig17_environment_robustness.png,
         fig18_structured_bath.png, p10_refinements_data.npz

Usage
-----
    python phase10_refinements.py
    python phase10_refinements.py --quick
    python phase10_refinements.py --no-heom
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from fmo_data import N_SITES, TRAP_SITE, ENTRY_SITES, MG_COORDS_ANG
from hamiltonian import build_electronic_H
from dynamics import compute_ete, _fs_to_cm, K_TRAP_FS, K_LOSS_FS, C_FS
from spectral_density import (LAMBDA_CM, GAMMA_CM, TEMPERATURE_K, J_Ohmic,
                             J_structured, OMEGA_RES_CM, LAMBDA_RES_CM, GAMMA_RES_CM)
from geometry_scan import inplane_to_disp

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BG, PANEL = "#f7f2ea", "#fffdf7"
_KT = 1.4388        # K·cm


def _role(i):
    return "trap→RC" if i == TRAP_SITE else "entry" if i in ENTRY_SITES else "core"


# ── (#8) Incoherent Förster/Marcus transport ──────────────────────────────────

def incoherent_ete(H, initial_sites=ENTRY_SITES, trap_site=TRAP_SITE,
                   lam=LAMBDA_CM, temperature=TEMPERATURE_K,
                   k_trap_fs=K_TRAP_FS, k_loss_fs=K_LOSS_FS):
    """ETE and trapping time from classical Förster/Marcus site-to-site hopping
    (no exciton delocalisation): k_{i→j} = 2π|J_ij|² · G(ΔE_ij), with G the
    Marcus Gaussian Franck–Condon factor at reorganisation energy λ."""
    E = np.diag(H).copy()
    J = H - np.diag(E)
    kT = temperature / _KT
    dE = E[None, :] - E[:, None]                       # dE[i,j] = E_j - E_i
    G = np.exp(-(dE + lam) ** 2 / (4 * lam * kT)) / np.sqrt(4 * np.pi * lam * kT)
    K = 2 * np.pi * (J ** 2) * G                       # rate i→j
    np.fill_diagonal(K, 0.0)
    Kmat = K.T.copy()
    np.fill_diagonal(Kmat, -K.sum(axis=1))
    w = np.zeros(N_SITES); w[trap_site] = _fs_to_cm(k_trap_fs)
    Kt = Kmat - _fs_to_cm(k_loss_fs) * np.eye(N_SITES) - np.diag(w)
    etes, taus = [], []
    for s in initial_sites:
        P0 = np.zeros(N_SITES); P0[s] = 1.0
        m1 = np.linalg.solve(Kt, P0); m2 = np.linalg.solve(Kt, m1)
        y = float(w @ -m1); etes.append(y)
        taus.append((w @ m2) / y / (2 * np.pi * C_FS) if y > 0 else np.inf)
    return float(np.mean(etes)), float(np.mean(taus))


def analysis_transport(quick) -> dict:
    H = build_electronic_H()
    lams = np.linspace(5, 150, 12 if quick else 30)
    coh = np.array([compute_ete(H, lambda_=l) for l in lams])
    inc = np.array([incoherent_ete(H, lam=l) for l in lams])
    return {"lams": lams, "coh_ete": coh[:, 0], "coh_tau": coh[:, 1] / 1000,
            "inc_ete": inc[:, 0], "inc_tau": inc[:, 1] / 1000,
            "coh_phys": compute_ete(H), "inc_phys": incoherent_ete(H)}


def plot_transport(t) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
    for ax, key, ylab, title in [
            (axes[0], "ete", "ETE (yield)", "Yield is robust to transport mechanism"),
            (axes[1], "tau", "trapping time (ps)", "Transfer time vs bath coupling")]:
        ax.plot(t["lams"], t["coh_" + key], "-o", color="#1a6e8c", lw=2, ms=4,
                label="coherent (secular Redfield)")
        ax.plot(t["lams"], t["inc_" + key], "-s", color="#d45f1e", lw=2, ms=4,
                label="incoherent (Förster/Marcus)")
        ax.axvline(LAMBDA_CM, color="#444", ls=":", lw=1.2)
        ax.text(LAMBDA_CM + 2, ax.get_ylim()[0], " physiological λ", fontsize=8, color="#444")
        ax.set_xlabel("reorganisation energy λ (cm⁻¹)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.18, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Fig. 16: Coherent vs incoherent transport — FMO efficiency does not "
                 "depend on the transport picture\n(HEOM, fig14, is the exact result "
                 "bracketed by these two limits)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "fig16_transport_mechanism.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── (#5) Correlated disorder + (#4) reorganisation-energy range ───────────────

def _correlated_disorder(sigma, ell, n, rng):
    """n draws of site-energy disorder with covariance σ² exp(−d_ij/ℓ)."""
    d = np.linalg.norm(MG_COORDS_ANG[:, None] - MG_COORDS_ANG[None], axis=-1)
    if ell <= 0:
        cov = sigma ** 2 * np.eye(N_SITES)
    else:
        cov = sigma ** 2 * np.exp(-d / ell)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(N_SITES))
    return (L @ rng.standard_normal((N_SITES, n))).T          # (n,8)


def analysis_environment(quick, sigma=80.0) -> dict:
    rng = np.random.default_rng(1)
    n = 200 if quick else 500
    H = build_electronic_H()
    ells = [0.0, 5.0, 15.0, 1e6]               # uncorrelated → fully correlated
    dist = {}
    for ell in ells:
        de = _correlated_disorder(sigma, ell, n, rng)
        dist[ell] = np.array([compute_ete(H + np.diag(de[k]))[0] for k in range(n)])

    # (#4) reorganisation-energy range: worst-case ETE over directions at radius
    # 3 Å for BChl3 (sensitive) vs BChl8 (tolerant) — its genuinely worst spot.
    lam_vals = [35.0, 75.0, 120.0]
    dirs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    sens, tol = {}, {}
    for lam in lam_vals:
        s3 = min(compute_ete(build_electronic_H(
            inplane_to_disp(TRAP_SITE, 3 * np.cos(a), 3 * np.sin(a))), lambda_=lam)[0]
            for a in dirs)
        s8 = min(compute_ete(build_electronic_H(
            inplane_to_disp(7, 3 * np.cos(a), 3 * np.sin(a))), lambda_=lam)[0]
            for a in dirs)
        sens[lam], tol[lam] = s3, s8
    return {"sigma": sigma, "ells": ells, "dist": dist,
            "lam_vals": lam_vals, "sens": sens, "tol": tol,
            "ete0": compute_ete(H)[0]}


def plot_environment(e) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    ax = axes[0]
    labels = {0.0: "uncorrelated (ℓ=0)", 5.0: "ℓ=5 Å", 15.0: "ℓ=15 Å",
              1e6: "fully correlated"}
    colors = ["#1a6e8c", "#1a9e4b", "#e0a020", "#c8531e"]
    for (ell, d), c in zip(e["dist"].items(), colors):
        ax.hist(d, bins=25, histtype="step", lw=2, color=c,
                label=f"{labels[ell]}: {d.mean():.4f}±{d.std():.4f}")
    ax.axvline(e["ete0"], color="#444", ls=":", lw=1.2, label="no disorder")
    ax.set_xlabel("ETE", fontsize=10); ax.set_ylabel("count", fontsize=10)
    ax.set_title(f"Spatially-correlated disorder (σ={e['sigma']:.0f} cm⁻¹)\n"
                 "positive correlations tighten the distribution", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = np.arange(len(e["lam_vals"]))
    ax.bar(x - 0.2, [e["sens"][l] for l in e["lam_vals"]], 0.4,
           label="BChl 3 (sensitive)", color="#c8531e", edgecolor="black", lw=0.5)
    ax.bar(x + 0.2, [e["tol"][l] for l in e["lam_vals"]], 0.4,
           label="BChl 8 (tolerant)", color="#1a9e4b", edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"λ={l:.0f}" for l in e["lam_vals"]], fontsize=9)
    ax.set_xlabel("reorganisation energy (cm⁻¹)", fontsize=10)
    ax.set_ylabel("worst-case ETE at 3 Å displacement", fontsize=10)
    ax.set_title("Two-tier sensitivity holds across the λ range\n"
                 "(BChl 3 collapses in its worst direction, BChl 8 never)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 17: Robustness to correlated disorder and to the "
                 "reorganisation-energy uncertainty", fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig17_environment_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


# ── (#3) Structured bath: spectral density + HEOM dynamics ────────────────────

def _heom_pops(H, init, times_fs, structured, depth=3, Nk=1):
    import qutip as qt
    from qutip.solver.heom import HEOMSolver, DrudeLorentzBath, UnderDampedBath
    Hq = qt.Qobj(H - np.mean(np.diag(H)) * np.eye(N_SITES))
    T_cm = TEMPERATURE_K / _KT
    baths = []
    for i in range(N_SITES):
        Q = qt.ket2dm(qt.basis(N_SITES, i))
        baths.append(DrudeLorentzBath(Q, LAMBDA_CM, GAMMA_CM, T_cm, Nk=Nk))
        if structured:
            baths.append(UnderDampedBath(Q, lam=LAMBDA_RES_CM, gamma=GAMMA_RES_CM,
                                         w0=OMEGA_RES_CM, T=T_cm, Nk=Nk))
    solver = HEOMSolver(Hq, baths, max_depth=depth, options={"nsteps": 30000})
    res = solver.run(qt.ket2dm(qt.basis(N_SITES, init)), times_fs * 2 * np.pi * C_FS)
    return np.array([[float(s[i, i].real) for i in range(N_SITES)] for s in res.states])


def analysis_structured(quick) -> dict:
    w = np.linspace(1, 600, 600)
    init = ENTRY_SITES[0]
    times = np.linspace(0, 1200, 40 if quick else 70)
    H = build_electronic_H()
    depth = 2 if quick else 3
    P_drude = _heom_pops(H, init, times, structured=False, depth=depth)
    P_struct = _heom_pops(H, init, times, structured=True, depth=depth)
    return {"w": w, "J_drude": J_Ohmic(w), "J_struct": J_structured(w),
            "times": times, "P_drude": P_drude, "P_struct": P_struct, "init": init}


def plot_structured(s) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    ax = axes[0]
    ax.plot(s["w"], s["J_drude"], color="#1a6e8c", lw=2, label="Drude only (λ=35)")
    ax.plot(s["w"], s["J_struct"], color="#c8531e", lw=2,
            label=f"structured (Drude + {OMEGA_RES_CM:.0f} cm⁻¹ mode)")
    ax.axvline(OMEGA_RES_CM, color="#a33", ls=":", lw=1)
    ax.text(OMEGA_RES_CM + 6, ax.get_ylim()[1] * 0.5, "180 cm⁻¹\n(resonant)",
            fontsize=8, color="#a33")
    ax.set_xlabel("frequency ω (cm⁻¹)", fontsize=10)
    ax.set_ylabel("spectral density J(ω) (cm⁻¹)", fontsize=10)
    ax.set_title("Adolphs–Renger structured spectral density", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    t = s["times"] / 1000
    _C = plt.cm.viridis(np.linspace(0, 0.92, N_SITES))
    for i in (s["init"], TRAP_SITE):
        ax.plot(t, s["P_drude"][:, i], color=_C[i], lw=2,
                label=f"BChl {i+1} Drude")
        ax.plot(t, s["P_struct"][:, i], color=_C[i], lw=2, ls="--",
                label=f"BChl {i+1} structured")
    ax.set_xlabel("time (ps)", fontsize=10); ax.set_ylabel("site population", fontsize=10)
    ax.set_title("HEOM dynamics: the resonant mode\nmodulates entry/sink transfer",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.18, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 18: Structured (Adolphs–Renger) bath with the resonant "
                 "180 cm⁻¹ mode", fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig18_structured_bath.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


def main() -> None:
    import argparse
    from gpu_utils import setup_gpu
    setup_gpu()
    parser = argparse.ArgumentParser(description="Phase 10: refinements")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-heom", action="store_true")
    args = parser.parse_args()

    print("=== Phase 10: literature-driven refinements ===\n")
    save = {}

    print("[#8] Coherent vs incoherent transport …")
    t = analysis_transport(args.quick)
    plot_transport(t)
    print(f"    physiological: coherent ETE={t['coh_phys'][0]:.4f} τ={t['coh_phys'][1]/1000:.2f} ps  |  "
          f"incoherent ETE={t['inc_phys'][0]:.4f} τ={t['inc_phys'][1]/1000:.2f} ps")
    save.update(lams=t["lams"], coh_tau=t["coh_tau"], inc_tau=t["inc_tau"],
                coh_ete=t["coh_ete"], inc_ete=t["inc_ete"])

    print("\n[#5/#4] Correlated disorder + reorganisation-energy range …")
    e = analysis_environment(args.quick)
    plot_environment(e)
    for ell, d in e["dist"].items():
        tag = "uncorr" if ell == 0 else "full-corr" if ell > 1e5 else f"ℓ={ell:.0f}Å"
        print(f"    {tag:<10}: ETE {d.mean():.4f} ± {d.std():.4f}")
    for lam in e["lam_vals"]:
        print(f"    λ={lam:.0f}: BChl3+3Å ETE={e['sens'][lam]:.3f}  "
              f"BChl8+3Å ETE={e['tol'][lam]:.3f}")
    save.update(env_ells=np.array(e["ells"]),
                env_means=np.array([e["dist"][k].mean() for k in e["ells"]]),
                env_stds=np.array([e["dist"][k].std() for k in e["ells"]]),
                lam_vals=np.array(e["lam_vals"]),
                sens=np.array([e["sens"][l] for l in e["lam_vals"]]),
                tol=np.array([e["tol"][l] for l in e["lam_vals"]]))

    if not args.no_heom:
        print("\n[#3] Structured bath (HEOM, slow) …")
        try:
            s = analysis_structured(args.quick)
            plot_structured(s)
            print(f"    final sink pop — Drude {s['P_drude'][-1, TRAP_SITE]:.3f}  "
                  f"structured {s['P_struct'][-1, TRAP_SITE]:.3f}")
            save.update(sd_w=s["w"], sd_J_drude=s["J_drude"], sd_J_struct=s["J_struct"],
                        sd_times=s["times"], sd_P_drude=s["P_drude"],
                        sd_P_struct=s["P_struct"])
        except Exception as ex:
            print(f"    [structured-bath HEOM skipped: {ex}]")

    np.savez(RESULTS_DIR / "p10_refinements_data.npz", **save)
    print(f"\nSaved {RESULTS_DIR / 'p10_refinements_data.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
