"""
Phase 8: Full FMO trimer (24-site) analysis — the rigorous test of BChl 8's
inter-monomer role that the monomer model (phase7_impact.py) could only proxy.

The 24-site trimer model is built in this file (it has a single consumer — this
analysis — so model and figure live together).

Findings:
  * Structural — BChl 8 carries the single strongest inter-monomer coupling
    (A-BChl8 ↔ B-BChl1 ≈ +36 cm⁻¹), confirming it is the physical bridge between
    monomers.
  * Functional — inter-monomer transfer is ESSENTIAL (removing all inter-monomer
    couplings sends the cross-monomer yield to zero) but REDUNDANT: removing only
    BChl 8's bridges leaves the yield and transfer time unchanged, because several
    parallel ~10 cm⁻¹ pathways carry the excitation.  The trimer's inter-monomer
    connectivity is fault-tolerant, echoing the native arrangement's noise
    robustness (phase7, fig12).

Output: fig13_trimer.png, p8_trimer_data.npz

Usage
-----
    python phase8_trimer.py
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
from fmo_data import (SITE_ENERGIES_CM, PUBLISHED_COUPLINGS_CM, C_DD,
                      N_SITES, TRAP_SITE, ENTRY_SITES)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 24-site trimer model ──────────────────────────────────────────────────────
# Geometry (Mg positions + Qy NB→ND dipole axes) for all 24 pigments is read from
# the geometry-optimised trimer structure (data/fmo.pdb, segments FMOA/B/C); the
# NB→ND axis reproduces the published monomer-A dipoles exactly.  Site energies are
# the published 8 values tiled over the three C3-symmetric monomers.  Intra-monomer
# coupling blocks are the published (anchored) matrix; inter-monomer blocks are the
# point-dipole couplings from the real geometry (no published inter-monomer matrix
# exists, and PDA reproduces the intra couplings to ~9 cm⁻¹).
_HERE = Path(__file__).parent
_SEGS = ("FMOA", "FMOB", "FMOC")   # each monomer numbers its 8 BChls independently
N_MONO = 3
N_TRIMER = N_MONO * N_SITES                                # 24


def _extract_geometry() -> tuple[np.ndarray, np.ndarray]:
    """(Mg coords (24,3), Qy dipole units (24,3)) for monomers A,B,C in order."""
    mg = {s: {} for s in _SEGS}
    nb = {s: {} for s in _SEGS}
    nd = {s: {} for s in _SEGS}
    for L in open(_HERE / "data" / "fmo.pdb"):
        if L[:4] != "ATOM" and L[:6] != "HETATM":
            continue
        seg = L[72:76].strip()
        if seg not in _SEGS or L[17:20].strip() not in ("BCA", "BCE"):
            continue
        ri = int(L[22:26])
        an = L[12:16].strip()
        xyz = (float(L[30:38]), float(L[38:46]), float(L[46:54]))
        if an == "MG":
            mg[seg][ri] = xyz
        elif an == "NB":
            nb[seg][ri] = xyz
        elif an == "ND":
            nd[seg][ri] = xyz

    coords, dipoles = [], []
    for s in _SEGS:
        for ri in sorted(mg[s]):                  # 8 BChls in residue order = BChl 1..8
            coords.append(mg[s][ri])
            ax = np.array(nd[s][ri]) - np.array(nb[s][ri])
            dipoles.append(ax / np.linalg.norm(ax))
    return np.array(coords), np.array(dipoles)


MG_COORDS_TRIMER, QY_DIPOLE_TRIMER = _extract_geometry()


def _pda_couplings(positions: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
    """Point-dipole coupling matrix (cm⁻¹, zero diagonal) for any geometry."""
    r = positions[None, :, :] - positions[:, None, :]
    rmag = np.linalg.norm(r, axis=-1)
    np.fill_diagonal(rmag, np.inf)
    rhat = r / rmag[..., None]
    dd = dipoles @ dipoles.T
    dri = np.einsum("ik,ijk->ij", dipoles, rhat)
    drj = np.einsum("jk,ijk->ij", dipoles, rhat)
    kappa = dd - 3.0 * dri * drj
    J = C_DD * kappa / rmag ** 3
    np.fill_diagonal(J, 0.0)
    return J


def build_trimer_H(displacements: np.ndarray | None = None) -> np.ndarray:
    """24×24 trimer Hamiltonian (cm⁻¹).  Intra-monomer blocks = published couplings
    (anchored); inter-monomer blocks = point-dipole couplings from the geometry.

    displacements : optional (24,3) rigid pigment shifts (Å)."""
    pos = MG_COORDS_TRIMER if displacements is None else MG_COORDS_TRIMER + displacements
    J_full = _pda_couplings(pos, QY_DIPOLE_TRIMER)
    J_nat = _pda_couplings(MG_COORDS_TRIMER, QY_DIPOLE_TRIMER)

    H = np.zeros((N_TRIMER, N_TRIMER))
    # intra-monomer blocks: published couplings, anchored (published at native)
    for m in range(N_MONO):
        s = slice(m * N_SITES, (m + 1) * N_SITES)
        H[s, s] = PUBLISHED_COUPLINGS_CM + (J_full[s, s] - J_nat[s, s])
    # inter-monomer blocks: raw point-dipole couplings from the geometry
    for m in range(N_MONO):
        for n in range(N_MONO):
            if m != n:
                sm = slice(m * N_SITES, (m + 1) * N_SITES)
                sn = slice(n * N_SITES, (n + 1) * N_SITES)
                H[sm, sn] = J_full[sm, sn]
    # diagonal: tiled published site energies
    H[np.diag_indices(N_TRIMER)] = np.tile(SITE_ENERGIES_CM, N_MONO)
    return H


def inter_monomer_couplings() -> np.ndarray:
    """(8,8) matrix of A↔B couplings: entry [i,j] = J(BChl i in A, BChl j in B)."""
    J = _pda_couplings(MG_COORDS_TRIMER, QY_DIPOLE_TRIMER)
    return J[0:N_SITES, N_SITES:2 * N_SITES]


# trap stays on monomer-A BChl 3; entry sites map per monomer
TRAP_TRIMER = TRAP_SITE                                  # site 2 (A's BChl3)
ENTRY_B = tuple(N_SITES + s for s in ENTRY_SITES)        # monomer B entries

BG, PANEL = "#f7f2ea", "#fffdf7"
_MONO_COLORS = ["#1a6e8c", "#c8531e", "#1a9e4b"]


def _ablate(H: np.ndarray, which: str) -> np.ndarray:
    """Zero inter-monomer couplings: 'all' or just BChl 8's ('b8')."""
    H2 = H.copy()
    for m in range(N_MONO):
        for n in range(N_MONO):
            if n == m:
                continue
            sm = slice(m * N_SITES, (m + 1) * N_SITES)
            sn = slice(n * N_SITES, (n + 1) * N_SITES)
            if which == "all":
                H2[sm, sn] = 0.0
            elif which == "b8":
                b8 = m * N_SITES + 7
                H2[b8, sn] = 0.0
                H2[sn, b8] = 0.0
    return H2


def redundancy_test(loss_rates_fs=(1e-6, 1e-4, 5e-4)) -> dict:
    """Cross-monomer yield (excite monomer B, trap in monomer A) with the full
    bridge network, with BChl 8's bridges removed, and with all bridges removed."""
    H = build_trimer_H()
    variants = {"full": H, "no BChl 8 bridge": _ablate(H, "b8"),
                "no inter-monomer": _ablate(H, "all")}
    out = {"loss_rates_fs": np.array(loss_rates_fs), "ete": {}, "tau": {}}
    for name, HH in variants.items():
        e_list, t_list = [], []
        for kl in loss_rates_fs:
            e, t = compute_ete(HH, initial_sites=ENTRY_B, trap_site=TRAP_TRIMER,
                               k_loss_fs=kl)
            e_list.append(e); t_list.append(t / 1000.0)
        out["ete"][name] = np.array(e_list)
        out["tau"][name] = np.array(t_list)
    return out


def plot_trimer(Jab: np.ndarray, red: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    # Panel A: inter-monomer (A↔B) coupling map
    ax = axes[0]
    vmax = np.abs(Jab).max()
    im = ax.imshow(Jab, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_SITES)); ax.set_xticklabels(range(1, N_SITES + 1), fontsize=8)
    ax.set_yticks(range(N_SITES)); ax.set_yticklabels(range(1, N_SITES + 1), fontsize=8)
    ax.set_xlabel("BChl in monomer B", fontsize=9)
    ax.set_ylabel("BChl in monomer A", fontsize=9)
    i, j = np.unravel_index(np.argmax(np.abs(Jab)), Jab.shape)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2))
    ax.text(j, i - 1.1, f"{Jab[i,j]:+.0f} cm⁻¹", ha="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("J (cm⁻¹)", fontsize=8)
    ax.set_title(f"Inter-monomer couplings (A↔B)\nstrongest: A-BChl{i+1} ↔ B-BChl{j+1}",
                 fontsize=10)

    # Panel B: 24-pigment geometry, projected onto its principal plane
    ax = axes[1]
    c = MG_COORDS_TRIMER.mean(0)
    _, _, VT = np.linalg.svd(MG_COORDS_TRIMER - c)
    xy = (MG_COORDS_TRIMER - c) @ VT[:2].T
    for m in range(N_MONO):
        s = slice(m * N_SITES, (m + 1) * N_SITES)
        ax.scatter(xy[s, 0], xy[s, 1], s=55, color=_MONO_COLORS[m],
                   edgecolor="white", zorder=3, label=f"monomer {'ABC'[m]}")
        b8 = m * N_SITES + 7
        ax.scatter(xy[b8, 0], xy[b8, 1], s=150, facecolor="none",
                   edgecolor="black", lw=1.6, zorder=4)
    # draw the strongest inter-monomer link in each A→B→C→A interface
    for m in range(N_MONO):
        a8 = m * N_SITES + 7
        nb = ((m + 1) % N_MONO) * N_SITES + 0          # adjacent monomer BChl1
        ax.plot([xy[a8, 0], xy[nb, 0]], [xy[a8, 1], xy[nb, 1]],
                color="black", lw=1.2, ls="--", zorder=2)
    ax.scatter([], [], s=150, facecolor="none", edgecolor="black", label="BChl 8 (bridge)")
    ax.set_aspect("equal")
    ax.set_xlabel("principal axis 1 (Å)", fontsize=9)
    ax.set_ylabel("principal axis 2 (Å)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Trimer geometry — BChl 8 bridges adjacent monomers", fontsize=10)

    # Panel C: redundancy — cross-monomer yield vs which bridges are present
    ax = axes[2]
    loss = red["loss_rates_fs"]
    xlab = [f"{1/(kl*1e3):.0f} ps" if kl >= 1e-4 else "1 ns (physiol.)" for kl in loss]
    x = np.arange(len(loss))
    width = 0.26
    colors = {"full": "#1a9e4b", "no BChl 8 bridge": "#1a6e8c", "no inter-monomer": "#c8531e"}
    for k, (name, e) in enumerate(red["ete"].items()):
        ax.bar(x + (k - 1) * width, e, width, label=name, color=colors[name],
               edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_xlabel("exciton loss time (trap–loss competition)", fontsize=9)
    ax.set_ylabel("cross-monomer ETE\n(excite monomer B → trap in A)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Inter-monomer transfer is essential but redundant\n"
                 "(removing BChl 8's bridge alone changes nothing)", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 13: Full FMO trimer — BChl 8 is the dominant but non-essential "
                 "inter-monomer bridge", fontsize=12, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "fig13_trimer.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")


def main() -> None:
    from gpu_utils import setup_gpu
    setup_gpu()
    print("=== Phase 8: Full FMO trimer (24-site) ===\n")

    Jab = inter_monomer_couplings()
    i, j = np.unravel_index(np.argmax(np.abs(Jab)), Jab.shape)
    print(f"  Strongest inter-monomer coupling: A-BChl{i+1} ↔ B-BChl{j+1} "
          f"= {Jab[i,j]:+.1f} cm⁻¹")
    n_b8 = sum(1 for a in range(N_SITES) for b in range(N_SITES)
               if (a == 7 or b == 7) and abs(Jab[a, b]) > 8)
    print(f"  Inter-monomer couplings > 8 cm⁻¹ involving a BChl 8: {n_b8}")

    H = build_trimer_H()
    eig = np.linalg.eigvalsh(H)
    print(f"  Trimer exciton band: {eig[-1]-eig[0]:.0f} cm⁻¹ ({N_TRIMER} states)")

    print("\n  Redundancy test (excite monomer B, trap in monomer A) …")
    red = redundancy_test()
    for name in red["ete"]:
        e = red["ete"][name]
        print(f"    {name:<18}: ETE@1ns={e[0]:.4f}  ETE@100ps={e[1]:.4f}  "
              f"ETE@20ps={e[2]:.4f}")

    plot_trimer(Jab, red)
    np.savez(RESULTS_DIR / "p8_trimer_data.npz",
             Jab=Jab, exciton_band=float(eig[-1] - eig[0]),
             loss_rates_fs=red["loss_rates_fs"],
             ete_full=red["ete"]["full"],
             ete_no_b8=red["ete"]["no BChl 8 bridge"],
             ete_no_inter=red["ete"]["no inter-monomer"])
    print(f"\nSaved {RESULTS_DIR / 'p8_trimer_data.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
