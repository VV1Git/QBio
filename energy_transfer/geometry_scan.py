"""
Position-vs-efficiency engine for the 8-site FMO complex.

The biological question: how does *where each bacteriochlorophyll sits* affect
the excitation-transfer efficiency to the reaction centre, and is the native
arrangement already optimal?

Each pigment is rigidly displaced within the principal plane of the FMO subunit
(the two largest-variance PCA axes of the 8 Mg positions — the flat "disk"; the
thin normal axis is held fixed).  A rigid displacement moves the whole pigment,
so it changes BOTH:
  * the point-dipole couplings (via the Mg/dipole geometry), and
  * the site energies (via the charge-density-coupling shift of the moved
    transition charges against the protein + other pigments — electrostatics.py).
Both are anchored to the published Hamiltonian at the native geometry, so the
native point is exact and the displacement physics is Coulomb-level.

Efficiency is the secular-Redfield ETE with trapping at BChl 3 (compute_ete).

    scan_pigment()         — sweep one pigment over a 2D in-plane grid
    optimize_arrangement() — move all pigments to minimise the trapping time,
                             subject to a minimum-separation (steric) constraint
"""

from __future__ import annotations

import numpy as np

from fmo_data import MG_COORDS_ANG, N_SITES
from hamiltonian import build_electronic_H
from dynamics import compute_ete
from electrostatics import scan_pigment_grid, site_energies_batch

# ── Principal (in-plane) axes of the FMO subunit ──────────────────────────────
CENTROID = MG_COORDS_ANG.mean(axis=0)
_, _, _VT = np.linalg.svd(MG_COORDS_ANG - CENTROID)
PLANE_AXES = _VT[:2]          # (2, 3): e1, e2 span the disk
NORMAL_AXIS = _VT[2]

MIN_SEPARATION_ANG = 9.0      # steric floor (native closest contact ~10.8 Å)


def inplane_to_disp(pigment: int, du: float, dv: float) -> np.ndarray:
    """(8,3) displacement array with only `pigment` shifted by du·e1 + dv·e2."""
    disp = np.zeros((N_SITES, 3))
    disp[pigment] = du * PLANE_AXES[0] + dv * PLANE_AXES[1]
    return disp


def disp_from_inplane(inplane: np.ndarray) -> np.ndarray:
    """(N,2) in-plane shifts → (N,3) Cartesian displacements."""
    return inplane @ PLANE_AXES


def min_pair_distance(displacements: np.ndarray) -> float:
    pos = MG_COORDS_ANG + displacements
    d = np.linalg.norm(pos[None] - pos[:, None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def efficiency(displacements: np.ndarray) -> tuple[float, float]:
    """(ETE, trapping time fs) for a displaced geometry (couplings + CDC)."""
    return compute_ete(build_electronic_H(displacements))


# ── Per-pigment 2D scan (batched CDC) ─────────────────────────────────────────

def scan_pigment(pigment: int, us: np.ndarray, vs: np.ndarray,
                 n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep one pigment over a 2D grid of in-plane displacements.

    Returns (ete_grid, tau_grid), each (len(us), len(vs)); tau in fs.  The CDC
    site-energy shifts for the whole grid are computed in one batched (GPU) call;
    the per-point coupling rebuild + ETE solve are ~100 µs each and run serially
    (cheaper than the joblib dispatch overhead for these tiny tasks).
    """
    U, V = np.meshgrid(us, vs, indexing="ij")
    disps2d = np.stack([U.ravel(), V.ravel()], axis=1)            # (G,2)
    disps3d = disps2d @ PLANE_AXES                                # (G,3) for pigment
    cdc = scan_pigment_grid(pigment, disps3d)                     # (G,8) site shifts

    out = np.empty((disps3d.shape[0], 2))
    full = np.zeros((N_SITES, 3))
    for g in range(disps3d.shape[0]):
        full[:] = 0.0; full[pigment] = disps3d[g]
        out[g] = compute_ete(build_electronic_H(full, site_shift=cdc[g]))
    arr = out.reshape(len(us), len(vs), 2)
    return arr[..., 0], arr[..., 1]


# ── Global optimisation of the whole arrangement ──────────────────────────────

def _objective_vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorised objective for differential_evolution(vectorized=True).

    X : (2*N_SITES, S) population of in-plane displacements.
    Returns (S,) mean trapping times (fs) + steric penalty.  The expensive CDC
    site-energy shifts for the whole population are computed in one batched GPU
    call; the per-member coupling rebuild + ETE solve are cheap.
    """
    S = X.shape[1]
    disp = (X.T.reshape(S, N_SITES, 2)) @ PLANE_AXES          # (S,8,3)
    shifts = site_energies_batch(disp)                        # (S,8) batched GPU
    out = np.empty(S)
    for s in range(S):
        H = build_electronic_H(disp[s], site_shift=shifts[s])
        _, tau = compute_ete(H)
        dmin = min_pair_distance(disp[s])
        pen = 1e6 * (MIN_SEPARATION_ANG - dmin) ** 2 if dmin < MIN_SEPARATION_ANG else 0.0
        out[s] = tau + pen
    return out


def optimize_arrangement(bound: float = 6.0, seed: int = 0,
                         maxiter: int = 60, popsize: int = 16) -> dict:
    """
    Minimise the mean trapping time by moving all 8 pigments in-plane within
    ±`bound` Å of native, respecting the steric floor.  Returns native/optimised
    ETE, tau, the displacements, and the moved positions.
    """
    from scipy.optimize import differential_evolution

    bounds = [(-bound, bound)] * (2 * N_SITES)
    res = differential_evolution(
        _objective_vec, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7, polish=False,
        vectorized=True, updating="deferred",
    )
    inplane = res.x.reshape(N_SITES, 2)
    disp = disp_from_inplane(inplane)
    ete_opt, tau_opt = efficiency(disp)
    ete_nat, tau_nat = efficiency(np.zeros((N_SITES, 3)))
    return {
        "ete_native": ete_nat, "tau_native": tau_nat,
        "ete_opt": ete_opt, "tau_opt": tau_opt,
        "disp_inplane": inplane,
        "shift_mag": np.linalg.norm(inplane, axis=1),
        "pos_native": MG_COORDS_ANG.copy(),
        "pos_opt": MG_COORDS_ANG + disp,
        "min_sep_opt": min_pair_distance(disp),
    }


if __name__ == "__main__":
    print(f"Native min Mg-Mg separation: {min_pair_distance(np.zeros((N_SITES,3))):.2f} Å")
    e, t = efficiency(np.zeros((N_SITES, 3)))
    print(f"Native: ETE={e:.4f}, tau={t/1000:.2f} ps  (should match published-H values)")
    # sanity: a single CDC-aware displacement
    d = inplane_to_disp(2, 3.0, 0.0)
    e2, t2 = efficiency(d)
    print(f"BChl3 moved 3 Å: ETE={e2:.4f}, tau={t2/1000:.2f} ps")
    print("\nOptimising (≈1 min) …")
    out = optimize_arrangement(maxiter=40)
    print(f"  Native    : ETE={out['ete_native']:.4f}  τ={out['tau_native']/1000:.2f} ps")
    print(f"  Optimised : ETE={out['ete_opt']:.4f}  τ={out['tau_opt']/1000:.2f} ps "
          f"(min sep {out['min_sep_opt']:.1f} Å)")
    print(f"  Shifts (Å): {np.round(out['shift_mag'],2)}")
