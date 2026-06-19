"""
High-fidelity refinement of transfer efficiency, combining the two point-only
fixes for a given geometry:

  * OpenMM protein relaxation (fix #2)         — openmm_relax.py
  * APBS Poisson-Boltzmann polarization (fix #1) — apbs_polarization.py

refined site energies:
    ε(d) = ε_published + [PB_protein(d) − PB_protein(native)]
                       + [pigment-pigment(d) − pigment-pigment(native)]
on the relaxed protein, with literal TrEsp couplings; anchored so d=0 reproduces
the published Hamiltonian exactly.

Each evaluation does one APBS PB solve (≈3 s at dime 97); the protein relaxation
is done once and cached.  PB solves are embarrassingly parallel, so grid-level
refinement is run across CPU cores with joblib.

Precision levels (LEVELS): higher = more points and/or a finer APBS grid.
Calibrated to finish within ~1 h on a 24-core machine with REFINE_JOBS workers.

Honest limitation: the relaxation minimises the protein to its own force field
(BChl pigments lack AMBER ligand parameters), so it does not yet relax the
protein *around* a moved pigment — see README note on BChl parameters.
"""
from __future__ import annotations
import numpy as np
from joblib import Parallel, delayed

from electrostatics import (PIG_XYZ, PIG_DQ, PIG_QG, PROT_XYZ, PROT_RESID,
                            PROT_NAME, DIELECTRIC, _C_CM, N_PIG)
from hamiltonian import build_electronic_H
from dynamics import compute_ete
from apbs_polarization import pb_protein_shift, scalar_protein_shift

REFINE_JOBS = 8                       # parallel APBS workers (each spawns apbs)

# level -> (per-pigment grid size or None for "key points only", APBS dime)
LEVELS = {
    1: (None, 97),    # native + global optimum + 8 per-pigment optima  (~10 pts, <1 min)
    2: (9,    97),    # coarse per-pigment heatmaps  (8×81 ≈ 650 pts, ~5 min)
    3: (15,   97),    # medium per-pigment heatmaps  (8×225 ≈ 1800 pts, ~12 min)
    4: (21,   129),   # fine per-pigment heatmaps + finer PB grid (8×441, ~45 min)
    5: (25,   129),   # "ultra": RE-OPTIMISE under the PB-polarized objective +
}                     #          high-res heatmaps + full-relax final optimum (~5-6 h)

_relax_cache: dict = {}
_ref_cache: dict = {}


def _relaxed_prot(disp) -> np.ndarray:
    """Relax the protein AROUND the displaced pigments; coords aligned to PROT order."""
    disp = np.asarray(disp, float)
    key = tuple(np.round(disp.ravel(), 3))
    if key not in _relax_cache:
        from openmm_relax import relax_protein_around, _ATOM_RENAME
        cmap = relax_protein_around(disp)
        xyz = PROT_XYZ.copy()
        for i, (rid, nm) in enumerate(zip(PROT_RESID, PROT_NAME)):
            nm = str(nm)
            k = (int(rid), nm)
            if k not in cmap:
                k = (int(rid), _ATOM_RENAME.get(nm, nm))
            if k in cmap:
                xyz[i] = cmap[k]
        _relax_cache[key] = xyz
    return _relax_cache[key]


def relaxed_protein_xyz() -> np.ndarray:
    """Protein relaxed around the native pigments (geometry-independent ref for grids)."""
    return _relaxed_prot(np.zeros((N_PIG, 3)))


def _pigpig_shift(pig_pos: np.ndarray) -> np.ndarray:
    """Scalar-ε pigment-pigment contribution to each site energy (8,), cm⁻¹."""
    out = np.empty(N_PIG)
    for m in range(N_PIG):
        others = [n for n in range(N_PIG) if n != m]
        env = np.concatenate([pig_pos[n] for n in others])
        envq = np.concatenate([PIG_QG for _ in others])
        d = pig_pos[m][:, None, :] - env[None, :, :]
        inv = 1.0 / np.sqrt((d * d).sum(-1))
        out[m] = _C_CM / DIELECTRIC * (PIG_DQ @ (inv @ envq))
    return out


def _native_refs(prot, polarize, dime):
    key = (id(prot), polarize, dime)
    if key not in _ref_cache:
        prot_fn = pb_protein_shift if polarize else scalar_protein_shift
        _ref_cache[key] = (prot_fn(prot, PIG_XYZ, dime=dime), _pigpig_shift(PIG_XYZ))
    return _ref_cache[key]


def _ete_from_disp(disp, prot, ps_nat, pp_nat, polarize, dime):
    prot_fn = pb_protein_shift if polarize else scalar_protein_shift
    pig_geo = PIG_XYZ + disp[:, None, :]
    shift = (prot_fn(prot, pig_geo, dime=dime) - ps_nat) + (_pigpig_shift(pig_geo) - pp_nat)
    return compute_ete(build_electronic_H(disp, site_shift=shift))


def refine_ete(displacements, relax=True, polarize=True, dime=97):
    """
    Refined (ETE, trapping time fs) for a single geometry, with the protein
    relaxed AROUND the displaced pigments (relax=True) and PB-polarized site
    energies (polarize=True).  Anchored so d=0 → published Hamiltonian.
    """
    disp = np.asarray(displacements, float)
    prot_geo = _relaxed_prot(disp) if relax else PROT_XYZ
    prot_nat = _relaxed_prot(np.zeros((N_PIG, 3))) if relax else PROT_XYZ
    prot_fn = pb_protein_shift if polarize else scalar_protein_shift
    ckey = (relax, polarize, dime)
    if ckey not in _ref_cache:
        _ref_cache[ckey] = (prot_fn(prot_nat, PIG_XYZ, dime=dime), _pigpig_shift(PIG_XYZ))
    ps_nat, pp_nat = _ref_cache[ckey]
    pig_geo = PIG_XYZ + disp[:, None, :]
    shift = (prot_fn(prot_geo, pig_geo, dime=dime) - ps_nat) + (_pigpig_shift(pig_geo) - pp_nat)
    return compute_ete(build_electronic_H(disp, site_shift=shift))


# ── Parallel per-pigment grid refinement ──────────────────────────────────────

def _grid_point(args):
    p, du, dv, e1, e2, prot, ps_nat, pp_nat, polarize, dime = args
    disp = np.zeros((N_PIG, 3))
    disp[p] = du * e1 + dv * e2
    return _ete_from_disp(disp, prot, ps_nat, pp_nat, polarize, dime)


def refine_scan_pigment(p, us, vs, prot, ps_nat, pp_nat, plane_axes,
                        polarize=True, dime=97, n_jobs=REFINE_JOBS):
    """Refined (ete_grid, tau_grid) for one pigment over a 2D in-plane grid (parallel PB)."""
    e1, e2 = plane_axes
    jobs = [(p, u, v, e1, e2, prot, ps_nat, pp_nat, polarize, dime)
            for u in us for v in vs]
    out = Parallel(n_jobs=n_jobs)(delayed(_grid_point)(a) for a in jobs)
    arr = np.array(out).reshape(len(us), len(vs), 2)
    return arr[..., 0], arr[..., 1]


# ── Re-optimisation under the high-fidelity (PB-polarized) objective ──────────
# Used by level 5: instead of refining the fast model's optimum, search the whole
# 16-D in-plane arrangement space with PB-polarized site energies.  This is the
# part that genuinely benefits from a multi-hour budget.

def _refined_objective(x, dime):
    """Mean trapping time (fs) under PB-polarized refinement + steric penalty."""
    from geometry_scan import (disp_from_inplane, min_pair_distance,
                               MIN_SEPARATION_ANG)
    disp = disp_from_inplane(x.reshape(N_PIG, 2))
    _, tau = refine_ete(disp, relax=False, polarize=True, dime=dime)   # PB, no per-eval relax
    dmin = min_pair_distance(disp)
    pen = 1e6 * (MIN_SEPARATION_ANG - dmin) ** 2 if dmin < MIN_SEPARATION_ANG else 0.0
    return tau + pen


def optimize_refined(bound=6.0, maxiter=150, popsize=16, dime=129,
                     workers=REFINE_JOBS, seed=0):
    """
    Global optimisation of the 8-pigment arrangement under the PB-polarized
    objective (parallel APBS workers).  Returns (inplane (8,2), disp (8,3)).
    Each objective eval is one PB solve; differential_evolution parallelises the
    population across `workers` processes.
    """
    from scipy.optimize import differential_evolution
    from geometry_scan import disp_from_inplane
    bounds = [(-bound, bound)] * (2 * N_PIG)
    res = differential_evolution(
        _refined_objective, bounds, args=(dime,), maxiter=maxiter, popsize=popsize,
        workers=workers, updating="deferred", seed=seed, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.7, polish=False)
    inplane = res.x.reshape(N_PIG, 2)
    return inplane, disp_from_inplane(inplane)


if __name__ == "__main__":
    print("refine.py self-test (pigment-aware relax + APBS PB) …")
    e0, t0 = refine_ete(np.zeros((N_PIG, 3)))
    print(f"  native refined: ETE={e0:.4f} τ={t0/1000:.2f} ps (≈ published 0.9956)")
    d = np.zeros((N_PIG, 3)); d[2] = [3.0, 0, 0]
    ef, tf = refine_ete(d)
    print(f"  BChl3 +3Å refined (protein relaxes around it): ETE={ef:.4f} τ={tf/1000:.2f} ps")
