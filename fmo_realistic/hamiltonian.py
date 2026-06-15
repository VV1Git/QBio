"""
8-site FMO monomer Hamiltonian from PDB 3EOJ via charge density coupling.

Source: Lorenzoni et al. / Adolphs & Renger 2006.
Site energies and couplings in cm⁻¹.
"""

import numpy as np

# ── Site energies (diagonal, cm⁻¹) ───────────────────────────────────────────

SITE_ENERGIES_CM = np.array([
    12505.0,   # BChl 1
    12425.0,   # BChl 2
    12195.0,   # BChl 3  ← lowest energy, trap site
    12375.0,   # BChl 4
    12600.0,   # BChl 5
    12515.0,   # BChl 6
    12465.0,   # BChl 7
    12700.0,   # BChl 8
])

MEAN_SITE_ENERGY = np.mean(SITE_ENERGIES_CM)   # ≈ 12472.5 cm⁻¹
N_SITES = 8

# ── Electronic couplings (off-diagonal, cm⁻¹) ─────────────────────────────────

_J = np.array([
    [  0.0,  -94.8,    5.5,   -5.9,    7.1,  -15.1,  -12.2,   39.5],
    [-94.8,    0.0,   29.8,    7.6,    1.6,   13.1,    5.7,    7.9],
    [  5.5,   29.8,    0.0,  -58.9,   -1.2,   -9.3,    3.4,    1.4],
    [ -5.9,    7.6,  -58.9,    0.0,  -64.1,  -17.4,  -62.3,   -1.6],
    [  7.1,    1.6,   -1.2,  -64.1,    0.0,   89.5,   -4.6,    4.4],
    [-15.1,   13.1,   -9.3,  -17.4,   89.5,    0.0,   35.1,   -9.1],
    [-12.2,    5.7,    3.4,  -62.3,   -4.6,   35.1,    0.0,  -11.1],
    [ 39.5,    7.9,    1.4,   -1.6,    4.4,   -9.1,  -11.1,    0.0],
])


def build_H_fmo() -> np.ndarray:
    """Return the 8×8 FMO Hamiltonian in cm⁻¹ (full site energies, not shifted)."""
    H = _J.copy()
    np.fill_diagonal(H, SITE_ENERGIES_CM)
    return H


def build_H_shifted() -> np.ndarray:
    """Return the 8×8 Hamiltonian with mean site energy subtracted.

    Subtracting ~12472 cm⁻¹ reduces ODE stiffness in brmesolve without
    affecting any observable (global energy offset is unphysical).
    """
    H = build_H_fmo()
    H -= MEAN_SITE_ENERGY * np.eye(N_SITES)
    return H


if __name__ == "__main__":
    H = build_H_fmo()
    eigvals = np.linalg.eigvalsh(H)
    print("FMO 8-site Hamiltonian")
    print(f"  Site energies: {SITE_ENERGIES_CM}")
    print(f"  Exciton energies (cm⁻¹): {np.sort(eigvals)}")
    print(f"  Exciton bandwidth: {eigvals.max()-eigvals.min():.1f} cm⁻¹")
