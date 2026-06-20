"""
8-site FMO Frenkel exciton Hamiltonian, built from real pigment coordinates.

Unlike the earlier toy two-dimer model (which varied an artificial intra-dimer
distance r and dipole angle theta), this Hamiltonian is constructed directly
from the geometry-optimized positions and Qy transition dipoles of the eight
bacteriochlorophylls of one FMO monomer (see fmo_data.py for sources).

    H_mm = epsilon_m                        (published site energies)
    H_mn = C_DD * kappa_mn / r_mn^3         (point-dipole coupling, m != n)
    kappa_mn = d_m . d_n - 3 (d_m . r_hat_mn)(d_n . r_hat_mn)

The Hamiltonian is anchored to the published values at the native geometry and
both parts respond to a rigid pigment displacement d:

    couplings   J(d) = J_published + [TrEsp(d) − TrEsp(native)]  (tresp.py)
    site energy ε(d) = ε_published + CDC_shift(d)                (electrostatics.py)

so d=0 reproduces the published TrEsp Hamiltonian exactly, the couplings follow
the literal TrEsp transition-charge change (Madjet, Abdurahman & Renger, J. Phys.
Chem. B 110, 17268 (2006), using their Supporting-Information BChl a transition
charges), and the site energies follow the charge-density-coupling (CDC)
electrostatic shift of the moved transition charges against the protein + the
other pigments.  The position scan therefore responds through BOTH couplings and
site energies, with no point-dipole approximation.  (For FMO the TrEsp and PDA
couplings agree to ~5 cm⁻¹ anyway — Madjet 2006 Fig. 7 — so this is fidelity
insurance rather than a large change.)
"""

from __future__ import annotations
import numpy as np

from fmo_data import (
    MG_COORDS_ANG, QY_DIPOLE_UNIT, SITE_ENERGIES_CM, PUBLISHED_H_CM,
    PUBLISHED_COUPLINGS_CM, C_DD, TRAP_SITE, ENTRY_SITES,
)


def coupling_from_geometry(positions: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
    """
    Vectorised point-dipole coupling matrix J_ij (cm^-1) for arbitrary geometry.

    Parameters
    ----------
    positions : (N, 3) Mg coordinates (Angstrom)
    dipoles   : (N, 3) Qy transition-dipole unit vectors

    Returns
    -------
    J : (N, N) symmetric coupling matrix, zero diagonal
    """
    r_vec = positions[None, :, :] - positions[:, None, :]      # (N,N,3)
    r_mag = np.linalg.norm(r_vec, axis=-1)                      # (N,N)
    np.fill_diagonal(r_mag, np.inf)                            # avoid 0-division
    r_hat = r_vec / r_mag[..., None]

    dd     = dipoles @ dipoles.T                               # d_i . d_j
    d_rh_i = np.einsum("ik,ijk->ij", dipoles, r_hat)           # d_i . r_hat_ij
    d_rh_j = np.einsum("jk,ijk->ij", dipoles, r_hat)           # d_j . r_hat_ij
    kappa  = dd - 3.0 * d_rh_i * d_rh_j

    J = C_DD * kappa / r_mag ** 3
    np.fill_diagonal(J, 0.0)
    return J


# Native point-dipole couplings, used to anchor displaced couplings so that the
# native geometry reproduces the published Hamiltonian exactly.
_PDA_NATIVE = coupling_from_geometry(MG_COORDS_ANG, QY_DIPOLE_UNIT)


def build_electronic_H(displacements: np.ndarray | None = None,
                       site_shift: np.ndarray | None = None,
                       cdc: bool = True, anchor: bool = True) -> np.ndarray:
    """
    Build the 8x8 FMO Frenkel Hamiltonian (cm^-1) for a (possibly displaced)
    geometry, anchored to the published Hamiltonian.

        couplings   J(d) = J_published + [PDA(d) − PDA(native)]   (anchor=True)
        site energy ε(d) = ε_published + CDC_shift(d)             (cdc=True)

    so the native geometry (d=0) reproduces the published Hamiltonian exactly and
    the displacement physics is captured at the Coulomb level (point-dipole
    couplings + charge-density-coupling site energies, see electrostatics.py).

    Parameters
    ----------
    displacements : (8, 3) rigid shift of each pigment (Å).  None → native.
    site_shift    : optional precomputed (8,) CDC site-energy shifts (cm⁻¹);
                    if given it is used instead of recomputing (used by the
                    batched position scan).
    cdc           : recompute site energies via CDC when displaced.
    anchor        : anchor couplings to the published values at native geometry.

    Returns
    -------
    H : (8, 8) real symmetric ndarray in cm^-1
    """
    if displacements is None and site_shift is None:
        return PUBLISHED_H_CM.copy()                       # exact native

    from tresp import coupling_tresp, TRESP_NATIVE
    disp = np.zeros((8, 3)) if displacements is None else displacements
    J = coupling_tresp(disp)                               # literal TrEsp couplings
    if anchor:
        J = PUBLISHED_COUPLINGS_CM + (J - TRESP_NATIVE)    # native-exact couplings

    eps = SITE_ENERGIES_CM.astype(float).copy()
    if site_shift is not None:
        eps = eps + site_shift
    elif cdc:
        from electrostatics import delta_site_energies
        eps = eps + delta_site_energies(disp)
    return np.diag(eps) + J


# ── Quick inspect ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fmo_data import validate_hamiltonian
    print(f"PDA vs published RMS: {validate_hamiltonian():.2f} cm^-1\n")

    H = build_electronic_H()
    print("Native 8-site FMO Hamiltonian (cm^-1):")
    np.set_printoptions(precision=1, suppress=True, linewidth=120)
    print(H)
    eig = np.linalg.eigvalsh(H)
    print(f"\nExciton energies (cm^-1): {eig}")
    print(f"Exciton band span: {eig[-1] - eig[0]:.0f} cm^-1")
    print(f"\nTrap site: BChl {TRAP_SITE + 1}   Entry sites: "
          f"BChl {', '.join(str(s + 1) for s in ENTRY_SITES)}")
