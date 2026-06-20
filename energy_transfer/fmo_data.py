"""
Real structural and spectroscopic data for the 8-site FMO complex (one monomer).

Everything in this module is taken from peer-reviewed sources — no fitted or
invented numbers — so that the position-vs-efficiency study downstream rests on
a literature-grounded reference geometry and Hamiltonian.

Sources
-------
* Geometry (Mg coordinates, Qy dipole axes):
  Geometry-optimized FMO trimer of *Chlorobaculum tepidum*, monomer A, from
  Klinger, Lindorfer, Müh & Renger, J. Chem. Phys. 153, 215103 (2020),
  deposited at Zenodo 10.5281/zenodo.4110066 (file fmo.pdb).  Consistent with
  the crystal structure PDB 3ENI (Tronrud, Wen, Gay & Blankenship,
  Photosynth. Res. 100, 79 (2009)), which first resolved the 8th BChl.

* Published exciton Hamiltonian (site energies + couplings):
  same Zenodo deposit (exc_hamiltonian.dat), i.e. the 8x8 monomer block of
  Klinger et al. 2020 / Schmidt am Busch, Müh, El-Amine Madjet & Renger,
  J. Phys. Chem. Lett. 2, 93 (2011), "The Eighth Bacteriochlorophyll Completes
  the Excitation Energy Funnel in the FMO Protein".  Site energies from CDC
  calculations refined against optical spectra; couplings from the TrEsp /
  charge-density-coupling method.

* Effective Qy transition-dipole strength |mu|^2 = 29.8 D^2:
  Klinger et al. 2020 (point-dipole approximation, Appendix C).

* Functional pigment roles:
  Initial excitation enters at BChl 1 and BChl 6 (baseplate/antenna-facing);
  BChl 3 is the reaction-centre-facing pigment that traps the excitation —
  Shabani, Mohseni, Rabitz & Lloyd, Phys. Rev. E 89, 042706 (2014); see also
  Tronrud et al. 2009 for the structural orientation of the complex.

Indexing: arrays are 0-based; "BChl k" (k = 1..8) is row/index k-1.
"""

from __future__ import annotations
import numpy as np

# ── Reference geometry: Mg (central atom) positions, Angstrom ──────────────────
# Monomer A of the geometry-optimized FMO trimer (Zenodo 4110066, fmo.pdb).
MG_COORDS_ANG = np.array([
    [53.120, 58.531, 21.179],   # BChl 1
    [55.611, 54.455, 32.841],   # BChl 2
    [49.558, 43.982, 44.542],   # BChl 3
    [39.137, 41.946, 42.361],   # BChl 4
    [33.751, 47.483, 31.543],   # BChl 5
    [41.444, 47.600, 23.135],   # BChl 6
    [47.912, 43.114, 32.786],   # BChl 7
    [34.720, 27.887, 14.430],   # BChl 8
])

# ── Qy transition-dipole orientations (unit vectors) ──────────────────────────
# Approximated along the NB->ND axis of each bacteriochlorin macrocycle
# (Madjet/Renger convention).  Extracted from the same optimized structure.
# Reproduces the published couplings to ~9 cm^-1 RMS (see validate_hamiltonian()).
QY_DIPOLE_UNIT = np.array([
    [-0.05386,  0.34834, -0.93582],   # BChl 1
    [-0.78433,  0.54938, -0.28810],   # BChl 2
    [-0.91581,  0.09218,  0.39089],   # BChl 3
    [ 0.09400,  0.44812, -0.88902],   # BChl 4
    [-0.71581,  0.69382,  0.07898],   # BChl 5
    [ 0.85591,  0.39700, -0.33137],   # BChl 6
    [ 0.18916, -0.05593, -0.98035],   # BChl 7
    [-0.96569, -0.17946, -0.18770],   # BChl 8
])

# ── Published exciton Hamiltonian, monomer block (cm^-1) ──────────────────────
# Diagonal = site energies; off-diagonal = excitonic couplings.
SITE_ENERGIES_CM = np.array([
    12505.0, 12425.0, 12195.0, 12375.0, 12600.0, 12515.0, 12465.0, 12700.0
])

# Full symmetric 8x8 published coupling matrix (diagonal zeroed); used only to
# validate the point-dipole map and as an optional "exact" reference geometry H.
PUBLISHED_COUPLINGS_CM = np.array([
    [  0.0, -94.8,   5.5,  -5.9,   7.1, -15.1, -12.2,  39.5],
    [-94.8,   0.0,  29.8,   7.6,   1.6,  13.1,   5.7,   7.9],
    [  5.5,  29.8,   0.0, -58.9,  -1.2,  -9.3,   3.4,   1.4],
    [ -5.9,   7.6, -58.9,   0.0, -64.1, -17.4, -62.3,  -1.6],
    [  7.1,   1.6,  -1.2, -64.1,   0.0,  89.5,  -4.6,   4.4],
    [-15.1,  13.1,  -9.3, -17.4,  89.5,   0.0,  35.1,  -9.1],
    [-12.2,   5.7,   3.4, -62.3,  -4.6,  35.1,   0.0, -11.1],
    [ 39.5,   7.9,   1.4,  -1.6,   4.4,  -9.1, -11.1,   0.0],
])

PUBLISHED_H_CM = np.diag(SITE_ENERGIES_CM) + PUBLISHED_COUPLINGS_CM

# ── Point-dipole coupling constant ────────────────────────────────────────────
# J_mn = C_DD * kappa_mn / r_mn^3  [cm^-1], with |mu|^2 = 29.8 D^2 (Klinger 2020)
# and 1 D^2 Ang^-3 = 5034.15 cm^-1 (e^2/(4 pi eps0 h c) in Gaussian CGS).
DIPOLE_STRENGTH_D2 = 29.8
C_DD = 5034.15 * DIPOLE_STRENGTH_D2     # cm^-1 Ang^3

# ── Functional pigment roles (0-based indices) ────────────────────────────────
N_SITES   = 8
TRAP_SITE = 2     # BChl 3 -> reaction centre  (Shabani et al. 2014)
ENTRY_SITES = (0, 5)   # BChl 1 and BChl 6 -> baseplate/antenna (Shabani et al. 2014)


def validate_hamiltonian() -> float:
    """
    Build the point-dipole coupling matrix from the reference coordinates and
    dipoles, compare to the published couplings, and return the RMS error
    (cm^-1).  Self-check that the position->coupling map is faithful.
    """
    pos, dip = MG_COORDS_ANG, QY_DIPOLE_UNIT
    r_vec = pos[None, :, :] - pos[:, None, :]
    r_mag = np.linalg.norm(r_vec, axis=-1)
    np.fill_diagonal(r_mag, np.inf)
    r_hat = r_vec / r_mag[..., None]
    dd     = dip @ dip.T
    d_rh_i = np.einsum("ik,ijk->ij", dip, r_hat)
    d_rh_j = np.einsum("jk,ijk->ij", dip, r_hat)
    kappa  = dd - 3.0 * d_rh_i * d_rh_j
    J = C_DD * kappa / r_mag ** 3
    np.fill_diagonal(J, 0.0)
    triu = np.triu_indices(N_SITES, k=1)
    return float(np.sqrt(np.mean((J[triu] - PUBLISHED_COUPLINGS_CM[triu]) ** 2)))


if __name__ == "__main__":
    rms = validate_hamiltonian()
    print(f"PDA vs published couplings: RMS = {rms:.2f} cm^-1")
    print("Mean Mg-Mg nearest-neighbour distance: ", end="")
    pos = MG_COORDS_ANG
    d = np.linalg.norm(pos[None] - pos[:, None], axis=-1)
    np.fill_diagonal(d, np.inf)
    print(f"{d.min():.1f} Ang (closest pair)")
