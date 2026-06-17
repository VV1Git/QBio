"""
4-site dimerized Frenkel Hamiltonian — Ai et al. geometry.

Geometry
--------
Two parallel dimers arranged along the x-axis:

  Site 1  ----r----  Site 2          Site 3  ----r----  Site 4
  (donor dimer)                      (acceptor dimer)
  |<------------- R = 40 Å ----------------->|
  (centre-to-centre of the two dimers)

  Site positions (x, y, z):
    1: (-r/2,  0, 0)
    2: (+r/2,  0, 0)
    3: (R - r/2, 0, 0)
    4: (R + r/2, 0, 0)

  All four dipoles point at angle θ from the x-axis:
    d̂_i = (cos θ, sin θ, 0)

Point-dipole coupling (Gaussian CGS → cm⁻¹):

    J_mn = (d² / R_mn³) × κ_mn  [cm⁻¹]

where the prefactor for d = 7.75 D converts as
    C_dd  [cm⁻¹ Å³] = 5034.15 × d [D]²
                     (derived from e²/(4πε₀ hc) in CGS)

and the orientation factor is
    κ_mn = d̂_m · d̂_n − 3 (d̂_m · R̂_mn)(d̂_n · R̂_mn)

Site energies (cm⁻¹):  ε = (13000, 12900, 12300, 12200)
"""

from __future__ import annotations
import numpy as np


# ── Physical constants ────────────────────────────────────────────────────────

SITE_ENERGIES_CM = np.array([13000.0, 12900.0, 12300.0, 12200.0])   # cm⁻¹
DIPOLE_MAGNITUDE_D = 7.75        # Debye
R_INTER_ANG = 40.0               # Å  (dimer-centre to dimer-centre)

# Conversion: d² [D²] / R³ [Å³]  →  cm⁻¹
# From e²/(4πε₀ ħc):  1 D² Å⁻³ = 5034.15 cm⁻¹
_C_DD = 5034.15 * DIPOLE_MAGNITUDE_D ** 2   # cm⁻¹ Å³


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _site_positions(r: float, R: float = R_INTER_ANG) -> np.ndarray:
    """Return (4, 3) array of site positions in Å."""
    return np.array([
        [-r / 2,         0.0, 0.0],
        [ r / 2,         0.0, 0.0],
        [ R - r / 2,     0.0, 0.0],
        [ R + r / 2,     0.0, 0.0],
    ])


def _dipole_orientations(theta: float) -> np.ndarray:
    """Return (4, 3) unit-vector array; all dipoles at angle θ from x-axis."""
    d_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
    return np.tile(d_hat, (4, 1))


# ── Main builder ──────────────────────────────────────────────────────────────

def build_electronic_H(r: float, theta: float) -> np.ndarray:
    """
    Build the 4×4 Frenkel exciton Hamiltonian (cm⁻¹).

    Fully vectorised: the point-dipole couplings J_ij and orientation
    factors κ_ij are computed for all site pairs at once with numpy
    broadcasting (no Python loop over pairs).

    Parameters
    ----------
    r     : intra-dimer distance (Å)
    theta : dipole angle from the inter-dimer (x) axis (radians)

    Returns
    -------
    H : (4, 4) real symmetric ndarray in cm⁻¹
    """
    pos = _site_positions(r)             # (4, 3)
    d   = _dipole_orientations(theta)    # (4, 3) unit vectors

    # Pairwise separation vectors and magnitudes
    r_vec = pos[None, :, :] - pos[:, None, :]          # (4, 4, 3): r_vec[i,j] = pos_j − pos_i
    r_mag = np.linalg.norm(r_vec, axis=-1)             # (4, 4)
    np.fill_diagonal(r_mag, np.inf)                    # avoid 0-division on diagonal
    r_hat = r_vec / r_mag[..., None]

    # Orientation factor κ_ij = d̂_i·d̂_j − 3 (d̂_i·r̂_ij)(d̂_j·r̂_ij).
    # Both projections use the SAME separation direction r̂_ij, so the second
    # factor is d̂_j·r̂_ij (einsum over j's dipole), NOT d_rhat.T (which would
    # use r̂_ji = −r̂_ij and flip the sign).
    dd     = d @ d.T                                    # (4, 4): d̂_i·d̂_j
    d_rh_i = np.einsum("ik,ijk->ij", d, r_hat)          # (4, 4): d̂_i·r̂_ij
    d_rh_j = np.einsum("jk,ijk->ij", d, r_hat)          # (4, 4): d̂_j·r̂_ij
    kappa  = dd - 3.0 * d_rh_i * d_rh_j

    J = _C_DD * kappa / r_mag ** 3                     # (4, 4); diagonal → 0 (r_mag=∞)
    H = np.diag(SITE_ENERGIES_CM.astype(float)) + J
    return H


def coupling_matrix(r: float, theta: float) -> np.ndarray:
    """Return the 4×4 off-diagonal coupling matrix J_ij (cm⁻¹)."""
    H = build_electronic_H(r, theta)
    return H - np.diag(np.diag(H))


# ── Quick inspect ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for r_ang in [8.0, 11.3, 13.4]:
        H = build_electronic_H(r_ang, theta=0.0)
        J = coupling_matrix(r_ang, theta=0.0)
        print(f"\nr = {r_ang} Å, θ = 0")
        print(f"  J12 = {J[0,1]:+.1f}  J23 = {J[1,2]:+.1f}  J34 = {J[2,3]:+.1f}  J13 = {J[0,2]:+.1f}  J24 = {J[1,3]:+.1f}  J14 = {J[0,3]:+.1f} cm⁻¹")
        eigvals = np.linalg.eigvalsh(H)
        print(f"  Eigenvalues: {eigvals}")
