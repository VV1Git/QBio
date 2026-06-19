"""
TrEsp (transition charges from electrostatic potential) excitonic couplings.

This replaces the point-dipole approximation for the inter-pigment couplings with
the literal method of Madjet, Abdurahman & Renger, J. Phys. Chem. B 110, 17268
(2006): each BChl a Qy transition density is represented by atomic transition
charges q_k(1,0) (their Supporting Information, Table II, TDDFT/B3LYP column),
placed at the macrocycle heavy-atom positions, and

    J_mn = C · Σ_{k∈m} Σ_{l∈n} q_k q_l / r_kl          [cm⁻¹]

with C = e²/(4πε₀) = 116140 cm⁻¹·Å.  The transition charges are rescaled so that
the transition-dipole magnitude matches the value behind the published FMO
Hamiltonian (|μ|² = 29.8 D², Klinger et al. 2020), so TrEsp and the published
couplings are on the same dipole-strength footing.

The paper showed (their Fig. 7) that for FMO the PDA already reproduces TrEsp at
all rotation angles; using the real transition charges here removes the
point-dipole approximation entirely from the coupling model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from fmo_data import DIPOLE_STRENGTH_D2

_C_CM = 116140.0                      # e²/(4πε₀) in cm⁻¹·Å
_D_PER_EANG = 4.803                   # 1 e·Å in Debye

_HERE = Path(__file__).parent
_t = np.load(_HERE / "data" / "bchl_tresp_charges.npz", allow_pickle=True)
_TRESP_ATOMS = [str(a) for a in _t["atoms"]]
_Q_RAW = _t["q_trans_b3lyp"].astype(float)          # (47,) raw TDDFT/B3LYP

_a = np.load(_HERE / "fmo_atoms.npz", allow_pickle=True)
_PIG_XYZ = _a["pig_xyz"]                              # (8, 84, 3)
_NAMES = [str(x) for x in _a["atom_names"]]

# Map the 47 TrEsp atoms onto the stored pigment-atom coordinates
_sel = [_NAMES.index(nm) for nm in _TRESP_ATOMS]
TRESP_XYZ = _PIG_XYZ[:, _sel, :]                     # (8, 47, 3) native heavy atoms

# Rescale transition charges so |μ| = sqrt(29.8 D²) (consistent with published H)
_mu_raw = _D_PER_EANG * np.linalg.norm((_Q_RAW[:, None] * TRESP_XYZ[0]).sum(0))
Q_TRANS = _Q_RAW * (np.sqrt(DIPOLE_STRENGTH_D2) / _mu_raw)
N_AT = len(Q_TRANS)


_N_PIG = TRESP_XYZ.shape[0]
_TRIU = np.triu_indices(_N_PIG, 1)        # 28 unique pigment pairs
_QQ = np.outer(Q_TRANS, Q_TRANS)          # (47,47) transition-charge products


def coupling_tresp(displacements: np.ndarray | None = None) -> np.ndarray:
    """
    TrEsp coupling matrix J_ij (cm⁻¹, zero diagonal) for rigid pigment
    displacements (8,3); None → native geometry.

    Only the 28 distinct off-diagonal pairs are summed (atoms of different
    pigments never coincide, so no self-distance singularity).  Distances use
    |a−b|² = |a|²+|b|²−2a·b to avoid a large 4-D temporary.
    """
    pos = TRESP_XYZ if displacements is None else TRESP_XYZ + displacements[:, None, :]
    mi, ni = _TRIU
    A, B = pos[mi], pos[ni]                                      # (28,47,3) each
    a2 = (A ** 2).sum(-1)[:, :, None]                            # (28,47,1)
    b2 = (B ** 2).sum(-1)[:, None, :]                            # (28,1,47)
    d2 = a2 + b2 - 2.0 * np.einsum("pkd,pld->pkl", A, B)         # (28,47,47)
    Jp = _C_CM * np.einsum("kl,pkl->p", _QQ, 1.0 / np.sqrt(d2))  # (28,)
    J = np.zeros((_N_PIG, _N_PIG))
    J[mi, ni] = Jp
    J[ni, mi] = Jp
    return J


TRESP_NATIVE = coupling_tresp()           # native-geometry TrEsp couplings


if __name__ == "__main__":
    from fmo_data import PUBLISHED_COUPLINGS_CM
    from hamiltonian import _PDA_NATIVE
    J = coupling_tresp()
    tri = np.triu_indices(8, 1)
    rms_pub = np.sqrt(np.mean((J[tri] - PUBLISHED_COUPLINGS_CM[tri]) ** 2))
    rms_pda = np.sqrt(np.mean((J[tri] - _PDA_NATIVE[tri]) ** 2))
    mu = _D_PER_EANG * np.linalg.norm((Q_TRANS[:, None] * TRESP_XYZ[0]).sum(0))
    print(f"TrEsp transition dipole |μ| = {mu:.2f} D  (target {np.sqrt(DIPOLE_STRENGTH_D2):.2f})")
    print(f"TrEsp vs published couplings: RMS = {rms_pub:.1f} cm⁻¹")
    print(f"TrEsp vs point-dipole (PDA) : RMS = {rms_pda:.1f} cm⁻¹  "
          f"(Madjet 2006: PDA≈TrEsp for FMO)")
    print(f"  J12: TrEsp={J[0,1]:+.1f}  PDA={_PDA_NATIVE[0,1]:+.1f}  "
          f"published={PUBLISHED_COUPLINGS_CM[0,1]:+.1f} cm⁻¹")
