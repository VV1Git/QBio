"""
Fix #1 (first-order CDC → polarization): replace the single scalar dielectric in
the CDC site-energy shift with an explicit Poisson-Boltzmann solve (APBS), so the
protein/solvent dielectric heterogeneity and the reaction field are treated
properly instead of by a constant ε.

Core: pb_site_potentials() solves the linearized PB equation once (protein
charges as sources, all pigments present as zero-charge dielectric cavities) and
returns the screened potential φ at every pigment atom.  The polarization-
corrected protein contribution to the site-energy shift of pigment p is then
ΔE_p = Σ_k Δq_{p,k} φ(r_k).

POINTS-ONLY (a PB solve is ~seconds) — used by refine.py for the native and
optimised geometries, not the dense scan.

Run with the project venv python (needs gridData); APBS is found by absolute
path, so do NOT prepend miniconda to PATH (that swaps in conda's python).

Small self-test:  python apbs_polarization.py
"""
from __future__ import annotations
from pathlib import Path
import subprocess, tempfile, shutil
import numpy as np

from electrostatics import (PIG_XYZ, PIG_DQ, PROT_XYZ, PROT_Q, PROT_NAME,
                            PIG_NAMES, N_PIG, N_AT, DIELECTRIC, _C_CM)

HERE = Path(__file__).parent
APBS = shutil.which("apbs") or str(Path.home() / "miniconda3/bin/apbs")

_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "MG": 1.18}
_KT_TO_CM = 208.5          # k_B T at ~298 K in cm⁻¹ (APBS potential is in kT/e)
PDIE, SDIE = float(DIELECTRIC), 80.0   # interior (matches scalar ε) / solvent


def _radius(name: str) -> float:
    nm = str(name).strip()
    if nm == "MG":
        return _RADII["MG"]
    return _RADII.get(nm[0], 1.7)


_PROT_RAD = np.array([_radius(n) for n in PROT_NAME])
_PIG_RAD = np.array([_radius(n) for n in PIG_NAMES])


def _pqr(prot_xyz, pig_positions) -> str:
    """PQR text: protein atoms (charge+radius) + all pigment atoms (charge 0, cavity)."""
    out = []
    i = 0
    for xyz, q, r in zip(prot_xyz, PROT_Q, _PROT_RAD):
        i += 1
        out.append(f"ATOM  {i:5d}  P   PRO A{i:5d}    "
                   f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f} {q:7.4f}{r:7.4f}")
    for m in range(N_PIG):
        for k in range(N_AT):
            i += 1
            x, y, z = pig_positions[m, k]
            out.append(f"ATOM  {i:5d}  X   PIG B{i:5d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f} {0.0:7.4f}{_PIG_RAD[k]:7.4f}")
    out.append("TER\nEND\n")
    return "\n".join(out)


def _apbs_in(dime: int = 97) -> str:
    return f"""read
    mol pqr sys.pqr
end
elec
    mg-auto
    dime {dime} {dime} {dime}
    cglen 110 110 110
    fglen 70 70 70
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    bcfl sdh
    pdie {PDIE}
    sdie {SDIE}
    srfm smol
    chgm spl2
    srad 1.4
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no
    write pot dx pot
end
quit
"""


def pb_site_potentials(prot_xyz: np.ndarray, pig_positions: np.ndarray,
                       dime: int = 97) -> np.ndarray:
    """
    One PB solve → screened potential φ (kT/e) at every pigment atom, shape (8, Na).

    prot_xyz       : (Np,3) protein atom coordinates (e.g. relaxed)
    pig_positions  : (8,Na,3) pigment atom coordinates (e.g. displaced)
    dime           : APBS grid points per axis (higher = finer/slower)
    """
    from gridData import Grid
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "sys.pqr").write_text(_pqr(prot_xyz, pig_positions))
        (td / "apbs.in").write_text(_apbs_in(dime))
        res = subprocess.run([APBS, "apbs.in"], cwd=td, capture_output=True, text=True)
        dx = td / "pot.dx"
        if not dx.exists():
            raise RuntimeError("APBS failed:\n" + res.stdout[-1500:] + res.stderr[-800:])
        g = Grid(str(dx))
        flat = pig_positions.reshape(-1, 3)
        phi = np.asarray(g.interpolated(flat[:, 0], flat[:, 1], flat[:, 2])).ravel()
    return phi.reshape(N_PIG, N_AT)


def pb_protein_shift(prot_xyz: np.ndarray, pig_positions: np.ndarray,
                     dime: int = 97) -> np.ndarray:
    """Polarization-screened protein contribution to each site energy (8,), cm⁻¹."""
    phi = pb_site_potentials(prot_xyz, pig_positions, dime=dime)   # (8, Na) kT/e
    return (PIG_DQ[None, :] * phi).sum(axis=1) * _KT_TO_CM         # (8,)


def scalar_protein_shift(prot_xyz: np.ndarray, pig_positions: np.ndarray,
                         dime: int = 97) -> np.ndarray:
    """Scalar-ε (current-model) protein contribution to each site energy (8,), cm⁻¹.
    (dime accepted for a uniform interface with pb_protein_shift; unused.)"""
    out = np.empty(N_PIG)
    for m in range(N_PIG):
        d = pig_positions[m][:, None, :] - prot_xyz[None, :, :]
        inv = 1.0 / np.sqrt((d * d).sum(-1))
        out[m] = _C_CM / DIELECTRIC * (PIG_DQ @ (inv @ PROT_Q))
    return out


if __name__ == "__main__":
    print(f"Fix #1 small test: APBS Poisson-Boltzmann polarization (APBS: {APBS})")
    print(f"  pdie={PDIE}, sdie={SDIE} — one PB solve, native geometry …")
    pb = pb_protein_shift(PROT_XYZ, PIG_XYZ)
    sc = scalar_protein_shift(PROT_XYZ, PIG_XYZ)
    for p in (2, 0, 5):
        print(f"  BChl{p+1} protein site-energy: scalar ε={PDIE:.0f} = {sc[p]:+.1f}  "
              f"PB = {pb[p]:+.1f} cm⁻¹")
    print("  → APBS polarization pipeline works; feeds screened shifts into refine.py.")
