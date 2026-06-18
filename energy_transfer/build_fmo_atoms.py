"""
One-time builder: assemble the atomistic electrostatic environment of FMO
monomer A into fmo_atoms.npz, used by electrostatics.py for the CDC site-energy
recomputation.

Inputs (vendored in data/):
    data/fmo.pdb                    geometry-optimised FMO trimer (Zenodo 4110066)
    data/BChla.dat                  BChl a ground/excited partial charges (Zenodo)
    data/charmm36_prot_charges.json CHARMM36 protein partial charges (parsed from
                                    top_all36_prot.rtf; THN/GLT use THR/GLU fallback)

Output: fmo_atoms.npz
    prot_xyz (Np,3), prot_q (Np,)          protein point charges (monomer A)
    pig_xyz  (8,Na,3)                       BChl atom coordinates (CDC atom subset)
    pig_qg   (Na,), pig_dq (Na,)            ground charge and Δq = excited−ground
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
SEG = "FMOA"
BCHL_RESIDS = [360, 361, 362, 363, 364, 365, 366, 367]   # BChl 1..8


def _load_bchl_charges():
    names, qg, qe = [], [], []
    for line in open(DATA / "BChla.dat"):
        p = line.split()
        if len(p) >= 3 and p[0] != "ATOM":
            names.append(p[0]); qg.append(float(p[1])); qe.append(float(p[2]))
    return names, np.array(qg), np.array(qe)


def main():
    prot_chg = json.load(open(DATA / "charmm36_prot_charges.json"))
    names, qg, qe = _load_bchl_charges()
    name_set = set(names)

    prot_xyz, prot_q = [], []
    pig_atoms = {r: {} for r in BCHL_RESIDS}   # resid -> {atomname: xyz}

    for L in open(DATA / "fmo.pdb"):
        if L[:4] != "ATOM" and L[:6] != "HETATM":
            continue
        if L[72:76].strip() != SEG:
            continue
        rn = L[17:20].strip(); an = L[12:16].strip()
        xyz = (float(L[30:38]), float(L[38:46]), float(L[46:54]))
        if rn in ("BCA", "BCE"):
            ri = int(L[22:26])
            if ri in pig_atoms and an in name_set:
                pig_atoms[ri][an] = xyz
        elif rn in ("HOH", "TIP3"):
            continue
        else:
            q = prot_chg.get(rn, {}).get(an, 0.0)
            prot_xyz.append(xyz); prot_q.append(q)

    prot_xyz = np.array(prot_xyz); prot_q = np.array(prot_q)

    # pigment atoms in BChla.dat order, for all 8 pigments
    pig_xyz = np.zeros((8, len(names), 3))
    for i, ri in enumerate(BCHL_RESIDS):
        for k, nm in enumerate(names):
            pig_xyz[i, k] = pig_atoms[ri][nm]

    dq = qe - qg
    np.savez(HERE / "fmo_atoms.npz",
             prot_xyz=prot_xyz, prot_q=prot_q,
             pig_xyz=pig_xyz, pig_qg=qg, pig_dq=dq,
             atom_names=np.array(names))

    print(f"protein atoms: {len(prot_q)},  net protein charge: {prot_q.sum():+.2f} e")
    print(f"pigment atoms/BChl: {len(names)},  Δq net per pigment: {dq.sum():+.4f} e "
          f"(should be ~0)")
    print(f"Saved {HERE/'fmo_atoms.npz'}")


if __name__ == "__main__":
    main()
