"""
Fix #2 (static-protein approximation): relax the protein with OpenMM so that,
for a displaced-pigment geometry, the surrounding protein can reorganise before
the site energies / couplings are recomputed.

This is a POINTS-ONLY refinement — an energy minimisation is seconds–minutes per
geometry, so it is meant for the native structure and a handful of optima, not
the dense 13k-point scan.

Pigment force-field caveat: the BChl a (BCA/BCE) pigments are non-standard
ligands without AMBER templates.  For this small test the protein (monomer A,
standard residues) is built and minimised with amber14; extending to an
all-atom relaxation that moves the pigments too needs BChl ligand parameters
(CGenFF / OpenFF), which is the remaining piece for production use.

Run a quick self-test:  python openmm_relax.py
"""
from __future__ import annotations
from pathlib import Path
import tempfile

HERE = Path(__file__).parent
DATA = HERE / "data"
SEG = "FMOA"
# CHARMM residue names in the optimized structure → standard names for AMBER
_RENAME = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "THN": "THR", "GLT": "GLN"}
# N-terminal acetyl-cap atoms on THN (drop so PDBFixer builds a clean N-terminus)
_DROP_ATOMS = {"CAY", "CY", "OY", "HY1", "HY2", "HY3"}
# C-terminal carboxylate oxygens: CHARMM OT1/OT2 → AMBER O/OXT
_ATOM_RENAME = {"OT1": "O", "OT2": "OXT"}


def _write_monomer_protein(out_pdb: Path) -> int:
    """Extract monomer-A protein atoms (no BChl/water), standardise names/termini."""
    n = 0
    with open(out_pdb, "w") as o:
        for L in open(DATA / "fmo.pdb"):
            if L[:4] != "ATOM" and L[:6] != "HETATM":
                continue
            if L[72:76].strip() != SEG:
                continue
            rn = L[17:20].strip()
            if rn in ("BCA", "BCE", "HOH", "TIP3"):
                continue
            an = L[12:16].strip()
            if an in _DROP_ATOMS:                       # strip N-terminal acetyl cap
                continue
            an2 = _ATOM_RENAME.get(an, an)              # OT1/OT2 → O/OXT
            rn2 = _RENAME.get(rn, rn)
            atom_field = f" {an2:<3}" if len(an2) < 4 else an2
            line = ("ATOM  " + L[6:12] + atom_field + L[16:17] + f"{rn2:>3}"
                    + " A" + L[22:])                    # force chain A
            o.write(line)
            n += 1
        o.write("END\n")
    return n


def relax_protein(max_iterations: int = 0, write_relaxed: Path | None = None) -> dict:
    """
    Build the FMO monomer-A protein in OpenMM (amber14) and energy-minimise it.

    max_iterations=0 runs to tolerance; pass a small number for a quick test.
    Returns energies (kJ/mol) before/after and the max heavy-atom displacement (Å).
    """
    from pdbfixer import PDBFixer
    import openmm
    from openmm import unit
    from openmm.app import ForceField, Modeller, NoCutoff, HBonds, PDBFile
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "monoA.pdb"
        natoms_raw = _write_monomer_protein(raw)

        fixer = PDBFixer(filename=str(raw))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        ff = ForceField("amber14-all.xml")
        modeller = Modeller(fixer.topology, fixer.positions)
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                 constraints=HBonds)
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        sim = openmm.app.Simulation(modeller.topology, system, integrator,
                                    openmm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(modeller.positions)

        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
        p0 = np.array(sim.context.getState(getPositions=True)
                      .getPositions().value_in_unit(unit.angstrom))
        sim.minimizeEnergy(maxIterations=max_iterations)
        e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
        p1 = np.array(sim.context.getState(getPositions=True)
                      .getPositions().value_in_unit(unit.angstrom))

        if write_relaxed is not None:
            with open(write_relaxed, "w") as fh:
                PDBFile.writeFile(modeller.topology,
                                  sim.context.getState(getPositions=True).getPositions(), fh)

        # heavy-atom map (resSeq, atom name) -> relaxed xyz (Å), for feeding CDC/APBS
        coords = {}
        for atom, xyz in zip(modeller.topology.atoms(), p1):
            if atom.element is not None and atom.element.symbol != "H":
                coords[(int(atom.residue.id), atom.name)] = xyz

        return {
            "n_atoms_built": modeller.topology.getNumAtoms(),
            "n_protein_atoms_raw": natoms_raw,
            "E_before_kJ": e0.value_in_unit(unit.kilojoule_per_mole),
            "E_after_kJ": e1.value_in_unit(unit.kilojoule_per_mole),
            "max_disp_A": float(np.linalg.norm(p1 - p0, axis=1).max()),
            "rms_disp_A": float(np.sqrt((np.linalg.norm(p1 - p0, axis=1) ** 2).mean())),
            "coords": coords,
        }


# Per-element Lennard-Jones (sigma nm, epsilon kJ/mol), AMBER/GAFF-like — gives the
# pigment its excluded volume so the protein relaxes *around* it.
_ELEM_LJ = {"C": (0.339, 0.359), "N": (0.325, 0.711), "O": (0.296, 0.879),
            "H": (0.106, 0.066), "S": (0.356, 1.046), "MG": (0.155, 3.74)}


def _elem(name: str) -> str:
    nm = str(name).strip()
    return "MG" if nm == "MG" else nm[0]


def relax_protein_around(displacements, max_iterations: int = 0) -> dict:
    """
    Relax the protein AROUND the (displaced) pigments: each pigment's 84 CDC
    macrocycle atoms are added as frozen point charges (BChla.dat ground charges)
    with GAFF-like per-element LJ, at PIG_XYZ + displacement; the protein then
    minimises in their presence.  Returns {(resSeq, atom name): relaxed xyz (Å)}.

    displacements : (8,3) rigid pigment shifts (Å).  The 84-atom macrocycle is
    present in all 8 pigments (incl. BChl 8, whose phytyl tail is absent), so
    this is uniform and tail-independent.
    """
    from pdbfixer import PDBFixer
    import openmm
    from openmm import unit
    from openmm.app import ForceField, Modeller, NoCutoff, HBonds, Element
    import numpy as np
    from electrostatics import PIG_XYZ, PIG_QG, PIG_NAMES, N_PIG, N_AT

    disp = np.asarray(displacements, float)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "monoA.pdb"
        _write_monomer_protein(raw)
        fixer = PDBFixer(filename=str(raw))
        fixer.findMissingResidues(); fixer.findMissingAtoms()
        fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.0)

        ff = ForceField("amber14-all.xml")
        modeller = Modeller(fixer.topology, fixer.positions)
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                 constraints=HBonds)
        n_prot = system.getNumParticles()
        nb = [f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce)][0]

        # extend topology + positions + system with frozen pigment particles
        top = modeller.topology
        pos = list(modeller.positions.value_in_unit(unit.nanometer))
        chain = top.addChain()
        for m in range(N_PIG):
            res = top.addResidue("PIG", chain)
            for k in range(N_AT):
                el = _elem(PIG_NAMES[k])
                top.addAtom(str(PIG_NAMES[k]), Element.getBySymbol(el.capitalize()), res)
                system.addParticle(0.0)                       # mass 0 → frozen
                sig, eps = _ELEM_LJ.get(el, (0.34, 0.36))
                nb.addParticle(float(PIG_QG[k]), sig, eps)
                xyz = (PIG_XYZ[m, k] + disp[m]) * 0.1          # Å → nm
                pos.append(openmm.Vec3(*xyz))

        sim = openmm.app.Simulation(top, system, openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
            openmm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(pos * unit.nanometer)
        sim.minimizeEnergy(maxIterations=max_iterations)
        p1 = np.array(sim.context.getState(getPositions=True)
                      .getPositions().value_in_unit(unit.angstrom))

        coords = {}
        for idx, atom in enumerate(top.atoms()):
            if idx >= n_prot:
                break                                          # skip pigment atoms
            if atom.element is not None and atom.element.symbol != "H":
                coords[(int(atom.residue.id), atom.name)] = p1[idx]
        return coords


if __name__ == "__main__":
    print("Fix #2 small test: OpenMM protein relaxation (200 minimisation steps) …")
    r = relax_protein(max_iterations=200)
    print(f"  built atoms (with H): {r['n_atoms_built']}  (from {r['n_protein_atoms_raw']} heavy)")
    print(f"  energy: {r['E_before_kJ']:.0f} → {r['E_after_kJ']:.0f} kJ/mol "
          f"(Δ = {r['E_after_kJ']-r['E_before_kJ']:.0f})")
    print(f"  atom displacement: max {r['max_disp_A']:.2f} Å, rms {r['rms_disp_A']:.3f} Å")
    print("  → relaxation engine works; coords can be fed back to TrEsp/CDC.")
