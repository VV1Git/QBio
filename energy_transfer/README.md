# FMO energy transfer — does pigment *position* set transfer efficiency?

A first-principles model of the **8-site Fenna–Matthews–Olson (FMO)** complex from
green sulfur bacteria, built to answer one question: **how much does each
bacteriochlorophyll's physical position matter for funnelling excitation energy to
the reaction centre?**

Every pigment is displaced rigidly and the excitation-transfer efficiency (ETE) and
trapping time (τ) are recomputed, with both the Hamiltonian's couplings *and* site
energies responding to the move via literature methods (TrEsp couplings, CDC + APBS
Poisson–Boltzmann site energies). The model is anchored to the published FMO
Hamiltonian at the native geometry, so the native arrangement reproduces the
literature exactly (ETE ≈ 0.9956, τ ≈ 4.42 ps) and every displacement is a faithful
perturbation around it.

## Headline findings

- **A two-tier architecture.** The reaction-centre core (BChl 3 and neighbours) is
  geometrically *rigid* — BChl 3 cannot move even ~0.3 Å without losing efficiency —
  while the periphery is *tolerant*, surviving multi-Å displacement. Position matters
  intensely, but only for the core.
- **The eighth bacteriochlorophyll, reframed.** BChl 8's position is irrelevant as a
  bystander (ETE varies by 0.007 across its whole accessible disk) yet decisive when it
  is the excitation *entry* point (range 0.998). In the full 24-site trimer it carries
  the single strongest inter-monomer coupling (+36 cm⁻¹, A-BChl 8 ↔ B-BChl 1), but
  inter-monomer transfer is *redundant*: removing BChl 8's bridge alone changes nothing
  (parallel ~10 cm⁻¹ pathways carry the excitation). The most important bridge is also
  the most expendable — the structural signature of a **robust, redundant, regulatory
  entry interface** rather than a finely-tuned funnel component.
- **Robust to the physics you assume.** The two-tier sensitivity ranking is reproduced
  by numerically exact **HEOM** dynamics (not just the fast secular-Redfield solver),
  survives realistic static disorder (σ = 80 cm⁻¹ → ETE 0.9928 ± 0.0065), holds across
  the reorganisation-energy range (35–120 cm⁻¹), and is insensitive to coherent-vs-
  incoherent transport assumptions.

## Quick start

```bash
pip install -r requirements.txt          # core deps; GPU is optional (see file)
python run_all.py                        # regenerate every figure (~few minutes)
python run_all.py --quick                # fast low-res test run
```

For the high-fidelity overnight run (level-5 Poisson–Boltzmann re-optimisation of the
arrangement + an exact-HEOM position scan, budget-split across a wall-clock limit):

```bash
python run_all.py --deep --hours 10      # checkpointed + resumable
python run_all.py --deep --dry-run       # print the plan only
```

## Repository layout

**Core physics**
| file | role |
|------|------|
| `fmo_data.py`        | published 8-site Hamiltonian, real coordinates, trap/entry sites, bath params |
| `hamiltonian.py`     | builds the (optionally displaced) electronic Hamiltonian |
| `spectral_density.py`| Ohmic and structured (Adolphs–Renger, 180 cm⁻¹ mode) bath spectral densities |
| `dynamics.py`        | secular-Redfield rate matrix, ETE and trapping time |
| `geometry_scan.py`   | in-plane / out-of-plane pigment-displacement helpers |
| `vibronic.py`        | vibronic (resonant-mode) dynamics, with optional JAX/GPU path |
| `gpu_utils.py`       | optional JAX/CUDA setup (silent CPU fallback) |

**First-principles parameterization** (structure → couplings + site energies)
| file | role |
|------|------|
| `build_fmo_atoms.py`  | one-time builder of `fmo_atoms.npz` from `data/` |
| `electrostatics.py`   | charge-density-coupling (CDC) site-energy shifts |
| `apbs_polarization.py`| APBS linearized Poisson–Boltzmann polarization correction |
| `openmm_relax.py`     | OpenMM relaxation of the protein around displaced pigments |
| `tresp.py`            | TrEsp excitonic couplings |
| `refine.py`           | wires relax + PB into the high-fidelity (points/level) refinement |

**Analysis phases** (each produces figures + a data `.npz`; orchestrated by `run_all.py`)
| file | figures |
|------|---------|
| `validate.py`            | fig1 funnel dynamics, fig2 Hamiltonian / exciton ladder |
| `analysis.py`            | fig3 Ohmic-vs-vibronic, fig4 coherence spectra |
| `phase4_scan.py`         | fig5 per-pigment position scan + global optimum |
| `phase5_sensitivity.py`  | fig6 bath-parameter sensitivity |
| `phase6_summary.py`      | fig7 six-panel summary |
| `phase7_impact.py`       | fig8–fig12 (tolerance fingerprint, BChl 8 entry role, out-of-plane, optimum anatomy, noise robustness) |
| `phase8_trimer.py`       | fig13 full 24-site trimer + BChl 8 bridge redundancy |
| `phase9_validation.py`   | fig14 HEOM vs secular Redfield, fig15 static-disorder robustness |
| `phase10_refinements.py` | fig16 transport mechanism, fig17 environment robustness, fig18 structured bath |
| `phase11_heom_deep.py`   | fig19 exact-HEOM position scan (**`--deep` only**) |

`data/` holds the vendored inputs (geometry-optimised `fmo.pdb`, BChl a charges, CHARMM36
protein charges, TrEsp charges); `bchl_params/` holds the GAFF2 force-field outputs for the
flexible bacteriochlorophyll; `results/` holds the generated figures and `.npz` data.

## Method, honestly

Couplings: `J(d) = J_published + [TrEsp(d) − TrEsp(native)]`. Site energies:
`ε(d) = ε_published + CDC_shift(d)`, optionally upgraded to an APBS Poisson–Boltzmann
shift in the refinement. The CDC model uses static protein charges, a single interior
dielectric (ε ≈ 2), point charges, and no charge-transfer/exchange — stated plainly
because these are real approximations. Spectral re-refinement of counterfactual
geometries is impossible (no experimental spectra exist for them), so anchoring to the
published Hamiltonian at the native geometry is the principled handling. The fast
position-scan optimum (ETE 0.996 → 0.998, τ 4.4 → ~2 ps) does **not** survive the
high-fidelity level-5 PB re-optimisation (which lands at ETE ≈ 0.76, τ ≈ 240 ps) — the
exact physics declines to endorse the cheap optimum, consistent with HEOM.

## Future work

- **Redox-switchable quenching at BChl 8.** FMO performs redox-regulated photoprotective
  quenching (cysteines C49/C353 tune an exciton–vibration resonance; Higgins et al.,
  *PNAS* 2021). Adding a gated loss channel at BChl 8 to `dynamics.compute_ete` would turn
  the photoprotection picture above into a quantitative speed-vs-quench trade-off.
- **Macrocycle conformation → site energy.** Pigments are perturbed by *rigid*
  displacement; real site-energy variation also comes from BChl-a macrocycle distortion
  (BChl 8 is notably non-planar). Mapping distortion → transition energy faithfully needs
  per-conformer TDDFT, beyond the present CDC/TrEsp electrostatic model — flagged rather
  than approximated.
- **MD conformational sampling.** A thermal ensemble from all-atom MD with *flexible*
  pigments would sample correlated geometric + energetic fluctuations directly; the
  blocker is a validated full-complex force field (the eight BChls are non-standard
  ligands and the eighth lacks its phytyl tail). The static- and correlated-disorder
  analyses (fig15, fig17) are the tractable proxy and already show the conclusions hold.

## References

Papers and software the model is built on, each with a one-line note on how it was used.
Every FMO-specific scientific paper below (1–11, 16, 17) was checked against the
publisher/PubMed record — authors, journal, volume, pages and year are confirmed (✓, with
DOI). The software/method packages (18–23) and the classic/textbook theory references
(12–15) are standard, well-established citations; confirm their exact bibliographic fields
against the publisher when assembling a formal bibliography.

**Structure, geometry, and the published Hamiltonian**

1. ✓ **Klinger, Lindorfer, Müh & Renger**, "Normal mode analysis of spectral density of
   FMO trimers: intra- and intermonomer energy transfer," *J. Chem. Phys.* **153**, 215103
   (2020). DOI 10.1063/5.0027994. — Geometry-optimised FMO trimer structure (Zenodo
   4110066, `fmo.pdb`): Mg coordinates and Qy dipole axes for all 24 pigments, and the
   reference for the structured (normal-mode) spectral density.
2. ✓ **Schmidt am Busch, Müh, El-Amine Madjet & Renger**, "The Eighth Bacteriochlorophyll
   Completes the Excitation Energy Funnel in the FMO Protein," *J. Phys. Chem. Lett.* **2**,
   93–98 (2011). DOI 10.1021/jz101541b. — The published 8-site exciton Hamiltonian the whole
   model is anchored to; establishes BChl 8 as the baseplate-facing entry pigment.
3. ✓ **Tronrud, Wen, Gay & Blankenship**, "The structural basis for the difference in
   absorbance spectra for the FMO antenna protein from various green sulfur bacteria,"
   *Photosynth. Res.* **100**, 79–87 (2009). — Crystal structure (PDB **3ENI**) that first
   resolved the eighth BChl; basis for the pigment arrangement and trap/entry orientation.

**Coupling and site-energy methods (the model's physics)**

4. ✓ **Madjet, Abdurahman & Renger**, "Intermolecular Coulomb couplings from ab initio
   electrostatic potentials …," *J. Phys. Chem. B* **110**, 17268–17281 (2006). DOI
   10.1021/jp0615398. — The TrEsp method and BChl a transition charges used in `tresp.py`.
5. ✓ **Adolphs & Renger**, "How proteins trigger excitation energy transfer in the FMO
   complex of green sulfur bacteria," *Biophys. J.* **91**, 2778–2797 (2006). — The
   charge-density-coupling (CDC) method (`electrostatics.py`), the ε ≈ 2 dielectric, and the
   structured spectral density with the resonant ~180 cm⁻¹ mode (S ≈ 0.22).
6. ✓ **Müh, Madjet, Adolphs, Abdurahman, Rabenstein, Ishikita, Knapp & Renger**, "α-Helices
   direct excitation energy flow in the Fenna–Matthews–Olson protein," *PNAS* **104**,
   16862–16867 (2007). DOI 10.1073/pnas.0708222104. — Supporting reference for the CDC
   electrostatic origin of the FMO site energies.
7. ✓ **Renger & Marcus**, "On the relation of protein dynamics and exciton relaxation in
   pigment–protein complexes …," *J. Chem. Phys.* **116**, 9997–10019 (2002). — The
   super-Ohmic functional form behind the structured FMO spectral density.

**Bath, spectral density, and vibronic structure**

8. ✓ **Wendling, Pullerits, Przyjalgowski, Vulto, Aartsma, van Grondelle & van Amerongen**,
   "Electron–vibrational coupling in the FMO complex …," *J. Phys. Chem. B* **104**,
   5825–5831 (2000). DOI 10.1021/jp000077+. — Experimental fluorescence-line-narrowing
   spectral density the Adolphs–Renger bath (and `J_structured`) is built on.
9. ✓ **Rätsep & Freiberg**, "Electron–phonon and vibronic couplings in the FMO
   bacteriochlorophyll a antenna complex …," *J. Lumin.* **127**, 251–259 (2007). —
   Intramolecular BChl a vibrational modes used in `vibronic.py`.
10. ✓ **Ishizaki & Fleming**, "Theoretical examination of quantum coherence in a
    photosynthetic system at physiological temperature," *PNAS* **106**, 17255–17260 (2009).
    — FMO bath parameters (λ ≈ 35 cm⁻¹, γ ≈ 53 cm⁻¹, 300 K) and the HEOM benchmark the
    exact-dynamics validation reproduces.

**Dynamics, trapping, and transport mechanism**

11. ✓ **Shabani, Mohseni, Rabitz & Lloyd**, "Numerical evidence for robustness of
    environment-assisted quantum transport," *Phys. Rev. E* **89**, 042706 (2014). — The
    trapping/entry assignment (BChl 3 → reaction centre; entry at BChl 1 & 6) and the
    trapping/loss rates in `dynamics.py`.
12. **Mohseni, Rebentrost, Lloyd & Aspuru-Guzik**, "Environment-assisted quantum walks in
    photosynthetic energy transfer," *J. Chem. Phys.* **129**, 174106 (2008). — Conceptual
    basis for the ENAQT framing of the coherent-vs-incoherent comparison (`phase10`).
13. **Plenio & Huelga**, "Dephasing-assisted transport: quantum networks and biomolecules,"
    *New J. Phys.* **10**, 113019 (2008). — The dephasing-assisted-transport idea behind the
    bath-coupling (λ) dependence of efficiency.
14. **Förster** (*Ann. Phys.* **437**, 55, 1948) and **Marcus** (*J. Chem. Phys.* **24**,
    966, 1956). — Classical incoherent energy-transfer / electron-transfer rate theory for
    the Förster–Marcus hopping model compared against coherent transport.
15. **Tanimura & Kubo** (*J. Phys. Soc. Jpn.* **58**, 101, 1989); review: **Tanimura**,
    *J. Chem. Phys.* **153**, 020901 (2020). — The HEOM formalism used for the exact-dynamics
    validation and the deep HEOM scan.

**BChl 8 and photoprotection**

16. ✓ **Higgins, Lloyd, Sohail, Allodi, Otto, Saer, Wood, Massey, Ting, Blankenship &
    Engel**, "Photosynthesis tunes quantum-mechanical mixing of electronic and vibrational
    states to steer exciton energy transfer," *PNAS* **118**, e2018240118 (2021). DOI
    10.1073/pnas.2018240118. — Redox-regulated photoprotective quenching in FMO (Cys
    C49/C353); basis for the BChl 8 photoprotection discussion.

**Spatially correlated fluctuations (robustness check)**

17. ✓ **Olbrich, Strümpfer, Schulten & Kleinekathöfer**, "Quest for spatially correlated
    fluctuations in the FMO light-harvesting complex," *J. Phys. Chem. B* **115**, 758–764
    (2011); and **Nalbach, Eckel & Thorwart**, "Quantum coherent biomolecular energy transfer
    with spatially correlated fluctuations," *New J. Phys.* **12**, 065043 (2010). — Motivate
    the correlated-disorder test (`phase10`): spatial correlations tighten the efficiency
    distribution, matching our finding.

**Computational tools and force fields**

18. **Johansson, Nation & Nori**, "QuTiP," *Comput. Phys. Commun.* **183**, 1760 (2012) and
    **184**, 1234 (2013); QuTiP v5 (Lambert et al., 2024). — Open-quantum-system solvers:
    Bloch–Redfield (`brmesolve`) and HEOM (`HEOMSolver`, `DrudeLorentzBath`, `UnderDampedBath`).
19. **Baker, Sept, Joseph, Holst & McCammon**, *PNAS* **98**, 10037 (2001); **Jurrus et al.**,
    "Improvements to the APBS biomolecular solvation software suite," *Protein Sci.* **27**,
    112 (2018). — APBS Poisson–Boltzmann solver for the polarization correction
    (`apbs_polarization.py`).
20. **Eastman et al.**, "OpenMM 7: rapid development of high-performance algorithms for
    molecular dynamics," *PLoS Comput. Biol.* **13**, e1005659 (2017). — Protein
    minimisation / relaxation around displaced pigments (`openmm_relax.py`).
21. **Wang, Wolf, Caldwell, Kollman & Case**, "Development and testing of a general AMBER
    force field," *J. Comput. Chem.* **25**, 1157 (2004). — GAFF/GAFF2 parameters (via
    AmberTools) for the flexible bacteriochlorophyll (`bchl_params/`).
22. **Best, Zhu, Shim, Lopes, Mittal, Feig & MacKerell**, "Optimization of the additive
    CHARMM all-atom protein force field …," *J. Chem. Theory Comput.* **8**, 3257 (2012). —
    CHARMM36 protein partial charges for the electrostatic environment (`build_fmo_atoms.py`).
23. **Maier et al.**, "ff14SB: improving the accuracy of protein side-chain and backbone
    parameters …," *J. Chem. Theory Comput.* **11**, 3696 (2015). — AMBER ff14SB protein
    force field used by OpenMM in the relaxation step.
