"""
run_all.py — orchestrate the FMO energy-transfer pipeline.

Two modes share the same phase scripts:

  Standard (default) — regenerate every figure and data file at full resolution.
      python run_all.py
      python run_all.py --quick           # fast low-res run for testing
      python run_all.py --gpu             # enable JAX/CUDA where available
      python run_all.py --refine-level 4  # high-fidelity position-scan refinement

  Deep (--deep) — the "do it properly, overnight" run.  Regenerates the full
  figure set AND runs the two expensive high-fidelity refinements (level-5 PB
  re-optimisation of the arrangement, and the exact-HEOM position scan),
  splitting a wall-clock budget across them.
      python run_all.py --deep --hours 10
      python run_all.py --deep --hours 10 --refine-jobs 16
      python run_all.py --deep --dry-run  # print the plan only

Standard mode FAILS FAST — a broken phase stops the run.  Deep mode is built for
an unattended overnight run: a phase failure is logged and the run CONTINUES to
the remaining phases, and the budgeted phases (phase4 level-5, phase11) are
checkpointed/resumable, so an interruption loses no completed work — just relaunch
with the same command.

Outputs (in results/) — 8-site FMO model
----------------------------------------
    fig1_funnel_dynamics.png    validate.py       — site populations + RC yield
    fig2_fmo_hamiltonian.png    validate.py       — 8x8 H, exciton ladder, τ/site
    fig3_ohmic_vs_vibronic.png  analysis.py       — populations, Ohmic vs vibronic
    fig4_coherence_spectra.png  analysis.py       — exciton coherence FFT
    fig5_position_scan.png      phase4_scan.py    — per-pigment scan + global optimum
    fig6_bath_sensitivity.png   phase5_sensitivity.py — ETE/τ vs bath params
    fig7_summary.png            phase6_summary.py — 6-panel summary
    fig8_sensitivity_fingerprint.png  phase7_impact.py — tolerance radius + pocket
    fig9_bchl8_trimer.png       phase7_impact.py — BChl 8 inter-monomer entry role
    fig10_out_of_plane.png      phase7_impact.py — in-plane vs out-of-plane tolerance
    fig11_optimum_anatomy.png   phase7_impact.py — site-energy/coupling shifts at opt
    fig12_noise_robustness.png  phase7_impact.py — native vs optimum under jitter
    fig13_trimer.png            phase8_trimer.py — 24-site trimer, BChl 8 bridge
    fig14_heom_validation.png   phase9_validation.py — HEOM vs secular Redfield
    fig15_disorder.png          phase9_validation.py — sensitivity under disorder
    fig16_transport_mechanism.png  phase10 — coherent vs incoherent transport
    fig17_environment_robustness.png phase10 — correlated disorder + λ range
    fig18_structured_bath.png   phase10 — Adolphs–Renger 180 cm⁻¹ mode (HEOM)
    fig19_heom_deep.png         phase11_heom_deep.py — exact-HEOM position scan (deep only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).parent


def _hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s//3600}h{(s%3600)//60:02d}m"


def run_phase(script: str, extra: list[str], log: list, fail_fast: bool) -> bool:
    """Run a phase script as a subprocess.  In fail_fast mode a non-zero exit
    aborts the whole run; otherwise the failure is logged and execution continues."""
    cmd = [sys.executable, str(HERE / script)] + extra
    tqdm.write(f"\n{'='*64}\n  {script}  {' '.join(extra)}\n{'='*64}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    dt = time.time() - t0
    ok = rc == 0
    log.append((script, ok, dt))
    if ok:
        tqdm.write(f"  [done in {dt/60:.1f} min]")
    else:
        tqdm.write(f"  [FAILED (exit {rc}) after {dt/60:.1f} min]")
        if fail_fast:
            sys.exit(rc)
    return ok


# ── Standard run ──────────────────────────────────────────────────────────────

def run_standard(args) -> None:
    base = ["--quick"] if args.quick else []
    refine = (["--refine-level", str(args.refine_level),
               "--refine-jobs", str(args.refine_jobs),
               "--refine-hours", str(args.refine_hours)] if args.refine_level else [])

    phases = [
        ("validate.py",            base),
        ("analysis.py",            base),
        ("phase4_scan.py",         base + refine),
        ("phase5_sensitivity.py",  base),
        ("phase6_summary.py",      []),     # loads cached data; no --quick needed
        ("phase7_impact.py",       base),   # impact analyses (needs phase4 output)
        ("phase8_trimer.py",       []),     # full 24-site trimer; BChl 8 bridge
        ("phase9_validation.py",   base),   # HEOM cross-check + static disorder
        ("phase10_refinements.py", base),   # transport/bath/disorder refinements
    ]

    t_start = time.time()
    log: list = []
    bar = tqdm(
        phases, unit="phase",
        bar_format="{desc} {bar} {n_fmt}/{total_fmt} phases  [{elapsed}<{remaining}]",
        ncols=72,
    )
    for script, extra in bar:
        bar.set_description(f"{script:<24}")
        run_phase(script, extra, log, fail_fast=True)
    bar.close()

    print(f"\nAll phases complete in {(time.time()-t_start)/60:.1f} min.")
    print(f"Figures saved to: {HERE / 'results'}/")


# ── Deep (overnight) run ──────────────────────────────────────────────────────

def run_deep(args) -> None:
    total_s = args.hours * 3600.0
    refine_hours = round(args.hours * args.refine_frac, 2)

    fast_pre = ["validate.py", "analysis.py"]                       # before phase4
    fast_post = ["phase5_sensitivity.py", "phase6_summary.py",
                 "phase7_impact.py", "phase8_trimer.py",
                 "phase9_validation.py", "phase10_refinements.py"]  # after phase4

    print("=== run_all.py --deep — full-depth regeneration ===")
    print(f"  total budget : {args.hours:.1f} h")
    print("  plan:")
    print(f"    1. fast full-res figures (figs 1–4): {', '.join(fast_pre)}")
    print(f"    2. phase4_scan.py --refine-level 5 --refine-hours {refine_hours} "
          f"--refine-jobs {args.refine_jobs}   (fig 5, refined scan)")
    print(f"    3. fast full-res figures (figs 6–18): {', '.join(fast_post)}")
    print(f"    4. phase11_heom_deep.py --hours <remaining> --grid {args.heom_grid} "
          f"(fig 19, exact-HEOM scan)")
    if args.dry_run:
        print("\n  [dry run — nothing executed]")
        return

    t_start = time.time()
    log: list = []

    # 1. fast phases that precede the position scan
    for s in fast_pre:
        run_phase(s, [], log, fail_fast=False)

    # 2. deep refinement #1 — level-5 PB re-optimisation (budgeted + checkpointed)
    run_phase("phase4_scan.py",
              ["--refine-level", "5", "--refine-hours", str(refine_hours),
               "--refine-jobs", str(args.refine_jobs)], log, fail_fast=False)

    # 3. fast phases that depend on the scan
    for s in fast_post:
        run_phase(s, [], log, fail_fast=False)

    # 4. deep refinement #2 — exact HEOM scan gets ALL remaining wall-clock
    remaining_h = max(0.5, (total_s - (time.time() - t_start)) / 3600.0)
    print(f"\n  {_hms(time.time()-t_start)} elapsed; "
          f"handing {remaining_h:.2f} h to the exact-HEOM scan.")
    run_phase("phase11_heom_deep.py",
              ["--hours", f"{remaining_h:.2f}", "--grid", str(args.heom_grid)],
              log, fail_fast=False)

    # summary
    total = time.time() - t_start
    print(f"\n{'='*64}\n  DEEP RUN COMPLETE in {_hms(total)}\n{'='*64}")
    for script, ok, dt in log:
        print(f"    {'✓' if ok else '✗'} {script:<26} {dt/60:6.1f} min")
    n_fail = sum(1 for _, ok, _ in log if not ok)
    print(f"  {len(log)-n_fail}/{len(log)} phases succeeded.  "
          f"Figures + data in {HERE/'results'}/")
    if n_fail:
        print(f"  NOTE: {n_fail} phase(s) failed — see the log above; "
              f"checkpointed phases can be resumed by re-running this command.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Run the energy_transfer pipeline")
    p.add_argument("--deep", action="store_true",
                   help="overnight full-depth run (level-5 refine + exact-HEOM scan)")
    # standard-mode options
    p.add_argument("--quick", action="store_true", help="low-res fast run (standard mode)")
    p.add_argument("--gpu", action="store_true",
                   help="GPU via JAX — each script enables it automatically if available")
    p.add_argument("--refine-level", type=int, choices=[1, 2, 3, 4, 5], default=0,
                   help="high-fidelity refinement (OpenMM relax + APBS polarization) of "
                        "the position scan: 1=key points (<1 min) … 4=fine heatmaps "
                        "(~45 min); 5=ULTRA re-optimise under PB objective (~5-6 h)")
    p.add_argument("--refine-hours", type=float, default=5.0,
                   help="wall-clock budget (hours) for the level-5 re-optimisation")
    # deep-mode options
    p.add_argument("--hours", type=float, default=10.0,
                   help="total wall-clock budget for --deep")
    p.add_argument("--refine-jobs", type=int, default=None,
                   help="parallel APBS workers for refinement "
                        "(default 8; 16 in --deep mode for the level-5 re-optimisation)")
    p.add_argument("--refine-frac", type=float, default=0.40,
                   help="fraction of the --deep budget for the level-5 re-optimisation")
    p.add_argument("--heom-grid", type=int, default=31,
                   help="HEOM sensitivity-scan grid for --deep (capped by remaining budget)")
    p.add_argument("--dry-run", action="store_true", help="print the --deep plan only")
    args = p.parse_args()
    if args.refine_jobs is None:
        args.refine_jobs = 16 if args.deep else 8

    if args.deep:
        run_deep(args)
    else:
        run_standard(args)


if __name__ == "__main__":
    main()
