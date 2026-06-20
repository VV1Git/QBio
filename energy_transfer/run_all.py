"""
Run all simulation phases in order and regenerate every figure and .npy file.

Usage
-----
    python run_all.py              # CPU, full resolution
    python run_all.py --gpu        # attempt GPU (JAX/CUDA) for QuTiP solvers
    python run_all.py --quick      # fast low-res run for testing
    python run_all.py --gpu --quick

Outputs (in results/) — 8-site FMO model
----------------------------------------
    fig1_funnel_dynamics.png    validate.py       — site populations + RC yield
    fig2_fmo_hamiltonian.png    validate.py       — 8x8 H, exciton ladder, τ/site
    fig3_ohmic_vs_vibronic.png  analysis.py       — populations, Ohmic vs vibronic
    fig4_coherence_spectra.png  analysis.py       — exciton coherence FFT (fixed)
    fig5_position_scan.png      phase4_scan.py    — per-pigment scan + global optimum
    fig6_bath_sensitivity.png   phase5_sensitivity.py — ETE/τ vs bath params
    fig7_summary.png            phase6_summary.py — 6-panel summary

    fmo_hamiltonian.npy            (Phase 1)
    p4_position_scan.npz           (Phase 4: per-pigment grids + optimisation)
    p5_sensitivity_data.npz        (Phase 5)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).parent


def run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script)] + extra_args
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {script}  {' '.join(extra_args)}")
    tqdm.write(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        tqdm.write(f"\n[ERROR] {script} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    tqdm.write(f"  [done in {elapsed/60:.1f} min]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all energy_transfer phases")
    parser.add_argument("--quick", action="store_true", help="Low-res fast run")
    parser.add_argument("--gpu",   action="store_true",
                        help="GPU via JAX — each script enables it automatically if available")
    parser.add_argument("--refine-level", type=int, choices=[1, 2, 3, 4, 5], default=0,
                        help="High-fidelity refinement (OpenMM relax + APBS polarization) "
                             "of the position scan: 1=key points (<1 min) … 4=fine heatmaps "
                             "(~45 min); 5=ULTRA re-optimise under PB objective (~5-6 h). "
                             "Points/grid only, not the full dense scan.")
    parser.add_argument("--refine-jobs", type=int, default=8,
                        help="Parallel APBS workers for refinement (use 16 for level 5)")
    parser.add_argument("--refine-hours", type=float, default=5.0,
                        help="Wall-clock budget (hours) for the level-5 re-optimisation")
    args = parser.parse_args()

    base = ["--quick"] if args.quick else []
    refine = (["--refine-level", str(args.refine_level),
               "--refine-jobs", str(args.refine_jobs),
               "--refine-hours", str(args.refine_hours)] if args.refine_level else [])

    phases = [
        ("validate.py",           base),
        ("analysis.py",           base),
        ("phase4_scan.py",        base + refine),
        ("phase5_sensitivity.py", base),
        ("phase6_summary.py",     []),     # loads cached data; no --quick needed
    ]

    t_start = time.time()
    bar = tqdm(
        phases, unit="phase",
        bar_format="{desc} {bar} {n_fmt}/{total_fmt} phases  [{elapsed}<{remaining}]",
        ncols=72,
    )
    for script, extra in bar:
        bar.set_description(f"{script:<24}")
        run(script, extra)
    bar.close()

    total = (time.time() - t_start) / 60
    print(f"\nAll phases complete in {total:.1f} min.")
    print(f"Figures saved to: {HERE / 'results'}/")


if __name__ == "__main__":
    main()
