"""
Run all simulation phases in order and regenerate every figure and .npy file.

Usage
-----
    python run_all.py              # CPU, full resolution
    python run_all.py --gpu        # GPU (JAX/CUDA) for vibronic solves
    python run_all.py --quick      # fast low-res run for testing
    python run_all.py --gpu --quick

Outputs (in results/)
---------------------
    fig1_dynamics.png        validate.py  — P4(t) traces at 3 geometries
    fig2a_reff_heatmap.png   validate.py  — Reff(r,θ) heatmap (Ohmic)
    fig3_p4_comparison.png   analysis.py  — Ohmic vs vibronic P4(t)
    fig4_coherence_spectra.png analysis.py — exciton coherence FFT
    fig5_reff_comparison.png phase4_scan.py — full Reff(r,θ) scan
    fig6_bath_sensitivity.png phase5_sensitivity.py — bath parameter sweep
    fig7_summary.png         phase6_summary.py — 6-panel summary

    r_grid.npy / theta_grid.npy / reff_ohmic.npy   (Phase 1)
    p4_r_grid.npy / p4_theta_grid.npy              (Phase 4)
    p4_reff_ohmic.npy / p4_reff_vibronic.npy       (Phase 4)
    p5_sensitivity_data.npz                         (Phase 5)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  {script}  {' '.join(extra_args)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  [done in {elapsed/60:.1f} min]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all energy_transfer phases")
    parser.add_argument("--quick", action="store_true", help="Low-res fast run")
    args = parser.parse_args()

    base = ["--quick"] if args.quick else []

    t_start = time.time()

    run("validate.py",             base)
    run("analysis.py",             base)
    run("phase4_scan.py",          base)
    run("phase5_sensitivity.py",   base)
    run("phase6_summary.py",       [])         # loads cached data, no --quick needed

    total = (time.time() - t_start) / 60
    print(f"\nAll phases complete in {total:.1f} min.")
    print(f"Figures saved to: {HERE / 'results'}/")


if __name__ == "__main__":
    main()
