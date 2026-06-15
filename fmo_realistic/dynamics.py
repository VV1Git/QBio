"""
Open-quantum-system dynamics for the realistic 8-site FMO complex.

Hilbert space: 9-dimensional
    Index 0       : electronic ground state (no excitation)
    Indices 1..8  : single-excitation on BChl 1..8

Hamiltonian   : 8-site FMO (Lorenzoni / Adolphs-Renger), mean-shifted.
Bath          : site-specific structured spectral density (Drude + 10 UBO modes)
                via pre-tabulated interpolation → brmesolve.
Trapping      : Lindblad L_trap = √k_trap |g⟩⟨3|  (BChl 3 → reaction centre)
Recombination : Lindblad L_loss = √k_loss |g⟩⟨m|  for all m

GPU note
--------
Two levels of GPU acceleration are available:

1. Spectral density tables (CuPy):
   Automatically used if `cupy` is importable.  The dense 40 000-point
   gamma tables are built on GPU then transferred to CPU arrays for QuTiP.

2. QuTiP JAX backend (qutip-jax + JAX/CUDA):
   Install: pip install qutip-jax "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   Activate before calling run_fmo():
       import qutip_jax
       import qutip as qt
       qt.settings.core["default_dtype"] = "jaxdia"
   This moves Hilbert-space matrix operations (Redfield tensor construction,
   ODE integration) onto the NVIDIA GPU via XLA/CUDA.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from hamiltonian import build_H_shifted, N_SITES
from spectral_density import (
    build_gamma_tables, make_gamma_interp,
    TEMPERATURE_K,
)

# ── Physical rates ─────────────────────────────────────────────────────────────

C_FS = 3e-5   # cm/fs

def _rate_to_cm(rate_per_fs: float) -> float:
    return rate_per_fs / (2.0 * np.pi * C_FS)

K_TRAP_CM = _rate_to_cm(1.0 / 500.0)        # (0.5 ps)⁻¹  ≈ 10.6 cm⁻¹
K_LOSS_CM = _rate_to_cm(1.0 / 1_000_000.0)  # (1 ns)⁻¹    ≈ 5.3e-3 cm⁻¹

TRAP_SITE = 3    # BChl 3 (1-indexed in the 9D space)
DIM = N_SITES + 1


# ── Builders ───────────────────────────────────────────────────────────────────

def _build_H9() -> qt.Qobj:
    H9 = np.zeros((DIM, DIM))
    H9[1:, 1:] = build_H_shifted()
    return qt.Qobj(H9)


def _build_a_ops(tables: dict) -> list:
    """Bath coupling operators using pre-tabulated spectral functions."""
    a_ops = []
    for m in range(1, DIM):           # BChl 1..8 → 9D indices 1..8
        P9 = np.zeros((DIM, DIM))
        P9[m, m] = 1.0
        A = qt.Qobj(P9)
        gamma_fn = make_gamma_interp(tables, site=m - 1)
        a_ops.append((A, gamma_fn))
    return a_ops


def _build_c_ops() -> list[qt.Qobj]:
    c_ops = []
    L_trap = np.zeros((DIM, DIM))
    L_trap[0, TRAP_SITE] = np.sqrt(K_TRAP_CM)
    c_ops.append(qt.Qobj(L_trap))
    for m in range(1, DIM):
        L = np.zeros((DIM, DIM))
        L[0, m] = np.sqrt(K_LOSS_CM)
        c_ops.append(qt.Qobj(L))
    return c_ops


# ── Main solver ────────────────────────────────────────────────────────────────

def run_fmo(
    init_site: int = 1,
    temperature: float = TEMPERATURE_K,
    t_end_fs: float = 15_000.0,
    n_steps: int = 600,
    sec_cutoff: float = 0.1,
    gamma_tables: dict | None = None,
) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Propagate the 9D FMO density matrix via Bloch-Redfield master equation.

    Parameters
    ----------
    init_site     : initially excited BChl (1-indexed; 1 or 6)
    temperature   : bath temperature in K
    t_end_fs      : propagation window (fs)
    n_steps       : output time-grid points
    sec_cutoff    : secular approximation cutoff (cm⁻¹)
    gamma_tables  : pre-built spectral density tables from build_gamma_tables().
                    If None, tables are built automatically (adds ~5 s startup).
                    Pass the same tables when running multiple cases at the same T.

    Returns
    -------
    times_fs : (n_steps,) array [fs]
    rhos     : list of n_steps 9×9 QuTiP density matrices
    """
    if gamma_tables is None:
        gamma_tables = build_gamma_tables(temperature=temperature)

    H     = _build_H9()
    a_ops = _build_a_ops(gamma_tables)
    c_ops = _build_c_ops()

    rho0_np = np.zeros((DIM, DIM))
    rho0_np[init_site, init_site] = 1.0
    rho0 = qt.Qobj(rho0_np)

    t_max_int = t_end_fs * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)

    result = qt.brmesolve(
        H, rho0, times_int,
        a_ops=a_ops,
        c_ops=c_ops,
        sec_cutoff=sec_cutoff,
        # Use BDF (implicit, A-stable) — handles stiff Redfield equations where
        # UBO-mode peaks create sub-ps decay rates alongside ps-scale population
        # transfer.  LSODA would auto-switch but BDF is more reliable here.
        options={"nsteps": 500_000, "rtol": 1e-6, "atol": 1e-8, "method": "bdf"},
    )

    times_fs = times_int / (2.0 * np.pi * C_FS)
    return times_fs, result.states


# ── Observables ────────────────────────────────────────────────────────────────

def site_population(rhos: list[qt.Qobj], site: int) -> np.ndarray:
    """P_site(t) for BChl `site` (1-indexed) from 9D density matrices."""
    return np.array([float(rho[site, site].real) for rho in rhos])


def ground_population(rhos: list[qt.Qobj]) -> np.ndarray:
    """Total population that has left the single-excitation manifold."""
    return np.array([float(rho[0, 0].real) for rho in rhos])


def compute_ete(times_fs: np.ndarray,
                rhos: list[qt.Qobj],
                k_trap_cm: float = K_TRAP_CM) -> float:
    """
    Energy Transfer Efficiency:
        η = k_trap [fs⁻¹] × ∫₀^∞ ρ₃₃(t) dt [fs]

    Integrates the trap-site population weighted by the trapping rate.
    """
    P3 = site_population(rhos, TRAP_SITE)
    k_trap_fs = k_trap_cm * 2.0 * np.pi * C_FS
    return k_trap_fs * np.trapz(P3, times_fs)


# ── Fast secular ETE (Pauli master equation, no coherences) ───────────────────

def run_fmo_secular(
    init_site: int = 1,
    temperature: float = TEMPERATURE_K,
    t_end_fs: float = 15_000.0,
    n_steps: int = 600,
    gamma_tables: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fast ETE via the secular (Pauli) master equation — populations only.

    Propagates in the EXCITON basis (where the secular master equation is
    diagonal in populations) then converts back to site populations via
    T[m,α] = |U[m,α]|².  Runs in milliseconds.

    Returns
    -------
    times_fs : (n_steps,) array [fs]
    P_site   : (n_steps, N_SITES) site-population matrix
    ete      : float — Energy Transfer Efficiency
    """
    from scipy.integrate import solve_ivp

    if gamma_tables is None:
        gamma_tables = build_gamma_tables(temperature=temperature)

    H_np = build_H_shifted()
    eigvals, U = np.linalg.eigh(H_np)    # U[:,α]: site→exciton transform
    n = N_SITES

    # T[m,α] = |U[m,α]|² : population of site m in exciton state α
    T = U**2                               # (n_sites, n_excitons)

    g_fns = [make_gamma_interp(gamma_tables, site=m) for m in range(n)]

    # Secular Redfield rate matrix in exciton basis
    # Γ[α→β] = Σ_m T[m,α] T[m,β] γ_m(E_α − E_β)
    Gamma = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            omega_ab = eigvals[a] - eigvals[b]
            Gamma[a, b] = sum(
                T[m, a] * T[m, b] * float(g_fns[m](omega_ab))
                for m in range(n)
            )

    # Pauli rate matrix K_ex: dP_ex/dt = K_ex @ P_ex
    K_ex = Gamma.T.copy()
    np.fill_diagonal(K_ex, -Gamma.sum(axis=1))

    # Trapping from BChl TRAP_SITE (0-indexed: trap_idx = TRAP_SITE-1)
    # Loss from all sites (uniform in exciton basis since Σ_m T[m,α] = 1)
    trap_idx = TRAP_SITE - 1
    w_trap   = T[trap_idx, :]          # |⟨α|trap⟩|²
    K_full   = K_ex - np.diag(w_trap * K_TRAP_CM) - np.diag(np.ones(n) * K_LOSS_CM)

    # Initial exciton populations for localised excitation on init_site
    P0_ex = T[init_site - 1, :]        # |⟨α|init_site⟩|²

    # Propagate via eigendecomposition of K_full (ms-scale)
    klam, kV = np.linalg.eig(K_full)
    c = np.linalg.solve(kV, P0_ex)    # coefficients

    t_max_int = t_end_fs * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)
    times_fs  = times_int / (2.0 * np.pi * C_FS)

    # P_ex(t): (n_steps, n_excitons)
    P_ex = np.real(
        (kV @ (c[:, None] * np.exp(klam[:, None] * times_int[None, :]))).T
    ).clip(0.0)

    # Convert to site populations: P_site[t, m] = Σ_α T[m,α] P_ex[t,α]
    P_site = (P_ex @ T.T).clip(0.0)   # (n_steps, n_sites)

    # ETE = k_trap_fs × ∫ P_trap_site(t) dt_fs
    k_trap_fs = K_TRAP_CM * 2.0 * np.pi * C_FS
    ete = k_trap_fs * np.trapz(P_site[:, trap_idx], times_fs)

    return times_fs, P_site, ete


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)

    print("Building spectral density tables …")
    tables = build_gamma_tables(temperature=300.0)

    print("Running dynamics: BChl 1 init, T=300 K, 10 ps …")
    t, rhos = run_fmo(init_site=1, temperature=300.0,
                      t_end_fs=10_000.0, n_steps=400,
                      gamma_tables=tables)

    ete = compute_ete(t, rhos)
    print(f"ETE = {ete*100:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for m in range(1, 9):
        ax.plot(t, site_population(rhos, m), color=colors[m-1],
                lw=1.8, label=f"BChl {m}")
    ax.plot(t, ground_population(rhos), "k--", lw=1.4, label="Ground")
    ax.set_xlabel("Time (fs)"); ax.set_ylabel("Population")
    ax.set_title(f"FMO 8-site realistic — T=300 K, BChl 1 init  (ETE={ete*100:.1f}%)")
    ax.legend(ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/fmo_dynamics_test.png", dpi=150)
    print("Saved results/fmo_dynamics_test.png")
