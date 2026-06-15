"""
Open-quantum-system dynamics for the 4-site energy-transfer model.

Uses QuTiP's Bloch-Redfield solver (brmesolve) with the Ohmic/Drude
spectral density as a proxy for the CMRT used in Ai et al.  Agreement
is expected to be qualitative rather than exact.

Each site couples *diagonally* (dephasing-type) to its own independent
Ohmic bath:   H_SB = Σ_i  |i⟩⟨i| ⊗ B_i

Initial state: full excitation on site 1  (ρ₀ = |1⟩⟨1|).
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from hamiltonian import build_electronic_H, SITE_ENERGIES_CM
from spectral_density import gamma_Ohmic, LAMBDA_CM, GAMMA_CM, TEMPERATURE_K


N_SITES = 4


def _basis_projectors() -> list[qt.Qobj]:
    """Return |i⟩⟨i| projectors for i = 0..3 as QuTiP Qobjs."""
    return [qt.ket2dm(qt.basis(N_SITES, i)) for i in range(N_SITES)]


def run_ohmic(
    r: float,
    theta: float,
    t_end: float = 5000.0,
    n_steps: int = 500,
    lambda_: float = LAMBDA_CM,
    gamma_bath: float = GAMMA_CM,
    temperature: float = TEMPERATURE_K,
) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Propagate the 4-site Frenkel Hamiltonian under an Ohmic bath.

    Parameters
    ----------
    r          : intra-dimer distance (Å)
    theta      : dipole angle (radians)
    t_end      : propagation end time (fs, converted to cm internally)
    n_steps    : number of output time points
    lambda_    : bath reorganisation energy (cm⁻¹)
    gamma_bath : Drude cut-off (cm⁻¹)
    temperature: bath temperature (K)

    Returns
    -------
    times : (n_steps,) array  [fs]
    rhos  : list of n_steps density matrices (QuTiP Qobj)
    """
    # Convert time from fs to cm (using c = 3e-5 cm/fs so that ωt is dimensionless
    # when ω is in cm⁻¹ and t is in 1/(2πc)):
    #   t [cm⁻¹⁻¹] = t [fs] × 2π × 3e-5 cm/fs  → t [cm] = t[fs] / (2π × 3e-5) × 2π × c
    # QuTiP brmesolve uses ħ=1, so time is in units of 1/cm⁻¹.
    # 1 fs = 1e-15 s;  1 cm⁻¹ = 2π × 3e10 rad/s  →  1/(cm⁻¹) = 5.309 fs
    FS_PER_INV_CM = 1.0 / (2.0 * np.pi * 3e-5)   # ≈ 5309 fs per (cm⁻¹)⁻¹  [wrong sign]
    # Correct: 1 cm⁻¹ × 2πc = 2π × 3×10¹⁰ Hz → period = 1/(3×10¹⁰ cm/s × 1 cm⁻¹) s
    #        = 1/(3×10¹⁰) s = 3.33×10⁻¹¹ s ≈ 33333 fs  — no, that's 1 period.
    # Properly:  ω [rad/fs] = 2π × c [cm/fs] × ω [cm⁻¹]
    #            c = 3×10-5 cm/fs
    # So ωt [dimensionless] = 2π × 3e-5 cm/fs × ω [cm⁻¹] × t [fs]
    # For brmesolve we need t in units where H*t is dimensionless with H in cm⁻¹.
    # → t_internal [cm] = t [fs] × 2π × 3e-5

    C_LIGHT_CM_FS = 3e-5   # speed of light in cm/fs

    t_max_internal = t_end * 2.0 * np.pi * C_LIGHT_CM_FS
    times_internal = np.linspace(0.0, t_max_internal, n_steps)

    # Hamiltonian (cm⁻¹) as QuTiP Qobj.
    # Subtract the mean site energy so oscillations are ~400 cm⁻¹ not ~13000 cm⁻¹;
    # this dramatically reduces ODE stiffness without affecting the physics.
    H_np = build_electronic_H(r, theta)
    H_np = H_np - np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H    = qt.Qobj(H_np)

    # Bath coupling: each site couples to its own bath via the projector |i⟩⟨i|
    projectors = _basis_projectors()

    def _make_gamma(lambda_: float, gamma_b: float, temp: float):
        """Return a closure for the bath spectral function accepted by brmesolve."""
        def _gamma(omega):
            return gamma_Ohmic(omega, lambda_=lambda_, gamma=gamma_b, temperature=temp)
        return _gamma

    spectral_fn = _make_gamma(lambda_, gamma_bath, temperature)
    a_ops = [(proj, spectral_fn) for proj in projectors]

    # Initial state: full excitation on site 0 (= site 1 in 1-indexed notation)
    rho0 = qt.basis(N_SITES, 0) * qt.basis(N_SITES, 0).dag()

    result = qt.brmesolve(
        H,
        rho0,
        times_internal,
        a_ops=a_ops,
        sec_cutoff=0.1,
        options={"nsteps": 100000, "rtol": 1e-8, "atol": 1e-10,
                 "method": "adams"},
    )

    times_fs = times_internal / (2.0 * np.pi * C_LIGHT_CM_FS)
    return times_fs, result.states


def population(rhos: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from a list of density matrices (0-indexed)."""
    return np.array([float(rho[site, site].real) for rho in rhos])


# ── Secular Redfield rate matrix (Fix 2) ─────────────────────────────────────

def secular_reff(
    r: float,
    theta: float,
    t_end: float = 200_000.0,
    n_steps: int = 400,
    lambda_: float = LAMBDA_CM,
    gamma_bath: float = GAMMA_CM,
    temperature: float = TEMPERATURE_K,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Reff via the secular (Pauli) master equation — no ODE integration.

    In the secular approximation populations decouple from coherences and
    evolve under a classical rate matrix K built directly from the Redfield
    decay rates Γ_{α→β}:

        Γ_{α→β} = [Σᵢ (U[i,α] U[i,β])²] × γ(Eα − Eβ)

    K is then 4×4, diagonalised analytically, and P₄(t) is computed as a
    sum of four real exponentials — microseconds per geometry point.

    Parameters
    ----------
    r, theta  : geometry
    t_end     : propagation window (fs); long enough to see full transfer
    n_steps   : time-grid points for P₄(t)
    lambda_, gamma_bath, temperature : bath parameters

    Returns
    -------
    times [fs], P4 [dimensionless], Reff [fs⁻¹]
    """
    from efficiency import compute_Reff
    from scipy.linalg import expm as _expm

    C_FS = 3e-5  # cm/fs

    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)

    eigvals, U = np.linalg.eigh(H_np)   # U[:,α] = eigenstate α in site basis

    n = N_SITES
    # Förster overlap factors: F[α,β] = Σᵢ U[i,α]² U[i,β]²
    F = (U ** 2).T @ (U ** 2)            # (n,n), F[α,β] = F[β,α]

    # Redfield rates Γ[α,β] = rate FROM α TO β
    Gamma = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            omega_ab = eigvals[a] - eigvals[b]   # E_a - E_b
            Gamma[a, b] = F[a, b] * gamma_Ohmic(
                omega_ab, lambda_=lambda_, gamma=gamma_bath, temperature=temperature
            )

    # Pauli rate matrix  dP/dt = K P
    # K[β,α] = Γ[α→β]  (gain into β from α)
    # K[α,α] = −Σ_{β≠α} Γ[α→β]  (total loss from α)
    K = Gamma.T.copy()
    np.fill_diagonal(K, -Gamma.sum(axis=1))

    # Initial exciton populations: excitation on site 0 (site 1)
    P0 = U[0, :] ** 2                    # Pα(0) = |⟨α|site 1⟩|²

    # Weights of site 4 (index 3) in each exciton
    w4 = U[3, :] ** 2                    # |⟨site4|α⟩|²

    # Analytical propagation via eigendecomposition of K
    klam, kV = np.linalg.eig(K)
    c = (w4 @ kV) * np.linalg.solve(kV, P0)   # spectral coefficients

    # Convert t_end from fs to internal (cm⁻¹)⁻¹ units
    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int  = np.linspace(0.0, t_max_int, n_steps)
    times_fs   = times_int / (2.0 * np.pi * C_FS)

    P4 = np.real(
        np.sum(c[None, :] * np.exp(klam[None, :] * times_int[:, None]), axis=1)
    ).clip(0.0, 1.0)

    Reff = compute_Reff(times_fs, P4)
    return times_fs, P4, Reff


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, rhos = run_ohmic(r=11.3, theta=0.0, t_end=3000.0, n_steps=300)
    for i, label in enumerate(["P1", "P2", "P3", "P4"]):
        plt.plot(t, population(rhos, i), label=label)
    plt.xlabel("Time (fs)")
    plt.ylabel("Population")
    plt.legend()
    plt.title("r=11.3 Å, θ=0")
    plt.tight_layout()
    plt.savefig("results/dynamics_single.png", dpi=150)
    print("Saved results/dynamics_single.png")
