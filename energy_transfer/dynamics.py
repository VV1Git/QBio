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
# t_internal = t_fs × 2π × C_FS  makes H × t_internal dimensionless (H in cm⁻¹)
C_FS = 3e-5  # speed of light in cm/fs


def _basis_projectors() -> list[qt.Qobj]:
    """Return |i⟩⟨i| projectors for i = 0..3 as QuTiP Qobjs."""
    return [qt.ket2dm(qt.basis(N_SITES, i)) for i in range(N_SITES)]


def _make_spectral_fn(lambda_: float, gamma_bath: float, temperature: float):
    """Return a spectral function closure for brmesolve a_ops."""
    def _fn(omega):
        return gamma_Ohmic(omega, lambda_=lambda_, gamma=gamma_bath, temperature=temperature)
    return _fn


def _secular_setup(
    r: float,
    theta: float,
    lambda_: float,
    gamma_bath: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Pauli rate matrix K for secular Redfield propagation.

    Vectorised: the full N×N off-diagonal rate matrix is computed in one
    numpy call rather than a Python double loop.

    Returns
    -------
    K  : (N_SITES, N_SITES) Pauli rate matrix  dP/dt = K P
    w4 : (N_SITES,) weight of site 4 in each exciton  |⟨site4|α⟩|²
    P0 : (N_SITES,) initial exciton populations  |⟨α|site1⟩|²
    """
    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    eigvals, U = np.linalg.eigh(H_np)

    F = (U ** 2).T @ (U ** 2)          # Förster overlap: F[α,β] = Σᵢ U[i,α]² U[i,β]²

    omega_matrix = eigvals[:, None] - eigvals[None, :]   # (N, N), ω[α,β] = Eα − Eβ
    gamma_matrix = gamma_Ohmic(omega_matrix, lambda_=lambda_, gamma=gamma_bath,
                               temperature=temperature)
    np.fill_diagonal(gamma_matrix, 0.0)                  # no self-transition
    Gamma = F * gamma_matrix                             # Gamma[α,β] = rate FROM α TO β

    K = Gamma.T.copy()                                   # K[β,α] = gain into β from α
    np.fill_diagonal(K, -Gamma.sum(axis=1))              # total loss from α

    w4 = U[3, :] ** 2   # |⟨site4|α⟩|²
    P0 = U[0, :] ** 2   # initial exciton pops (excitation on site 1)
    return K, w4, P0


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
    t_end      : propagation end time (fs)
    n_steps    : number of output time points
    lambda_    : bath reorganisation energy (cm⁻¹)
    gamma_bath : Drude cut-off (cm⁻¹)
    temperature: bath temperature (K)

    Returns
    -------
    times : (n_steps,) array  [fs]
    rhos  : list of n_steps density matrices (QuTiP Qobj)
    """
    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)

    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H = qt.Qobj(H_np)

    spectral_fn = _make_spectral_fn(lambda_, gamma_bath, temperature)
    a_ops = [(proj, spectral_fn) for proj in _basis_projectors()]

    rho0 = qt.basis(N_SITES, 0) * qt.basis(N_SITES, 0).dag()

    result = qt.brmesolve(
        H, rho0, times_int,
        a_ops=a_ops,
        sec_cutoff=0.1,
        options={"nsteps": 100000, "rtol": 1e-8, "atol": 1e-10, "method": "adams"},
    )

    times_fs = times_int / (2.0 * np.pi * C_FS)
    return times_fs, result.states


def population(rhos: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from a list of density matrices (0-indexed)."""
    return np.array([float(rho[site, site].real) for rho in rhos])


# ── Secular Redfield rate matrix ──────────────────────────────────────────────

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

    K is 4×4, diagonalised analytically, and P₄(t) is computed as a sum of
    four real exponentials — microseconds per geometry point.

    Parameters
    ----------
    r, theta  : geometry
    t_end     : propagation window (fs)
    n_steps   : time-grid points for P₄(t)

    Returns
    -------
    times [fs], P4 [dimensionless], Reff [fs⁻¹]
    """
    from efficiency import compute_Reff

    K, w4, P0 = _secular_setup(r, theta, lambda_, gamma_bath, temperature)

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)
    times_fs  = times_int / (2.0 * np.pi * C_FS)

    klam, kV = np.linalg.eig(K)
    c = (w4 @ kV) * np.linalg.solve(kV, P0)   # spectral coefficients

    P4 = np.real(
        np.sum(c[None, :] * np.exp(klam[None, :] * times_int[:, None]), axis=1)
    ).clip(0.0, 1.0)

    Reff = compute_Reff(times_fs, P4)
    return times_fs, P4, Reff


def run_ohmic_with_trap(
    r: float,
    theta: float,
    kappa_trap_fs: float = 0.001,
    t_end: float = 15_000.0,
    n_steps: int = 500,
    lambda_: float = LAMBDA_CM,
    gamma_bath: float = GAMMA_CM,
    temperature: float = TEMPERATURE_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bloch-Redfield dynamics on a 5-site system: sites 1–4 (Frenkel) + trap (RC).

    A Lindblad collapse operator c_trap = √κ_trap × |trap⟩⟨site4⟩ drains
    excitation from site 4 irreversibly into the reaction-centre trap state.
    Q(t) = P_trap(t) rises from 0 → 1, showing complete energy harvesting.

    Parameters
    ----------
    kappa_trap_fs : trapping rate at site 4 [fs⁻¹] (default 0.001 = 1 ps⁻¹)
    t_end         : propagation window [fs]
    n_steps       : output time points

    Returns
    -------
    times_fs [fs], P4 [site-4 population], Q [trap population 0→1]
    """
    N5 = N_SITES + 1

    H_np = build_electronic_H(r, theta)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)

    H5 = np.zeros((N5, N5))
    H5[:N_SITES, :N_SITES] = H_np
    H = qt.Qobj(H5)

    spectral_fn = _make_spectral_fn(lambda_, gamma_bath, temperature)
    a_ops = [
        (qt.ket2dm(qt.basis(N5, i)), spectral_fn)
        for i in range(N_SITES)
    ]

    kappa_cm = kappa_trap_fs / (2.0 * np.pi * C_FS)
    c_trap_arr = np.zeros((N5, N5))
    c_trap_arr[N_SITES, N_SITES - 1] = np.sqrt(kappa_cm)   # |trap⟩⟨site4|
    c_trap = qt.Qobj(c_trap_arr)

    rho0 = qt.basis(N5, 0) * qt.basis(N5, 0).dag()

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)

    result = qt.brmesolve(
        H, rho0, times_int,
        a_ops=a_ops, c_ops=[c_trap],
        sec_cutoff=0.1,
        options={"nsteps": 100000, "rtol": 1e-8, "atol": 1e-10, "method": "adams"},
    )

    times_fs = times_int / (2.0 * np.pi * C_FS)
    P4 = np.array([float(rho[N_SITES - 1, N_SITES - 1].real) for rho in result.states])
    Q  = np.array([float(rho[N_SITES,     N_SITES].real)     for rho in result.states])

    return times_fs, P4, Q


def secular_with_trap(
    r: float,
    theta: float,
    kappa_trap_fs: float = 0.002,
    t_end: float = 10_000.0,
    n_steps: int = 500,
    lambda_: float = LAMBDA_CM,
    gamma_bath: float = GAMMA_CM,
    temperature: float = TEMPERATURE_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Secular Pauli propagation with irreversible trapping at site 4 (reaction centre).

    Extends the Pauli rate matrix with a sink at site 4: each exciton α loses
    population at rate w₄[α] × κ_trap.  The cumulative yield Q(t) rises 0 → 1,
    reflecting the fraction of excitation irreversibly trapped by time t.

    Parameters
    ----------
    kappa_trap_fs : trapping rate at site 4 in fs⁻¹  (default 0.002 = 2 ps⁻¹)
    t_end         : propagation window [fs]
    n_steps       : time-grid points

    Returns
    -------
    times_fs  : (n_steps,) [fs]
    P4_site   : (n_steps,) transient population at site 4 during dynamics
    Q         : (n_steps,) cumulative trapping yield, 0 → 1
    """
    K, w4, P0 = _secular_setup(r, theta, lambda_, gamma_bath, temperature)

    kappa_trap_cm = kappa_trap_fs / (2.0 * np.pi * C_FS)
    K -= np.diag(w4 * kappa_trap_cm)   # add irreversible sink at site 4

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)
    times_fs  = times_int / (2.0 * np.pi * C_FS)

    klam, kV = np.linalg.eig(K)
    c_all = np.linalg.solve(kV, P0)

    w4_kV = w4 @ kV
    P4_site = np.real(
        np.sum(
            w4_kV[None, :] * c_all[None, :] * np.exp(klam[None, :] * times_int[:, None]),
            axis=1,
        )
    ).clip(0.0, 1.0)

    dt_fs = times_fs[1] - times_fs[0] if n_steps > 1 else 1.0
    Q = np.cumsum(P4_site) * dt_fs * kappa_trap_fs
    Q = np.clip(Q, 0.0, 1.0)

    return times_fs, P4_site, Q


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
