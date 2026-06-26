"""
Open-quantum-system dynamics and trapping efficiency for the 8-site FMO model.

Two layers:

* compute_ete()  — FAST secular-Redfield excitation-transfer-efficiency (ETE)
  and mean trapping time, via a single linear solve on the 8x8 exciton-
  population rate matrix plus reaction-centre trapping and exciton loss.
  This is the workhorse for the position scans (thousands of geometries).

* run_with_trap() / run_ohmic() — full Lindblad/Bloch-Redfield trajectories on
  the 8 sites (+ trap + ground states) for the showcase figures.

Physics / parameters (all literature-grounded; see fmo_data.py):
    * Ohmic/Drude bath, lambda = 35, gamma = 53 cm^-1, T = 300 K
      (Ishizaki-Fleming; Shabani et al. 2014).
    * Reaction-centre trapping at BChl 3, k_trap = (0.5 ps)^-1 = 2 ps^-1.
    * Excitation loss everywhere, k_loss = (1 ns)^-1.
    * Initial excitation at the antenna-facing pigments BChl 1 and BChl 6.
      (all Shabani, Mohseni, Rabitz & Lloyd, Phys. Rev. E 89, 042706 (2014))
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from hamiltonian import build_electronic_H
from fmo_data import SITE_ENERGIES_CM, N_SITES, TRAP_SITE, ENTRY_SITES
from spectral_density import gamma_Ohmic, LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

# t_internal = t_fs * 2*pi*C_FS makes H * t_internal dimensionless (H in cm^-1)
C_FS = 3e-5  # speed of light in cm/fs

# Trapping and loss rates (converted fs^-1 -> cm^-1 via /(2*pi*C_FS))
K_TRAP_FS = 0.002      # (0.5 ps)^-1 = 2 ps^-1   reaction-centre trapping
K_LOSS_FS = 1.0e-6     # (1 ns)^-1               excitation loss

def _fs_to_cm(rate_fs: float) -> float:
    return rate_fs / (2.0 * np.pi * C_FS)


# ── Fast ETE via the secular (Pauli) exciton rate matrix ──────────────────────

def _redfield_rate_matrix(H: np.ndarray,
                          lambda_: float, gamma_bath: float, temperature: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Secular-Redfield inter-exciton population rate matrix K (cm^-1) and the
    eigenvector matrix U (columns = excitons in the site basis).

        Gamma_{a->b} = [sum_i (U[i,a] U[i,b])^2] * gamma(E_a - E_b)
        K[b,a] = Gamma_{a->b}   (gain),   K[a,a] = -sum_b Gamma_{a->b}

    Works for any number of sites (N = H.shape[0]); the uniform energy reference
    only conditions the eigensolve and does not affect the dynamics.
    """
    H = H - np.mean(np.diag(H)) * np.eye(H.shape[0])      # reduce stiffness
    eigvals, U = np.linalg.eigh(H)

    F = (U ** 2).T @ (U ** 2)                              # overlap F[a,b]
    omega = eigvals[:, None] - eigvals[None, :]           # E_a - E_b
    g = gamma_Ohmic(omega, lambda_=lambda_, gamma=gamma_bath, temperature=temperature)
    np.fill_diagonal(g, 0.0)
    Gamma = F * g                                          # rate a->b
    K = Gamma.T.copy()
    np.fill_diagonal(K, -Gamma.sum(axis=1))
    return K, U


def compute_ete(H: np.ndarray,
                initial_sites: tuple[int, ...] = ENTRY_SITES,
                trap_site: int = TRAP_SITE,
                k_trap_fs: float = K_TRAP_FS,
                k_loss_fs: float = K_LOSS_FS,
                lambda_: float = LAMBDA_CM,
                gamma_bath: float = GAMMA_CM,
                temperature: float = TEMPERATURE_K) -> tuple[float, float]:
    """
    Excitation-transfer efficiency (trapping yield) and mean trapping time.

    Populations evolve as dP/dt = K_tot P in the exciton basis, where
        K_tot = K_redfield - k_loss I - diag(k_trap |U[trap, a]|^2).
    Trapping yield and mean time follow from moments of P(t):
        integral_0^inf P dt   = -K_tot^{-1} P0
        integral_0^inf t P dt =  K_tot^{-2} P0
        ETE = k_trap * sum_a |U[trap,a]|^2 (integral P dt)_a
        tau = (sum_a w_a integral t P dt) / (sum_a w_a integral P dt)

    Averaged over the requested initial entry pigments.

    Returns
    -------
    ete : trapping yield in [0, 1]
    tau_fs : mean trapping time (fs)
    """
    K, U = _redfield_rate_matrix(H, lambda_, gamma_bath, temperature)

    k_trap = _fs_to_cm(k_trap_fs)
    k_loss = _fs_to_cm(k_loss_fs)

    w_trap = k_trap * U[trap_site, :] ** 2                 # trapping weight per exciton
    K_tot = K - k_loss * np.eye(H.shape[0]) - np.diag(w_trap)

    etes, taus = [], []
    for s in initial_sites:
        P0 = U[s, :] ** 2                                  # localized site -> exciton pops
        m1 = np.linalg.solve(K_tot, P0)                    # K^{-1} P0
        int_P = -m1                                        # integral P dt   (>= 0)
        m2 = np.linalg.solve(K_tot, m1)                    # K^{-2} P0
        int_tP = m2                                        # integral t P dt
        yield_ = float(w_trap @ int_P)                     # trapping yield (ETE)
        etes.append(yield_)
        tau_int = (w_trap @ int_tP) / yield_ if yield_ > 0 else np.inf
        taus.append(tau_int / (2.0 * np.pi * C_FS))        # internal -> fs
    return float(np.mean(etes)), float(np.mean(taus))


# ── Full trajectories (showcase figures) ──────────────────────────────────────

def _basis_projectors(dim: int, n: int) -> list[qt.Qobj]:
    return [qt.ket2dm(qt.basis(dim, i)) for i in range(n)]


def _spectral_fn(lambda_, gamma_bath, temperature):
    def _fn(omega):
        return gamma_Ohmic(omega, lambda_=lambda_, gamma=gamma_bath, temperature=temperature)
    return _fn


def run_ohmic(displacements: np.ndarray | None = None,
              initial_site: int = ENTRY_SITES[0],
              t_end: float = 5000.0, n_steps: int = 500,
              lambda_: float = LAMBDA_CM, gamma_bath: float = GAMMA_CM,
              temperature: float = TEMPERATURE_K) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Bloch-Redfield dynamics on the 8 sites (no trap) — used for coherence
    analysis.  Each site couples diagonally to its own Ohmic bath.
    """
    H_np = build_electronic_H(displacements)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H = qt.Qobj(H_np)

    a_ops = [(proj, _spectral_fn(lambda_, gamma_bath, temperature))
             for proj in _basis_projectors(N_SITES, N_SITES)]
    rho0 = qt.ket2dm(qt.basis(N_SITES, initial_site))

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)
    result = qt.brmesolve(H, rho0, times_int, a_ops=a_ops, sec_cutoff=0.1,
                          options={"nsteps": 100000, "rtol": 1e-8, "atol": 1e-10,
                                   "method": "adams"})
    return times_int / (2.0 * np.pi * C_FS), result.states


def run_with_trap(displacements: np.ndarray | None = None,
                  initial_site: int = ENTRY_SITES[0],
                  trap_site: int = TRAP_SITE,
                  k_trap_fs: float = K_TRAP_FS, k_loss_fs: float = K_LOSS_FS,
                  t_end: float = 15000.0, n_steps: int = 500,
                  lambda_: float = LAMBDA_CM, gamma_bath: float = GAMMA_CM,
                  temperature: float = TEMPERATURE_K
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full trajectory on 8 sites + reaction-centre trap + ground (loss) state.

    Returns
    -------
    times_fs, P_sites (n_steps, 8), Q (trap population), L (lost population)
    """
    dim = N_SITES + 2
    i_trap, i_ground = N_SITES, N_SITES + 1

    H_np = build_electronic_H(displacements)
    H_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H_big = np.zeros((dim, dim))
    H_big[:N_SITES, :N_SITES] = H_np
    H = qt.Qobj(H_big)

    a_ops = [(qt.ket2dm(qt.basis(dim, i)), _spectral_fn(lambda_, gamma_bath, temperature))
             for i in range(N_SITES)]

    c_ops = []
    k_trap = _fs_to_cm(k_trap_fs)
    c_tr = np.zeros((dim, dim)); c_tr[i_trap, trap_site] = np.sqrt(k_trap)
    c_ops.append(qt.Qobj(c_tr))
    k_loss = _fs_to_cm(k_loss_fs)
    for i in range(N_SITES):
        c_l = np.zeros((dim, dim)); c_l[i_ground, i] = np.sqrt(k_loss)
        c_ops.append(qt.Qobj(c_l))

    rho0 = qt.ket2dm(qt.basis(dim, initial_site))
    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)
    result = qt.brmesolve(H, rho0, times_int, a_ops=a_ops, c_ops=c_ops, sec_cutoff=0.1,
                          options={"nsteps": 100000, "rtol": 1e-8, "atol": 1e-10,
                                   "method": "adams"})

    states = result.states
    P_sites = np.array([[float(rho[i, i].real) for i in range(N_SITES)] for rho in states])
    Q = np.array([float(rho[i_trap, i_trap].real) for rho in states])
    L = np.array([float(rho[i_ground, i_ground].real) for rho in states])
    return times_int / (2.0 * np.pi * C_FS), P_sites, Q, L


def population(rhos: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from a list of density matrices (0-indexed)."""
    return np.array([float(rho[site, site].real) for rho in rhos])


if __name__ == "__main__":
    H = build_electronic_H()
    for s in range(N_SITES):
        ete, tau = compute_ete(H, initial_sites=(s,))
        print(f"start BChl {s+1}:  ETE = {ete:.4f}   tau = {tau/1000:.2f} ps")
    ete, tau = compute_ete(H)
    print(f"\nentry-averaged (BChl 1 & 6):  ETE = {ete:.4f}   tau = {tau/1000:.2f} ps")
