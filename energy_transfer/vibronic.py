"""
Phase 2: Structured vibronic spectral density model.

Two intramolecular vibronic modes are explicitly quantised and coupled
to specific sites of the 4-site Frenkel Hamiltonian.  The residual
Ohmic bath drives electronic dephasing via Lindblad collapse operators.

Mode parameters (Ai et al. / FMO literature):
    Mode 1 : ω₁ = 726 cm⁻¹ coupled to site 1  (Huang-Rhys S₁ = 0.025)
    Mode 2 : ω₂ = 243 cm⁻¹ coupled to site 3  (Huang-Rhys S₂ = 0.013)
    g_k = ω_k × √S_k  →  g₁ ≈ 115 cm⁻¹,  g₂ ≈ 28 cm⁻¹

Hilbert-space ordering: electronic ⊗ mode-1 ⊗ mode-2
    dims = [4, N_MAX+1, N_MAX+1] = [4, 4, 4]  →  64-dimensional
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from hamiltonian import build_electronic_H, SITE_ENERGIES_CM
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

# ── Mode constants ─────────────────────────────────────────────────────────────

OMEGA1_CM   = 726.0   # vibronic mode 1 frequency (cm⁻¹), site 1
OMEGA2_CM   = 243.0   # vibronic mode 2 frequency (cm⁻¹), site 3
S1          = 0.025   # Huang-Rhys factor, mode 1
S2          = 0.013   # Huang-Rhys factor, mode 2
N_MAX       = 3       # Fock truncation (keeps 0..N_MAX; each mode dim = N_MAX+1)
N_SITES     = 4
N_MODES     = 2
DIM_EL      = N_SITES
DIM_M1      = N_MAX + 1          # 4
DIM_M2      = N_MAX + 1          # 4
DIM_FULL    = DIM_EL * DIM_M1 * DIM_M2   # 64

GAMMA_VIB_CM = 20.0  # vibrational damping rate (cm⁻¹), ~265 fs

_HBAR_OVER_KB = 1.4388   # K·cm

C_FS = 3e-5  # speed of light in cm/fs


# ── Hilbert-space building blocks ─────────────────────────────────────────────

def _eye(d: int) -> qt.Qobj:
    return qt.qeye(d)


def _mode_ops(d: int) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Return (a, a†, N) for a d-dimensional Fock space."""
    a   = qt.destroy(d)
    adag = a.dag()
    N   = qt.num(d)
    return a, adag, N


def build_vibronic_H(r: float, theta: float) -> qt.Qobj:
    """
    Construct the full vibronic Hamiltonian in the tensor-product space.

    H = H_el ⊗ I₁ ⊗ I₂
      + I_el ⊗ ω₁ N₁ ⊗ I₂
      + I_el ⊗ I₁ ⊗ ω₂ N₂
      + g₁ |0⟩⟨0| ⊗ (a₁+a₁†) ⊗ I₂
      + g₂ |2⟩⟨2| ⊗ I₁ ⊗ (a₂+a₂†)

    Mean site energy is subtracted from H_el to reduce stiffness.

    Parameters
    ----------
    r, theta : geometry (Å, radians)

    Returns
    -------
    H : 64×64 QuTiP Qobj
    """
    H_el_np = build_electronic_H(r, theta)
    H_el_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H_el = qt.Qobj(H_el_np)

    I_el = _eye(DIM_EL)
    I_m1 = _eye(DIM_M1)
    I_m2 = _eye(DIM_M2)

    a1, a1dag, N1 = _mode_ops(DIM_M1)
    a2, a2dag, N2 = _mode_ops(DIM_M2)

    g1 = OMEGA1_CM * np.sqrt(S1)
    g2 = OMEGA2_CM * np.sqrt(S2)

    # Site projectors in electronic subspace
    proj0 = qt.basis(DIM_EL, 0) * qt.basis(DIM_EL, 0).dag()  # site 1
    proj2 = qt.basis(DIM_EL, 2) * qt.basis(DIM_EL, 2).dag()  # site 3

    H = (
        qt.tensor(H_el,         I_m1,        I_m2)
      + qt.tensor(I_el, OMEGA1_CM * N1,      I_m2)
      + qt.tensor(I_el,         I_m1, OMEGA2_CM * N2)
      + qt.tensor(g1 * proj0, a1 + a1dag,    I_m2)
      + qt.tensor(g2 * proj2,   I_m1,  a2 + a2dag)
    )
    return H


def _collapse_operators(temperature: float = TEMPERATURE_K,
                        lambda_: float = LAMBDA_CM,
                        gamma_bath: float = GAMMA_CM,
                        gamma_vib: float = GAMMA_VIB_CM) -> list[qt.Qobj]:
    """
    Build Lindblad collapse operators for vibrational damping + electronic dephasing.

    Vibrational damping (thermal):
        c_down_k = √[γ_vib (n_th_k + 1)] × a_k   (emission)
        c_up_k   = √[γ_vib  n_th_k]       × a_k†  (absorption)

    Electronic dephasing from residual Ohmic bath:
        c_deph_i = √γ_φ × |i⟩⟨i|   for each site i
        γ_φ = 2 λ_res T / (hbar_kB γ_bath)
        λ_res = λ - S₁ω₁ - S₂ω₂   (reorganisation energy of residual bath)
    """
    I_el = _eye(DIM_EL)
    I_m1 = _eye(DIM_M1)
    I_m2 = _eye(DIM_M2)
    a1, a1dag, _ = _mode_ops(DIM_M1)
    a2, a2dag, _ = _mode_ops(DIM_M2)

    def n_th(omega: float) -> float:
        beta_omega = _HBAR_OVER_KB * omega / temperature
        if beta_omega > 500:
            return 0.0
        return 1.0 / (np.expm1(beta_omega))

    # Vibrational collapse operators (act only on mode subspaces)
    n1 = n_th(OMEGA1_CM)
    n2 = n_th(OMEGA2_CM)

    c_ops: list[qt.Qobj] = [
        np.sqrt(gamma_vib * (n1 + 1.0)) * qt.tensor(I_el, a1,     I_m2),
        np.sqrt(gamma_vib *  n1)         * qt.tensor(I_el, a1dag,  I_m2),
        np.sqrt(gamma_vib * (n2 + 1.0)) * qt.tensor(I_el, I_m1,   a2),
        np.sqrt(gamma_vib *  n2)         * qt.tensor(I_el, I_m1,   a2dag),
    ]

    # Residual Ohmic reorganisation energy after removing vibronic modes
    lambda_res = max(0.0, lambda_ - S1 * OMEGA1_CM - S2 * OMEGA2_CM)
    gamma_phi  = 2.0 * lambda_res * temperature / (_HBAR_OVER_KB * gamma_bath)

    if gamma_phi > 0.0:
        sqrt_phi = np.sqrt(gamma_phi)
        for i in range(N_SITES):
            proj_i = qt.basis(DIM_EL, i) * qt.basis(DIM_EL, i).dag()
            c_ops.append(sqrt_phi * qt.tensor(proj_i, I_m1, I_m2))

    return c_ops


def run_structured(
    r: float,
    theta: float,
    t_end: float = 5000.0,
    n_steps: int = 500,
    temperature: float = TEMPERATURE_K,
    lambda_: float = LAMBDA_CM,
    gamma_bath: float = GAMMA_CM,
    gamma_vib: float = GAMMA_VIB_CM,
) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Propagate the vibronic Hamiltonian via Lindblad mesolve.

    Initial state: site 1 excited, both modes in vacuum.

    Parameters
    ----------
    r, theta  : geometry (Å, radians)
    t_end     : propagation end time (fs)
    n_steps   : output time points

    Returns
    -------
    times [fs], rhos_el : list of reduced 4×4 electronic density matrices
    """
    H   = build_vibronic_H(r, theta)
    c_ops = _collapse_operators(temperature, lambda_, gamma_bath, gamma_vib)

    # Initial state: |site1, vac1, vac2⟩
    psi0 = qt.tensor(
        qt.basis(DIM_EL, 0),
        qt.basis(DIM_M1, 0),
        qt.basis(DIM_M2, 0),
    )
    rho0 = psi0 * psi0.dag()

    t_max_int  = t_end * 2.0 * np.pi * C_FS
    times_int  = np.linspace(0.0, t_max_int, n_steps)

    result = qt.mesolve(H, rho0, times_int, c_ops=c_ops,
                        options={"nsteps": 50000, "rtol": 1e-8, "atol": 1e-10})

    # Partial trace over mode subspaces to get electronic reduced density matrix
    # Subsystem indices: 0=electronic, 1=mode1, 2=mode2
    rhos_el = [rho.ptrace([0]) for rho in result.states]
    times_fs = times_int / (2.0 * np.pi * C_FS)
    return times_fs, rhos_el


def population_vibronic(rhos_el: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from the reduced electronic density matrices."""
    return np.array([float(rho[site, site].real) for rho in rhos_el])


def vibronic_reff(
    r: float,
    theta: float,
    t_end: float = 5000.0,
    n_steps: int = 300,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run structured dynamics and return (times, P4, Reff) — mirrors secular_reff API.

    Used by Phase 4 heatmap scan for the vibronic variant.
    """
    from efficiency import compute_Reff

    times, rhos_el = run_structured(r, theta, t_end=t_end, n_steps=n_steps, **kwargs)
    P4 = population_vibronic(rhos_el, site=3)
    Reff = compute_Reff(times, P4)
    return times, P4, Reff


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("Building vibronic H at r=11.3 Å, θ=0 …")
    H = build_vibronic_H(11.3, 0.0)
    print(f"  H dims: {H.shape}, norm: {H.norm():.1f}")

    print("Running structured dynamics (5 ps, 300 points) …")
    t, rhos_el = run_structured(11.3, 0.0, t_end=5000.0, n_steps=300)
    P4 = population_vibronic(rhos_el, site=3)
    print(f"  P4(∞) = {P4[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(4):
        ax.plot(t, population_vibronic(rhos_el, i), label=f"Site {i+1}")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Population")
    ax.set_title("Vibronic model: r=11.3 Å, θ=0")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(__file__).parent / "results" / "vibronic_test.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
