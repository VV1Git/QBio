"""
Phase 2: Structured vibronic spectral density model.

Two intramolecular vibronic modes are explicitly quantised and coupled
to specific sites of the 4-site Frenkel Hamiltonian.  The residual
Ohmic bath drives electronic dephasing via Lindblad collapse operators.

Mode parameters — the two most strongly coupled intramolecular vibrations
of bacteriochlorophyll a in the FMO complex, from the ΔFLN measurements of
Rätsep & Freiberg, J. Lumin. 127, 251 (2007), Table 1:

    Mode 1 : ω₁ = 770 cm⁻¹ coupled to site 1  (Huang-Rhys S₁ = 0.018, the
                                                 strongest intramolecular mode)
    Mode 2 : ω₂ = 243 cm⁻¹ coupled to site 3  (Huang-Rhys S₂ = 0.012)
    g_k = ω_k × √S_k  →  g₁ ≈ 103 cm⁻¹,  g₂ ≈ 27 cm⁻¹

The remaining ~60 weaker modes (ΣS_vib ≈ 0.31) plus the phonon continuum
(S_ph ≈ 0.29, peak ≈ 22 cm⁻¹) are folded into the residual Ohmic bath that
drives pure electronic dephasing.

Hilbert-space ordering: electronic ⊗ mode-1 ⊗ mode-2
    dims = [4, N_MAX+1, N_MAX+1] = [4, 5, 5]  →  100-dimensional

N_MAX = 4 (5 Fock states/mode) converges P₄(∞) to <0.1 % and captures the
thermal occupation of the 243 cm⁻¹ mode (n_th ≈ 0.45 at 300 K).
"""

from __future__ import annotations

import warnings

import numpy as np
import qutip as qt

from hamiltonian import build_electronic_H, SITE_ENERGIES_CM
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

# ── Mode constants ─────────────────────────────────────────────────────────────

OMEGA1_CM   = 770.0   # vibronic mode 1 frequency (cm⁻¹), site 1 — Rätsep 2007
OMEGA2_CM   = 243.0   # vibronic mode 2 frequency (cm⁻¹), site 3 — Rätsep 2007
S1          = 0.018   # Huang-Rhys factor, mode 1 (strongest intramolecular mode)
S2          = 0.012   # Huang-Rhys factor, mode 2
N_MAX       = 4       # Fock truncation (keeps 0..N_MAX; each mode dim = N_MAX+1)
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


def build_vibronic_H(r: float, theta: float, n_max: int = N_MAX) -> qt.Qobj:
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
    n_max    : Fock truncation per mode (each mode dim = n_max+1)

    Returns
    -------
    H : QuTiP Qobj of dimension 4·(n_max+1)²
    """
    dim_m = n_max + 1
    H_el_np = build_electronic_H(r, theta)
    H_el_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H_el = qt.Qobj(H_el_np)

    I_el = _eye(DIM_EL)
    I_m1 = _eye(dim_m)
    I_m2 = _eye(dim_m)

    a1, a1dag, N1 = _mode_ops(dim_m)
    a2, a2dag, N2 = _mode_ops(dim_m)

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
                        gamma_vib: float = GAMMA_VIB_CM,
                        n_max: int = N_MAX) -> list[qt.Qobj]:
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
    dim_m = n_max + 1
    I_el = _eye(DIM_EL)
    I_m1 = _eye(dim_m)
    I_m2 = _eye(dim_m)
    a1, a1dag, _ = _mode_ops(dim_m)
    a2, a2dag, _ = _mode_ops(dim_m)

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
    n_max: int = N_MAX,
) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Propagate the vibronic Hamiltonian via Lindblad mesolve.

    Initial state: site 1 excited, both modes in vacuum.

    Parameters
    ----------
    r, theta  : geometry (Å, radians)
    t_end     : propagation end time (fs)
    n_steps   : output time points
    n_max     : Fock truncation per mode.  Default 4 (dim 100, P₄ converged
                <0.1 %).  The Reff scans use n_max=3 (dim 64, <0.25 %, ~3×
                faster) since the rate is insensitive to the truncation.

    Returns
    -------
    times [fs], rhos_el : list of reduced 4×4 electronic density matrices
    """
    from gpu_utils import gpu_active

    dim_full = DIM_EL * (n_max + 1) ** 2
    n_m_sq   = (n_max + 1) ** 2

    # Build H and c_ops with CSR (sparse ops, avoids any dense matmul on GPU)
    _saved_dtype = qt.settings.core["default_dtype"]
    qt.settings.core["default_dtype"] = "CSR"
    try:
        H     = build_vibronic_H(r, theta, n_max=n_max)
        c_ops = _collapse_operators(temperature, lambda_, gamma_bath, gamma_vib,
                                    n_max=n_max)
        psi0  = qt.tensor(
            qt.basis(DIM_EL, 0),
            qt.basis(n_max + 1, 0),
            qt.basis(n_max + 1, 0),
        )
        rho0 = psi0 * psi0.dag()
    finally:
        qt.settings.core["default_dtype"] = _saved_dtype

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)

    if gpu_active():
        import jax.numpy as jnp
        import diffrax

        # dim×dim JAX arrays (~80 KB each) — negligible GPU memory.
        # Avoids the dim²×dim² dense Liouvillian and cuBLAS-lt autotuning.
        H_jax   = jnp.array(H.full())
        c_jax   = jnp.array([c.full() for c in c_ops])
        cd_jax  = jnp.conj(c_jax).transpose(0, 2, 1)         # c†
        ctc_jax = jnp.einsum("kij,kjl->kil", cd_jax, c_jax)  # c†c per operator

        def _lindblad_rhs(t, y, args):
            H_, c_, cd_, ctc_ = args
            rho = y.reshape(dim_full, dim_full)
            dr = -1j * (H_ @ rho - rho @ H_)
            # sum_k [c_k ρ c_k† − ½ c_k†c_k ρ − ½ ρ c_k†c_k]
            dr = dr + (
                jnp.einsum("kij,jl,klm->im", c_, rho, cd_)
                - 0.5 * jnp.einsum("kij,jl->il", ctc_, rho)
                - 0.5 * jnp.einsum("ij,kjl->il", rho, ctc_)
            )
            return dr.ravel()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax")
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(_lindblad_rhs),
                diffrax.Tsit5(),
                t0=float(times_int[0]),
                t1=float(times_int[-1]),
                dt0=None,
                y0=jnp.array(rho0.full().ravel()),
                saveat=diffrax.SaveAt(ts=jnp.array(times_int)),
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
                max_steps=200_000,
                args=(H_jax, c_jax, cd_jax, ctc_jax),
            )

        ys = np.array(sol.ys)                        # (T, dim_full²) complex
        rhos_full = ys.reshape(-1, dim_full, dim_full)
        # Batched ptrace over modes: rho_el[t,i,j] = Σ_m rho[t, i·n_m_sq+m, j·n_m_sq+m]
        rho_el_np = np.einsum(
            "tikjk->tij",
            rhos_full.reshape(-1, DIM_EL, n_m_sq, DIM_EL, n_m_sq),
        )
        rhos_el = [qt.Qobj(rho_el_np[t], dims=[[DIM_EL], [DIM_EL]])
                   for t in range(len(times_int))]
    else:
        # CSR Liouvillian + scipy LSODA — runs on CPU (used by the parallel scans)
        L = qt.liouvillian(H, c_ops)
        result = qt.mesolve(
            L, rho0, times_int, c_ops=[],
            options={"method": "adams", "nsteps": 50000, "rtol": 1e-8, "atol": 1e-10},
        )
        rhos_el = [rho.ptrace([0]) for rho in result.states]

    times_fs = times_int / (2.0 * np.pi * C_FS)
    return times_fs, rhos_el


def population_vibronic(rhos_el: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from the reduced electronic density matrices."""
    return np.array([rho.full()[site, site].real for rho in rhos_el])


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
