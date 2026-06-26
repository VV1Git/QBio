"""
Structured (vibronic) bath model for the 8-site FMO complex — showcase only.

Two intramolecular bacteriochlorophyll-a modes are explicitly quantised and
coupled to the BChl 3–4 dimer, which forms the lowest exciton states where
vibronic coherence is most relevant.  The remaining weaker modes plus the
phonon continuum are folded into a residual Ohmic bath that drives pure
electronic dephasing.

Mode parameters — the ~180 cm^-1 intramolecular BChl a vibration that is
NEAR-RESONANT with the FMO exciton energy gaps and is the mode implicated in
vibronically-assisted transfer / long-lived coherence (Adolphs & Renger,
Biophys. J. 91, 2778 (2006), effective high-energy mode S = 0.22, w = 180 cm^-1;
see also Thyrhaug et al., Nat. Chem. 2018 / "tuning quantum-mechanical mixing",
PNAS 115, 2018).  It is coupled to BOTH partners of the strongly-coupled
BChl 3-4 dimer, which forms the lowest excitons where the resonance lives:
    Mode 1 : w1 = 180 cm^-1, Huang-Rhys S1 = 0.22  (coupled to BChl 3)
    Mode 2 : w2 = 180 cm^-1, Huang-Rhys S2 = 0.22  (coupled to BChl 4)
    g_k = w_k * sqrt(S_k)  ->  g1 = g2 ~ 84 cm^-1
(The earlier 770/243 cm^-1 modes from Rätsep & Freiberg 2007 are off-resonant
and far more weakly coupled, so they do not drive the coherence physics.)

Because the full vibronic Hilbert space is 8*(n_max+1)^2-dimensional, this model
is used only for single-geometry showcase trajectories (Fig. 3 dynamics and
Fig. 4 coherence spectra), never for the position scans.
"""

from __future__ import annotations

import warnings

import numpy as np
import qutip as qt

from hamiltonian import build_electronic_H
from fmo_data import SITE_ENERGIES_CM, N_SITES, ENTRY_SITES
from spectral_density import LAMBDA_CM, GAMMA_CM, TEMPERATURE_K

# ── Mode constants ─────────────────────────────────────────────────────────────

OMEGA1_CM = 180.0   # resonant mode (cm^-1), BChl 3 — Adolphs & Renger 2006
OMEGA2_CM = 180.0   # resonant mode (cm^-1), BChl 4 — near-resonant w/ exciton gap
S1 = 0.22           # Huang-Rhys factor, mode 1 (Adolphs-Renger effective mode)
S2 = 0.22           # Huang-Rhys factor, mode 2
MODE1_SITE = 2      # BChl 3  (lower partner of the coherent BChl 3-4 dimer)
MODE2_SITE = 3      # BChl 4
N_MAX = 2           # Fock truncation per mode (dim = 8*(N_MAX+1)^2 = 72)

GAMMA_VIB_CM = 20.0   # vibrational damping rate (cm^-1), ~265 fs
_HBAR_OVER_KB = 1.4388  # K·cm
C_FS = 3e-5             # speed of light in cm/fs


def _mode_ops(d: int):
    a = qt.destroy(d)
    return a, a.dag(), qt.num(d)


def build_vibronic_H(displacements: np.ndarray | None = None,
                     n_max: int = N_MAX) -> qt.Qobj:
    """Full vibronic Hamiltonian: H_el (x) mode1 (x) mode2."""
    dim_m = n_max + 1
    H_el_np = build_electronic_H(displacements)
    H_el_np -= np.mean(SITE_ENERGIES_CM) * np.eye(N_SITES)
    H_el = qt.Qobj(H_el_np)

    I_el = qt.qeye(N_SITES)
    I_m  = qt.qeye(dim_m)
    a1, a1d, N1 = _mode_ops(dim_m)
    a2, a2d, N2 = _mode_ops(dim_m)

    g1 = OMEGA1_CM * np.sqrt(S1)
    g2 = OMEGA2_CM * np.sqrt(S2)
    proj1 = qt.ket2dm(qt.basis(N_SITES, MODE1_SITE))
    proj2 = qt.ket2dm(qt.basis(N_SITES, MODE2_SITE))

    H = (
        qt.tensor(H_el,        I_m,            I_m)
      + qt.tensor(I_el, OMEGA1_CM * N1,        I_m)
      + qt.tensor(I_el,        I_m,  OMEGA2_CM * N2)
      + qt.tensor(g1 * proj1, a1 + a1d,        I_m)
      + qt.tensor(g2 * proj2,        I_m, a2 + a2d)
    )
    return H


def _collapse_operators(temperature, lambda_, gamma_bath, gamma_vib,
                        n_max: int = N_MAX) -> list[qt.Qobj]:
    """Vibrational thermal damping + residual-Ohmic electronic dephasing."""
    dim_m = n_max + 1
    I_el = qt.qeye(N_SITES)
    I_m  = qt.qeye(dim_m)
    a1, a1d, _ = _mode_ops(dim_m)
    a2, a2d, _ = _mode_ops(dim_m)

    def n_th(omega):
        bo = _HBAR_OVER_KB * omega / temperature
        return 0.0 if bo > 500 else 1.0 / np.expm1(bo)

    n1, n2 = n_th(OMEGA1_CM), n_th(OMEGA2_CM)
    c_ops = [
        np.sqrt(gamma_vib * (n1 + 1.0)) * qt.tensor(I_el, a1,  I_m),
        np.sqrt(gamma_vib *  n1)        * qt.tensor(I_el, a1d, I_m),
        np.sqrt(gamma_vib * (n2 + 1.0)) * qt.tensor(I_el, I_m, a2),
        np.sqrt(gamma_vib *  n2)        * qt.tensor(I_el, I_m, a2d),
    ]

    # The explicit 180 cm⁻¹ mode is the structured peak ADDED on top of the Drude
    # continuum (J_structured = Drude + mode), so the residual pure-dephasing bath
    # is the full low-frequency Drude reorganisation λ — not λ minus the modes.
    lambda_res = lambda_
    gamma_phi  = 2.0 * lambda_res * temperature / (_HBAR_OVER_KB * gamma_bath)
    if gamma_phi > 0.0:
        sqrt_phi = np.sqrt(gamma_phi)
        for i in range(N_SITES):
            c_ops.append(sqrt_phi * qt.tensor(qt.ket2dm(qt.basis(N_SITES, i)), I_m, I_m))
    return c_ops


def run_structured(displacements: np.ndarray | None = None,
                   initial_site: int = ENTRY_SITES[0],
                   t_end: float = 5000.0, n_steps: int = 500,
                   temperature: float = TEMPERATURE_K,
                   lambda_: float = LAMBDA_CM, gamma_bath: float = GAMMA_CM,
                   gamma_vib: float = GAMMA_VIB_CM,
                   n_max: int = N_MAX) -> tuple[np.ndarray, list[qt.Qobj]]:
    """
    Propagate the vibronic Hamiltonian and return the reduced 8x8 electronic
    density matrices.  Uses a direct Lindblad RHS on dim x dim matrices via
    diffrax on the GPU; otherwise sparse QuTiP mesolve on the CPU (no dense
    Liouvillian is ever formed — that would be (8*(n_max+1)^2)^2 elements).
    """
    from gpu_utils import gpu_active

    dim_m = n_max + 1
    dim_full = N_SITES * dim_m * dim_m
    n_m_sq = dim_m * dim_m

    _saved = qt.settings.core["default_dtype"]
    qt.settings.core["default_dtype"] = "CSR"
    try:
        H = build_vibronic_H(displacements, n_max=n_max)
        c_ops = _collapse_operators(temperature, lambda_, gamma_bath, gamma_vib,
                                    n_max=n_max)
        psi0 = qt.tensor(qt.basis(N_SITES, initial_site),
                         qt.basis(dim_m, 0), qt.basis(dim_m, 0))
        rho0 = psi0 * psi0.dag()
    finally:
        qt.settings.core["default_dtype"] = _saved

    t_max_int = t_end * 2.0 * np.pi * C_FS
    times_int = np.linspace(0.0, t_max_int, n_steps)

    if gpu_active():
        import jax.numpy as jnp
        import diffrax

        H_jax  = jnp.array(H.full())
        c_jax  = jnp.array([c.full() for c in c_ops])
        cd_jax = jnp.conj(c_jax).transpose(0, 2, 1)
        ctc    = jnp.einsum("kij,kjl->kil", cd_jax, c_jax)

        def _rhs(t, y, args):
            H_, c_, cd_, ctc_ = args
            rho = y.reshape(dim_full, dim_full)
            dr = -1j * (H_ @ rho - rho @ H_)
            dr = dr + (jnp.einsum("kij,jl,klm->im", c_, rho, cd_)
                       - 0.5 * jnp.einsum("kij,jl->il", ctc_, rho)
                       - 0.5 * jnp.einsum("ij,kjl->il", rho, ctc_))
            return dr.ravel()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Complex dtype support in Diffrax")
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(_rhs), diffrax.Tsit5(),
                t0=float(times_int[0]), t1=float(times_int[-1]), dt0=None,
                y0=jnp.array(rho0.full().ravel()),
                saveat=diffrax.SaveAt(ts=jnp.array(times_int)),
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
                max_steps=400_000, args=(H_jax, c_jax, cd_jax, ctc),
            )
        ys = np.array(sol.ys).reshape(-1, dim_full, dim_full)
        rho_el = np.einsum("tikjk->tij",
                           ys.reshape(-1, N_SITES, n_m_sq, N_SITES, n_m_sq))
        rhos_el = [qt.Qobj(rho_el[t], dims=[[N_SITES], [N_SITES]])
                   for t in range(len(times_int))]
    else:
        # Sparse mesolve (CSR) on the CPU — no dense Liouvillian formed.
        result = qt.mesolve(
            H, rho0, times_int, c_ops=c_ops,
            options={"method": "adams", "nsteps": 50000, "rtol": 1e-8, "atol": 1e-10},
        )
        rhos_el = [rho.ptrace([0]) for rho in result.states]

    return times_int / (2.0 * np.pi * C_FS), rhos_el


def population_vibronic(rhos_el: list[qt.Qobj], site: int) -> np.ndarray:
    """Extract P_site(t) from the reduced electronic density matrices."""
    return np.array([rho.full()[site, site].real for rho in rhos_el])


if __name__ == "__main__":
    print(f"Building 8-site vibronic H (n_max={N_MAX}, dim={N_SITES*(N_MAX+1)**2}) …")
    H = build_vibronic_H()
    print(f"  dims {H.shape}")
    print("Running structured dynamics (native geometry, 3 ps) …")
    t, rhos_el = run_structured(t_end=3000.0, n_steps=200)
    for s in (ENTRY_SITES[0], 2):
        P = population_vibronic(rhos_el, s)
        print(f"  P(BChl {s+1}, final) = {P[-1]:.3f}")
