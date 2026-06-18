"""
Charge-density-coupling (CDC) recomputation of FMO site-energy *shifts* when a
pigment is moved — the atomistic electrostatic part of the position scan.

Physics (Adolphs & Renger, Biophys. J. 91, 2778 (2006); Müh et al. PNAS 104,
16862 (2007)): the 0-0 transition energy of pigment m shifts when its
transition-induced charge redistribution Δq_m (= excited − ground partial
charges) interacts with the electrostatic field of its environment (protein +
the other pigments' ground-state charges):

    V_m = C · Σ_{k∈m} Δq_{m,k} Σ_{l∈env(m)} Q_l / (ε · |r_{m,k} − r_l|)

with C = e²/(4πε₀) = 116140 cm⁻¹·Å.  We use this only for the *shift* relative
to the native geometry and add it to the published site energy:

    ε_m(geometry) = ε_m^published + [V_m(geometry) − V_m(native)]

so the native geometry reproduces the published Hamiltonian exactly and the
displacement physics is captured at the Coulomb level.  Δq conserves charge
(net 0), so V_m is a well-defined multipole–field interaction.

All heavy Coulomb sums are batched and run on the GPU via JAX when available
(scan_pigment_grid for per-pigment heatmaps, site_energies_batch for the global
optimiser's whole population at once); a chunked numpy path is the fallback.

Approximations (stated honestly): static protein (no relaxation around the moved
pigment), no spectral re-refinement, point charges with a single effective
dielectric ε (DIELECTRIC), no exchange/charge-transfer terms.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_C_CM = 116140.0          # e²/(4πε₀) in cm⁻¹·Å
DIELECTRIC = 2.0          # effective relative permittivity (protein interior)

_HERE = Path(__file__).parent
_d = np.load(_HERE / "fmo_atoms.npz")
PROT_XYZ = _d["prot_xyz"]      # (Np, 3)
PROT_Q   = _d["prot_q"]        # (Np,)
PIG_XYZ  = _d["pig_xyz"]       # (8, Na, 3) native pigment atom coords
PIG_QG   = _d["pig_qg"]        # (Na,) ground-state charges
PIG_DQ   = _d["pig_dq"]        # (Na,) Δq = excited − ground
N_PIG    = PIG_XYZ.shape[0]
N_AT     = PIG_XYZ.shape[1]

try:
    from gpu_utils import gpu_active
except Exception:                                  # pragma: no cover
    def gpu_active():
        return False


# ── Batched all-pigment CDC potential (GPU) ───────────────────────────────────

def _batch_jax():
    import jax, jax.numpy as jnp

    pig_xyz = jnp.asarray(PIG_XYZ); prot_xyz = jnp.asarray(PROT_XYZ)
    prot_q = jnp.asarray(PROT_Q); pig_qg = jnp.asarray(PIG_QG); pig_dq = jnp.asarray(PIG_DQ)
    not_self = 1.0 - jnp.eye(N_PIG)

    @jax.jit
    def kernel(disp):                              # disp (B,8,3) -> V (B,8)
        pos = pig_xyz[None] + disp[:, :, None, :]  # (B,8,Na,3)
        B = disp.shape[0]

        def prot_atom(carry, k):
            d = pos[:, :, k, :][:, :, None, :] - prot_xyz[None, None]
            inv = jax.lax.rsqrt((d * d).sum(-1))   # (B,8,Np)
            return carry + pig_dq[k] * (inv @ prot_q), None
        Vprot, _ = jax.lax.scan(prot_atom, jnp.zeros((B, N_PIG)), jnp.arange(N_AT))

        def pp_atom(carry, k):
            d = pos[:, :, k, :][:, :, None, None, :] - pos[:, None, :, :, :]
            d2 = (d * d).sum(-1)                    # (B,8m,8n,Na)
            d2 = jnp.where((not_self == 0)[None, :, :, None], 1.0, d2)
            inv = jax.lax.rsqrt(d2) * not_self[None, :, :, None]
            return carry + pig_dq[k] * jnp.einsum("bmnj,j->bm", inv, pig_qg), None
        Vpp, _ = jax.lax.scan(pp_atom, jnp.zeros((B, N_PIG)), jnp.arange(N_AT))
        return _C_CM / DIELECTRIC * (Vprot + Vpp)
    return kernel


def _batch_numpy(disp, chunk: int = 64):           # disp (B,8,3) -> (B,8)
    """Chunked over the batch so the (chunk,8,Nprot) tensors stay small."""
    B = disp.shape[0]
    not_self = 1.0 - np.eye(N_PIG)
    out = np.zeros((B, N_PIG))
    for i in range(0, B, chunk):
        d_chunk = disp[i:i + chunk]
        pos = PIG_XYZ[None] + d_chunk[:, :, None, :]
        V = np.zeros((d_chunk.shape[0], N_PIG))
        for k in range(N_AT):
            d = pos[:, :, k, :][:, :, None, :] - PROT_XYZ[None, None]
            inv = 1.0 / np.sqrt((d * d).sum(-1))
            V += PIG_DQ[k] * (inv @ PROT_Q)
            d = pos[:, :, k, :][:, :, None, None, :] - pos[:, None, :, :, :]
            d2 = (d * d).sum(-1)
            d2[:, np.arange(N_PIG), np.arange(N_PIG), :] = 1.0
            inv = (1.0 / np.sqrt(d2)) * not_self[None, :, :, None]
            V += PIG_DQ[k] * np.einsum("bmnj,j->bm", inv, PIG_QG)
        out[i:i + chunk] = _C_CM / DIELECTRIC * V
    return out


_KERNEL = None
def _potential_batch(disp: np.ndarray) -> np.ndarray:
    """(B,8,3) rigid displacements → (B,8) CDC potentials V_m (cm⁻¹)."""
    global _KERNEL
    if gpu_active():
        if _KERNEL is None:
            _KERNEL = _batch_jax()
        import jax.numpy as jnp
        return np.asarray(_KERNEL(jnp.asarray(disp)))
    return _batch_numpy(np.asarray(disp, float))


_V_NATIVE = _potential_batch(np.zeros((1, N_PIG, 3)))[0]


def site_energies_batch(disp_batch: np.ndarray) -> np.ndarray:
    """CDC site-energy shifts (cm⁻¹) for a batch of (B,8,3) displacements → (B,8)."""
    return _potential_batch(np.asarray(disp_batch, float)) - _V_NATIVE[None]


def delta_site_energies(displacements: np.ndarray) -> np.ndarray:
    """CDC site-energy shifts (cm⁻¹) for a single (8,3) displacement → (8,)."""
    return site_energies_batch(np.asarray(displacements, float)[None])[0]


# ── Per-pigment grid (one pigment moves) ──────────────────────────────────────

def scan_pigment_grid(p: int, disps: np.ndarray) -> np.ndarray:
    """
    Batched CDC shifts for a grid of displacements of a single pigment p.
    disps : (G, 3).  Returns (G, 8) site-energy shifts.
    """
    G = disps.shape[0]
    full = np.zeros((G, N_PIG, 3))
    full[:, p, :] = disps
    return site_energies_batch(full)


if __name__ == "__main__":
    import time
    print(f"protein atoms {len(PROT_Q)}, pigment atoms/BChl {N_AT}, ε={DIELECTRIC}, "
          f"GPU={gpu_active()}")
    print(f"V_native (cm⁻¹): {np.round(_V_NATIVE,1)}")
    for p, mag in [(2, 3.0), (0, 3.0), (7, 6.0)]:
        d = np.zeros((8, 3)); d[p, 0] = mag
        s = delta_site_energies(d)
        print(f"  move BChl{p+1} +{mag:.0f}Å: Δε_self={s[p]:+.1f}  "
              f"max|Δε_other|={np.abs(np.delete(s,p)).max():.1f} cm⁻¹")
    # timing: batched optimiser-style eval
    B = 256
    db = np.random.uniform(-3, 3, (B, 8, 3))
    site_energies_batch(db)  # warmup/JIT
    t0 = time.time(); site_energies_batch(db); dt = time.time() - t0
    print(f"\nbatched {B} geometries: {dt*1000:.0f} ms ({B/dt:.0f} geoms/s)")
