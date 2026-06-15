"""
Configure QuTiP to use the JAX/CUDA backend when available.

Install prerequisites (CUDA 12.x):
    pip install "jax[cuda12]" qutip-jax

Usage:
    from gpu_utils import setup_gpu
    setup_gpu()   # call once before creating any QuTiP objects
"""
from __future__ import annotations

_GPU_ACTIVE = False


def setup_gpu(verbose: bool = True) -> bool:
    """
    Activate the JAX CUDA backend for QuTiP.

    Must be called before any qt.Qobj / qt.mesolve / qt.brmesolve calls.
    Returns True if a CUDA GPU was found and the backend is active.
    """
    global _GPU_ACTIVE
    try:
        import jax
        jax.config.update("jax_enable_x64", True)

        gpu_devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpu_devs:
            if verbose:
                print("[GPU] JAX found no CUDA devices — falling back to CPU.")
            return False

        import qutip_jax          # registers 'jax' and 'jaxdia' dtypes with QuTiP
        import qutip as qt
        # 'jax' (dense) is safest across both mesolve and brmesolve
        qt.settings.core["default_dtype"] = "jax"
        _GPU_ACTIVE = True
        if verbose:
            print(f"[GPU] Active on {gpu_devs[0].device_kind} via JAX "
                  f"({gpu_devs[0]})")
        return True

    except ImportError as exc:
        if verbose:
            print(f"[GPU] Not available ({exc}).\n"
                  f"      Install with:  pip install 'jax[cuda12]' qutip-jax")
        return False


def gpu_active() -> bool:
    """Return True if setup_gpu() succeeded in a prior call."""
    return _GPU_ACTIVE
