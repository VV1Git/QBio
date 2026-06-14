"""QAOA circuit builders for the QBio experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit

from src.biological_circuit import create_vibronic_bus, create_fmo_vibronic_bus


# ---------------------------------------------------------------------------
# FMO Hamiltonian — Badu, Melnik & Singh (Applied Sciences 2020), Eq. (11).
# Units: cm⁻¹.  Diagonal = site energies; off-diagonal = excitonic couplings.
# H[4][7] = 3.3 (not 33) based on symmetry and Adolphs & Renger (2006).
# ---------------------------------------------------------------------------
_H_FMO_CM: tuple[tuple[float, ...], ...] = (
    (12405.0,  -87.7,    5.5,   -5.9,    6.7,  -13.7,   -9.9,   21.0),
    ( -87.7, 12530.0,   30.8,    8.2,    0.7,   11.8,    4.3,   42.0),
    (   5.5,   30.8, 12210.0,  -53.5,   -2.2,   -9.6,    6.0,    0.6),
    (  -5.9,    8.2,  -53.5, 12320.0,  -70.7,  -17.0,  -63.3,   -1.3),
    (   6.7,    0.7,   -2.2,  -70.7, 12480.0,   81.1,   -1.3,    3.3),
    ( -13.7,   11.8,   -9.6,  -17.0,   81.1, 12630.0,   39.7,   -7.9),
    (  -9.9,    4.3,    6.0,  -63.3,   -1.3,   39.7, 12440.0,   -9.3),
    (  21.0,   42.0,    0.6,   -1.3,    3.3,   -7.9,   -9.3, 12430.0),
)
_FMO_SCALE = 100.0          # cm⁻¹ → dimensionless QAOA angle units
_FMO_COUPLING_THRESHOLD = 0.1  # skip |J| < 0.1 in QAOA circuit layers


# ---------------------------------------------------------------------------
# Problem dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoNodeIsingProblem:
    """Minimal 2-variable Ising objective used for the QAOA benchmark."""

    local_fields: tuple[float, float] = (0.0, 0.0)
    coupling: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class EightSiteFMOProblem:
    """8-site FMO excitonic Hamiltonian mapped to an Ising spin model.

    All values are in units of _FMO_SCALE (100 cm⁻¹).
    local_fields[i]  = (E_i − E_mean) / _FMO_SCALE  (Z term for site i)
    couplings[i][j]  = H_FMO[i][j]  / _FMO_SCALE   (ZZ term for pair i,j)
    offset           = E_mean / _FMO_SCALE           (global energy shift)
    """

    local_fields: tuple[float, ...]
    couplings: tuple[tuple[float, ...], ...]
    offset: float = 0.0


def make_fmo_problem() -> EightSiteFMOProblem:
    """Derive the FMO Ising problem from the Badu et al. (2020) Hamiltonian."""
    n = 8
    diag = [_H_FMO_CM[i][i] for i in range(n)]
    mean_e = sum(diag) / n
    local_fields = tuple((d - mean_e) / _FMO_SCALE for d in diag)
    couplings = tuple(
        tuple(0.0 if i == j else _H_FMO_CM[i][j] / _FMO_SCALE for j in range(n))
        for i in range(n)
    )
    return EightSiteFMOProblem(
        local_fields=local_fields,
        couplings=couplings,
        offset=mean_e / _FMO_SCALE,
    )


# ---------------------------------------------------------------------------
# Shared circuit helpers
# ---------------------------------------------------------------------------

def _validate_layer_parameters(p: int, gammas: Sequence[float], betas: Sequence[float]) -> None:
    if p < 1:
        raise ValueError("p must be at least 1.")
    if len(gammas) != p:
        raise ValueError("gammas must contain exactly p values.")
    if len(betas) != p:
        raise ValueError("betas must contain exactly p values.")


def _apply_zz_cost_rotation(circuit: QuantumCircuit, qubit_0: int, qubit_1: int, angle: float) -> None:
    """Apply a ZZ rotation using the standard CX-RZ-CX decomposition."""
    circuit.cx(qubit_0, qubit_1)
    circuit.rz(angle, qubit_1)
    circuit.cx(qubit_0, qubit_1)


# ---------------------------------------------------------------------------
# 2-node QAOA (retained for reference)
# ---------------------------------------------------------------------------

def _apply_qaoa_layer(
    circuit: QuantumCircuit,
    gamma: float,
    beta: float,
    problem: TwoNodeIsingProblem,
    system_qubits: tuple[int, int],
) -> None:
    """Apply one cost layer and one mixer layer to the 2 system qubits."""
    first_qubit, second_qubit = system_qubits
    if problem.local_fields[0]:
        circuit.rz(2.0 * gamma * problem.local_fields[0], first_qubit)
    if problem.local_fields[1]:
        circuit.rz(2.0 * gamma * problem.local_fields[1], second_qubit)
    _apply_zz_cost_rotation(circuit, first_qubit, second_qubit, 2.0 * gamma * problem.coupling)
    circuit.rx(2.0 * beta, first_qubit)
    circuit.rx(2.0 * beta, second_qubit)


def _apply_vibronic_recoupling(
    circuit: QuantumCircuit,
    coupling_g: Sequence[float],
    vib_freqs: Sequence[float],
    system_qubits: tuple[int, int],
    ancilla_qubits: tuple[int, int],
) -> None:
    """Refresh the biological interaction between the 2 system and 2 ancilla qubits."""
    for ancilla_qubit, vib_freq in zip(ancilla_qubits, vib_freqs):
        circuit.rx(vib_freq, ancilla_qubit)
    for system_qubit, ancilla_qubit, coupling in zip(system_qubits, ancilla_qubits, coupling_g):
        circuit.cp(coupling, system_qubit, ancilla_qubit)
        circuit.cx(ancilla_qubit, system_qubit)
        circuit.rz(coupling, system_qubit)


def build_standard_qaoa_ansatz(
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    problem: TwoNodeIsingProblem | None = None,
) -> QuantumCircuit:
    """Build a standard 2-qubit QAOA ansatz for the benchmark Ising problem."""
    _validate_layer_parameters(p, gammas, betas)
    problem = problem or TwoNodeIsingProblem()
    circuit = QuantumCircuit(2, name="standard_qaoa")
    circuit.h(0)
    circuit.h(1)
    for gamma, beta in zip(gammas, betas):
        _apply_qaoa_layer(circuit, gamma, beta, problem, (0, 1))
    circuit.global_phase = problem.offset
    return circuit


def build_biological_qaoa_ansatz(
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    site_energies: Sequence[float],
    coupling_J: float,
    vib_freqs: Sequence[float],
    coupling_g: Sequence[float],
    problem: TwoNodeIsingProblem | None = None,
) -> QuantumCircuit:
    """Build the biological QAOA circuit on top of the 2-node vibronic bus scaffold."""
    _validate_layer_parameters(p, gammas, betas)
    problem = problem or TwoNodeIsingProblem()
    circuit = create_vibronic_bus(site_energies, coupling_J, vib_freqs, coupling_g)
    system_qubits = (0, 1)
    ancilla_qubits = (2, 3)
    for gamma, beta in zip(gammas, betas):
        _apply_qaoa_layer(circuit, gamma, beta, problem, system_qubits)
        _apply_vibronic_recoupling(circuit, coupling_g, vib_freqs, system_qubits, ancilla_qubits)
    circuit.global_phase = problem.offset
    circuit.name = "biological_qaoa"
    return circuit


# ---------------------------------------------------------------------------
# 8-site FMO QAOA  (8 system qubits + 2 global vibronic ancilla = 10 qubits)
# ---------------------------------------------------------------------------

def _apply_fmo_qaoa_layer(
    circuit: QuantumCircuit,
    gamma: float,
    beta: float,
    problem: EightSiteFMOProblem,
    system_qubits: Sequence[int],
) -> None:
    """Apply one FMO cost layer (Z + ZZ) and mixer (RX) to the 8 system qubits.

    ZZ terms with |J| < _FMO_COUPLING_THRESHOLD are skipped for circuit efficiency.
    """
    n = len(system_qubits)
    for i, qi in enumerate(system_qubits):
        h_i = problem.local_fields[i]
        if h_i:
            circuit.rz(2.0 * gamma * h_i, qi)
    for i in range(n):
        for j in range(i + 1, n):
            J_ij = problem.couplings[i][j]
            if abs(J_ij) > _FMO_COUPLING_THRESHOLD:
                _apply_zz_cost_rotation(
                    circuit, system_qubits[i], system_qubits[j], 2.0 * gamma * J_ij
                )
    for qi in system_qubits:
        circuit.rx(2.0 * beta, qi)


def _apply_fmo_vibronic_recoupling(
    circuit: QuantumCircuit,
    coupling_g: Sequence[float],
    vib_freqs: Sequence[float],
    system_qubits: Sequence[int],
    ancilla_qubits: Sequence[int],
) -> None:
    """Refresh the FMO vibronic coupling: each system qubit couples to both modes."""
    for aq, freq in zip(ancilla_qubits, vib_freqs):
        circuit.rx(freq, aq)
    for sq in system_qubits:
        for aq, g in zip(ancilla_qubits, coupling_g):
            circuit.cp(g, sq, aq)
            circuit.cx(aq, sq)
            circuit.rz(g, sq)


def build_fmo_standard_qaoa_ansatz(
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    problem: EightSiteFMOProblem,
) -> QuantumCircuit:
    """Build a standard 8-qubit QAOA ansatz for the FMO Ising problem."""
    _validate_layer_parameters(p, gammas, betas)
    circuit = QuantumCircuit(8, name="fmo_standard_qaoa")
    system_qubits = list(range(8))
    for q in system_qubits:
        circuit.h(q)
    for gamma, beta in zip(gammas, betas):
        _apply_fmo_qaoa_layer(circuit, gamma, beta, problem, system_qubits)
    circuit.global_phase = problem.offset
    return circuit


def build_fmo_biological_qaoa_ansatz(
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    vib_freqs: Sequence[float],
    coupling_g: Sequence[float],
    problem: EightSiteFMOProblem,
) -> QuantumCircuit:
    """Build the FMO biological QAOA circuit: 8 system + 2 vibronic ancilla = 10 qubits."""
    _validate_layer_parameters(p, gammas, betas)
    circuit = create_fmo_vibronic_bus(
        site_energies=list(problem.local_fields),
        couplings=problem.couplings,
        vib_freqs=vib_freqs,
        coupling_g=coupling_g,
    )
    system_qubits = list(range(8))
    ancilla_qubits = [8, 9]
    for gamma, beta in zip(gammas, betas):
        _apply_fmo_qaoa_layer(circuit, gamma, beta, problem, system_qubits)
        _apply_fmo_vibronic_recoupling(circuit, coupling_g, vib_freqs, system_qubits, ancilla_qubits)
    circuit.global_phase = problem.offset
    circuit.name = "fmo_biological_qaoa"
    return circuit


if __name__ == "__main__":
    fmo = make_fmo_problem()
    vib_freqs = (0.8, 1.2)
    coupling_g = (0.2, 0.15)
    std = build_fmo_standard_qaoa_ansatz(p=1, gammas=(0.7,), betas=(0.4,), problem=fmo)
    bio = build_fmo_biological_qaoa_ansatz(
        p=1, gammas=(0.7,), betas=(0.4,),
        vib_freqs=vib_freqs, coupling_g=coupling_g, problem=fmo,
    )
    print(std)
    print(bio)
