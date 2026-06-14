"""QAOA circuit builders for the QBio experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit

from src.biological_circuit import create_vibronic_bus


@dataclass(frozen=True)
class TwoNodeIsingProblem:
    """Minimal 2-variable Ising objective used for the QAOA benchmark."""

    local_fields: tuple[float, float] = (0.0, 0.0)
    coupling: float = 1.0
    offset: float = 0.0


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


def _apply_qaoa_layer(
    circuit: QuantumCircuit,
    gamma: float,
    beta: float,
    problem: TwoNodeIsingProblem,
    system_qubits: tuple[int, int],
) -> None:
    """Apply one cost layer and one mixer layer to the system qubits."""

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
    """Refresh the biological interaction between the system and ancillas."""

    for ancilla_qubit, vib_freq in zip(ancilla_qubits, vib_freqs[2:]):
        circuit.rx(vib_freq, ancilla_qubit)

    for system_qubit, ancilla_qubit, coupling in zip(system_qubits, ancilla_qubits, coupling_g[:2]):
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
    """Build the biological QAOA circuit on top of the vibronic bus scaffold."""

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


if __name__ == "__main__":
    standard = build_standard_qaoa_ansatz(p=1, gammas=(0.7,), betas=(0.4,))
    biological = build_biological_qaoa_ansatz(
        p=1,
        gammas=(0.7,),
        betas=(0.4,),
        site_energies=(0.8, 1.1, 0.0, 0.0),
        coupling_J=0.35,
        vib_freqs=(0.0, 0.0, 0.6, 0.9),
        coupling_g=(0.25, 0.3, 0.0, 0.0),
    )
    print(standard)
    print(biological)
