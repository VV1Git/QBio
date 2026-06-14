"""Build the biological circuit scaffold used in the QBio experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qiskit import QuantumCircuit


def _validate_four_values(name: str, values: Sequence[float]) -> None:
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly 4 values for the 4-qubit scaffold.")


def create_vibronic_bus(
    site_energies: Sequence[float],
    coupling_J: float,
    vib_freqs: Sequence[float],
    coupling_g: Sequence[float],
) -> QuantumCircuit:
    """Create the 2-system-qubit plus 2-ancilla biological circuit.

    The first two qubits represent the system, and the final two qubits represent
    the vibrational ancillas used to emulate a vibronic bus.
    """

    _validate_four_values("site_energies", site_energies)
    _validate_four_values("vib_freqs", vib_freqs)
    _validate_four_values("coupling_g", coupling_g)

    circuit = QuantumCircuit(4, name="biological_bus")
    system_qubits = [0, 1]
    ancilla_qubits = [2, 3]

    for qubit_index, energy in zip(system_qubits, site_energies[:2]):
        circuit.rz(energy, qubit_index)

    circuit.cx(system_qubits[0], system_qubits[1])
    circuit.rz(coupling_J, system_qubits[1])
    circuit.cx(system_qubits[0], system_qubits[1])

    for qubit_index, vib_freq in zip(ancilla_qubits, vib_freqs[2:]):
        circuit.rx(vib_freq, qubit_index)

    for system_qubit, ancilla_qubit, coupling in zip(system_qubits, ancilla_qubits, coupling_g[:2]):
        circuit.cp(coupling, system_qubit, ancilla_qubit)
        circuit.cx(ancilla_qubit, system_qubit)
        circuit.rz(coupling, system_qubit)

    circuit.barrier()
    return circuit


def save_biological_circuit_diagram(output_path: str | Path) -> Path:
    """Render the biological circuit to an image file for quick inspection."""

    circuit = create_vibronic_bus(
        site_energies=(0.8, 1.1, 0.0, 0.0),
        coupling_J=0.35,
        vib_freqs=(0.0, 0.0, 0.6, 0.9),
        coupling_g=(0.25, 0.3, 0.0, 0.0),
    )
    figure = circuit.draw(output="mpl")
    output_path = Path(output_path)
    figure.savefig(output_path, bbox_inches="tight")
    return output_path


if __name__ == "__main__":
    save_biological_circuit_diagram(Path("biological_circuit.png"))
