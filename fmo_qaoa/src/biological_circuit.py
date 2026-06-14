"""Build the biological circuit scaffold used in the QBio experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qiskit import QuantumCircuit


def _validate_two_values(name: str, values: Sequence[float]) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly 2 values.")


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
    _validate_two_values("site_energies", site_energies)
    _validate_two_values("vib_freqs", vib_freqs)
    _validate_two_values("coupling_g", coupling_g)

    circuit = QuantumCircuit(4, name="biological_bus")
    system_qubits = [0, 1]
    ancilla_qubits = [2, 3]

    circuit.h(system_qubits[0])
    circuit.h(system_qubits[1])

    for qubit_index, energy in zip(system_qubits, site_energies):
        circuit.rz(energy, qubit_index)

    circuit.cx(system_qubits[0], system_qubits[1])
    circuit.rz(coupling_J, system_qubits[1])
    circuit.cx(system_qubits[0], system_qubits[1])

    for qubit_index, vib_freq in zip(ancilla_qubits, vib_freqs):
        circuit.rx(vib_freq, qubit_index)

    for system_qubit, ancilla_qubit, coupling in zip(system_qubits, ancilla_qubits, coupling_g):
        circuit.cp(coupling, system_qubit, ancilla_qubit)
        circuit.cx(ancilla_qubit, system_qubit)
        circuit.rz(coupling, system_qubit)

    circuit.barrier()
    return circuit


def create_fmo_vibronic_bus(
    site_energies: Sequence[float],
    couplings: Sequence[Sequence[float]],
    vib_freqs: Sequence[float],
    coupling_g: Sequence[float],
    coupling_threshold: float = 0.1,
) -> QuantumCircuit:
    """Create the 8-system-qubit + 2-ancilla FMO biological circuit (10 qubits total).

    Qubits 0-7 are BChla system sites; qubits 8-9 are two global vibronic modes.
    Only excitonic couplings with |J| > coupling_threshold are applied in the bus.
    """
    n_sys = 8
    circuit = QuantumCircuit(10, name="fmo_biological_bus")
    system_qubits = list(range(n_sys))
    ancilla_qubits = [8, 9]

    for q in system_qubits:
        circuit.h(q)

    for q, e in zip(system_qubits, site_energies):
        if e:
            circuit.rz(e, q)

    for i in range(n_sys):
        for j in range(i + 1, n_sys):
            J = couplings[i][j]
            if abs(J) > coupling_threshold:
                circuit.cx(system_qubits[i], system_qubits[j])
                circuit.rz(J, system_qubits[j])
                circuit.cx(system_qubits[i], system_qubits[j])

    for aq, freq in zip(ancilla_qubits, vib_freqs):
        circuit.rx(freq, aq)

    for sq in system_qubits:
        for aq, g in zip(ancilla_qubits, coupling_g):
            circuit.cp(g, sq, aq)
            circuit.cx(aq, sq)
            circuit.rz(g, sq)

    circuit.barrier()
    return circuit


def save_biological_circuit_diagram(output_path: str | Path) -> Path:
    """Render the FMO 8-site biological circuit to an image file."""
    _diag = (12405.0, 12530.0, 12210.0, 12320.0, 12480.0, 12630.0, 12440.0, 12430.0)
    _mean = sum(_diag) / 8
    _scale = 100.0
    _h_rows = (
        (0,  1, -87.7), (0,  2,   5.5), (0,  3,  -5.9), (0,  4,   6.7),
        (0,  5, -13.7), (0,  6,  -9.9), (0,  7,  21.0),
        (1,  2,  30.8), (1,  3,   8.2), (1,  4,   0.7), (1,  5,  11.8),
        (1,  6,   4.3), (1,  7,  42.0),
        (2,  3, -53.5), (2,  4,  -2.2), (2,  5,  -9.6), (2,  6,   6.0),
        (2,  7,   0.6),
        (3,  4, -70.7), (3,  5, -17.0), (3,  6, -63.3), (3,  7,  -1.3),
        (4,  5,  81.1), (4,  6,  -1.3), (4,  7,   3.3),
        (5,  6,  39.7), (5,  7,  -7.9),
        (6,  7,  -9.3),
    )
    site_energies = [(_d - _mean) / _scale for _d in _diag]
    couplings: list[list[float]] = [[0.0] * 8 for _ in range(8)]
    for i, j, v_cm in _h_rows:
        v = v_cm / _scale
        couplings[i][j] = v
        couplings[j][i] = v

    circuit = create_fmo_vibronic_bus(
        site_energies=site_energies,
        couplings=couplings,
        vib_freqs=(0.8, 1.2),
        coupling_g=(0.2, 0.15),
    )
    figure = circuit.draw(output="mpl")
    output_path = Path(output_path)
    figure.savefig(output_path, bbox_inches="tight")
    return output_path


if __name__ == "__main__":
    save_biological_circuit_diagram(Path("biological_circuit.png"))
