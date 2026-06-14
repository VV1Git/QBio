"""Run the Step 4 noise-sweep experiment for the QBio repository."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel, phase_damping_error, thermal_relaxation_error
from qiskit_aer.primitives import EstimatorV2

from src.qaoa_optimizer import (
    TwoNodeIsingProblem,
    build_biological_qaoa_ansatz,
    build_standard_qaoa_ansatz,
)


RESULTS_DIR = Path("results")
JSON_RESULTS_PATH = RESULTS_DIR / "step4_results.json"
CSV_RESULTS_PATH = RESULTS_DIR / "step4_results.csv"

DEFAULT_P = 3
DEFAULT_GAMMAS = (0.9, 0.7, 0.5)
DEFAULT_BETAS = (0.4, 0.3, 0.2)
DEFAULT_SITE_ENERGIES = (0.8, 1.1, 0.0, 0.0)
DEFAULT_COUPLING_J = 0.35
DEFAULT_VIB_FREQS = (0.0, 0.0, 0.6, 0.9)
DEFAULT_COUPLING_G = (0.25, 0.3, 0.0, 0.0)
DEFAULT_NOISE_SCALES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_SHOTS = 4096
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class ExperimentResult:
    """One row of the Step 4 sweep."""

    noise_scale: float
    circuit_type: str
    estimated_cost: float
    baseline_cost: float
    optimal_cost: float
    cost_gap: float
    normalized_accuracy: float
    retained_accuracy: float


def build_cost_hamiltonian(num_qubits: int, problem: TwoNodeIsingProblem) -> SparsePauliOp:
    """Build the Ising cost Hamiltonian on the leading two qubits."""

    operator_terms: list[tuple[str, list[int], complex]] = []

    if problem.coupling:
        operator_terms.append(("ZZ", [0, 1], complex(problem.coupling)))
    if problem.local_fields[0]:
        operator_terms.append(("Z", [0], complex(problem.local_fields[0])))
    if problem.local_fields[1]:
        operator_terms.append(("Z", [1], complex(problem.local_fields[1])))

    if not operator_terms:
        operator_terms.append(("I", [0], complex(problem.offset)))

    operator = SparsePauliOp.from_sparse_list(operator_terms, num_qubits=num_qubits)
    if problem.offset:
        operator = operator + SparsePauliOp.from_list([("I" * num_qubits, complex(problem.offset))])
    return operator


def exact_optimum_energy(problem: TwoNodeIsingProblem) -> float:
    """Compute the exact minimum energy of the two-node Ising objective."""

    best_energy = math.inf
    for z0 in (1.0, -1.0):
        for z1 in (1.0, -1.0):
            energy = (
                problem.coupling * z0 * z1
                + problem.local_fields[0] * z0
                + problem.local_fields[1] * z1
                + problem.offset
            )
            best_energy = min(best_energy, energy)
    return best_energy


def _combined_gate_error(single_qubit_t1: float, single_qubit_t2: float, gate_time: float):
    """Compose thermal relaxation and dephasing into one single-qubit error."""

    thermal_error = thermal_relaxation_error(single_qubit_t1, single_qubit_t2, gate_time)

    inverse_tphi = max(0.0, (1.0 / single_qubit_t2) - (0.5 / single_qubit_t1))
    dephasing_probability = 0.0 if inverse_tphi == 0.0 else 1.0 - math.exp(-gate_time * inverse_tphi)
    dephasing_error = phase_damping_error(dephasing_probability)

    return thermal_error.compose(dephasing_error)


def build_noise_model(scale: float) -> NoiseModel | None:
    """Create a simple thermal-relaxation plus dephasing model for a noise scale."""

    if scale <= 0.0:
        return None

    base_t1 = 120_000.0
    base_t2 = 90_000.0
    effective_t1 = base_t1 / scale
    effective_t2 = base_t2 / scale

    single_qubit_gate_time = 50.0
    two_qubit_gate_time = 250.0

    single_qubit_error = _combined_gate_error(effective_t1, effective_t2, single_qubit_gate_time)
    two_qubit_single_error = _combined_gate_error(effective_t1, effective_t2, two_qubit_gate_time)
    two_qubit_error = two_qubit_single_error.tensor(two_qubit_single_error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ["rx", "rz", "h", "sx"])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cp"])
    return noise_model


def _run_estimator(circuit, observable, noise_model: NoiseModel | None, shots: int, seed: int) -> float:
    """Evaluate a circuit with Aer's Estimator primitive."""

    estimator = EstimatorV2(
        options={
            "backend_options": {"noise_model": noise_model},
            "run_options": {"shots": shots, "seed": seed},
        }
    )
    job = estimator.run([(circuit, observable)])
    result = job.result()
    return float(result[0].data.evs)


def _normalized_accuracy(estimated_cost: float, optimal_cost: float) -> float:
    """Turn cost error into a bounded accuracy score in [0, 1]."""

    return 1.0 / (1.0 + abs(estimated_cost - optimal_cost))


def _retained_accuracy(estimated_cost: float, optimal_cost: float, baseline_cost: float) -> float:
    """Measure how much of the no-noise performance is retained under noise."""

    baseline_gap = abs(baseline_cost - optimal_cost)
    if baseline_gap == 0.0:
        return 1.0

    current_gap = abs(estimated_cost - optimal_cost)
    retained = 1.0 - abs(current_gap - baseline_gap) / baseline_gap
    return max(0.0, min(1.0, retained))


def run_step4_experiment() -> list[ExperimentResult]:
    """Run the standard-versus-biological QAOA noise sweep."""

    problem = TwoNodeIsingProblem(local_fields=(0.15, -0.05), coupling=1.0, offset=0.0)
    optimal_cost = exact_optimum_energy(problem)

    standard_circuit = build_standard_qaoa_ansatz(DEFAULT_P, DEFAULT_GAMMAS, DEFAULT_BETAS, problem)
    biological_circuit = build_biological_qaoa_ansatz(
        DEFAULT_P,
        DEFAULT_GAMMAS,
        DEFAULT_BETAS,
        DEFAULT_SITE_ENERGIES,
        DEFAULT_COUPLING_J,
        DEFAULT_VIB_FREQS,
        DEFAULT_COUPLING_G,
        problem,
    )

    standard_observable = build_cost_hamiltonian(standard_circuit.num_qubits, problem)
    biological_observable = build_cost_hamiltonian(biological_circuit.num_qubits, problem)

    baseline_costs = {
        "standard": _run_estimator(standard_circuit, standard_observable, None, DEFAULT_SHOTS, DEFAULT_SEED),
        "biological": _run_estimator(biological_circuit, biological_observable, None, DEFAULT_SHOTS, DEFAULT_SEED),
    }

    results: list[ExperimentResult] = []

    for noise_scale in DEFAULT_NOISE_SCALES:
        noise_model = build_noise_model(noise_scale)

        standard_cost = _run_estimator(
            standard_circuit,
            standard_observable,
            noise_model,
            DEFAULT_SHOTS,
            DEFAULT_SEED,
        )
        biological_cost = _run_estimator(
            biological_circuit,
            biological_observable,
            noise_model,
            DEFAULT_SHOTS,
            DEFAULT_SEED,
        )

        for circuit_type, estimated_cost in (
            ("standard", standard_cost),
            ("biological", biological_cost),
        ):
            baseline_cost = baseline_costs[circuit_type]
            results.append(
                ExperimentResult(
                    noise_scale=noise_scale,
                    circuit_type=circuit_type,
                    estimated_cost=estimated_cost,
                    baseline_cost=baseline_cost,
                    optimal_cost=optimal_cost,
                    cost_gap=estimated_cost - optimal_cost,
                    normalized_accuracy=_normalized_accuracy(estimated_cost, optimal_cost),
                    retained_accuracy=_retained_accuracy(estimated_cost, optimal_cost, baseline_cost),
                )
            )

    return results


def save_results(results: Sequence[ExperimentResult]) -> None:
    """Persist the Step 4 results as JSON and CSV."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_RESULTS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump([asdict(result) for result in results], file_handle, indent=2)

    with CSV_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    results = run_step4_experiment()
    save_results(results)
    print(f"Saved {len(results)} rows to {JSON_RESULTS_PATH} and {CSV_RESULTS_PATH}")


if __name__ == "__main__":
    main()
