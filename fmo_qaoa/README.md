# QBio

This repository currently includes the setup for the first two implementation steps of the QBio experiment.

## Step 1

The project uses a local virtual environment at `.venv/` and records its installed packages in [requirements.txt](requirements.txt).

Install the dependency stack with:

```bash
cd /Users/vedant/Documents/QBio
.venv/bin/python -m pip install -r requirements.txt
```

## Step 2

The biological circuit scaffold lives in [src/biological_circuit.py](src/biological_circuit.py).

To generate a circuit diagram image, run:

```bash
cd /Users/vedant/Documents/QBio
.venv/bin/python -m src.biological_circuit
```

This creates `biological_circuit.png` in the repository root.

## Step 5

The analysis notebook lives in [notebooks/data_analysis.ipynb](notebooks/data_analysis.ipynb).

It reads [results/step4_results.json](results/step4_results.json), plots the noise-retention score for both circuits, and saves the comparison plot to `results/step5_money_shot.png`.
