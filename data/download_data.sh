#!/usr/bin/env bash
# Downloads the two public FEP benchmark datasets used by DeltaBind-GNN.
# No registration or API keys required — both are public GitHub repos.

set -e

RAW_DIR="$(dirname "$0")/raw"
mkdir -p "$RAW_DIR"

echo ">>> Cloning Schrodinger public FEP+ benchmark..."
if [ ! -d "$RAW_DIR/schrodinger_fep_benchmark" ]; then
    git clone --depth 1 https://github.com/schrodinger/public_binding_free_energy_benchmark.git \
        "$RAW_DIR/schrodinger_fep_benchmark"
else
    echo "Already exists, skipping."
fi

echo ">>> Cloning Merck/Schindler FEP benchmark..."
if [ ! -d "$RAW_DIR/merck_fep_benchmark" ]; then
    git clone --depth 1 https://github.com/MCompChem/fep-benchmark.git \
        "$RAW_DIR/merck_fep_benchmark"
else
    echo "Already exists, skipping."
fi

echo ">>> Done. Raw data in: $RAW_DIR"
