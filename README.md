# DeltaBind-GNN

**A graph neural network surrogate for relative protein-ligand binding free energy (ΔΔG) prediction — approximating Free Energy Perturbation (FEP+) at a fraction of the compute cost.**

## Problem

Free Energy Perturbation (FEP) is the most accurate computational method for ranking how small structural changes to a ligand affect its binding affinity to a protein target. It's the gold standard used in pharma lead optimization — but each ligand-pair calculation costs hours of GPU/simulation time. Medicinal chemists routinely need to rank dozens to hundreds of candidate analogs before committing to synthesis and assay. There's a real bottleneck between "FEP is accurate" and "FEP is fast enough to guide every decision."

## Solution

DeltaBind-GNN learns to predict **relative binding free energy (ΔΔG) between pairs of congeneric ligands** (structurally similar analogs bound to the same target), directly from 3D structure — using a GNN encoder + pairwise comparison head trained on public FEP benchmark data. The pairwise framing mirrors what FEP itself computes (a difference between two ligands in the same pocket), which lets systematic pocket-level error cancel and makes the problem tractable with a few hundred training examples instead of millions.

We benchmark directly against FEP+'s own published predictions on the same systems — not just against experiment — so the central question the project answers is: **how much of FEP+'s accuracy can a fast ML surrogate recover, and where does it break down?**

## Data

Two fully public, no-registration datasets (git-clonable):

1. **Schrödinger public FEP+ benchmark** — `github.com/schrodinger/public_binding_free_energy_benchmark`
   3D protein-ligand structures, experimental binding data, and Schrödinger's own FEP+ predictions as baseline.
2. **Merck/Schindler FEP benchmark** — `github.com/MCompChem/fep-benchmark`
   8 targets, 264 ligands, 550 relative binding free energy edges (ligand-pair "maps").

Run `data/download_data.sh` to clone both into `data/raw/`.

## Architecture

1. **Featurization** — each ligand's bound 3D pose is converted into a graph (atoms = nodes, bonds = edges), enriched with pocket-residue context from the target.
2. **Encoder** — a 3D-aware GNN (SchNet-style message passing to start, upgradable to an E(3)-equivariant model) embeds each ligand conditioned on its pocket.
3. **Pairwise head** — a Siamese-style comparison of embeddings for ligand A and ligand B predicts ΔΔG = ΔG(B) − ΔG(A) directly along each benchmark "edge."
4. **Evaluation** — per-target Spearman/Kendall correlation and RMSE vs. experimental ΔΔG, reported alongside FEP+'s own accuracy on the same edges.

## Repo structure

```
deltabind-gnn/
├── data/download_data.sh      # clones both benchmark datasets
├── src/data/parse_benchmark.py    # unifies both datasets into one edge table
├── src/data/featurize.py          # 3D graph featurization
├── src/data/dataset.py            # PyTorch Geometric Dataset
├── src/models/model.py            # GNN encoder + pairwise head
├── src/train.py
├── src/evaluate.py
└── notebooks/train_colab.ipynb    # run everything end-to-end in Colab
```

## Quickstart (Colab)

Open `notebooks/train_colab.ipynb` in Google Colab — it clones this repo, downloads the benchmark data, installs dependencies (PyTorch Geometric, RDKit), featurizes, trains, and evaluates end-to-end. No local setup required.

## Status

Early-stage research project. Baseline: SchNet-style encoder + pairwise MLP head. Planned: equivariant encoder (EGNN/e3nn), pocket-conditioning ablations, uncertainty quantification.
