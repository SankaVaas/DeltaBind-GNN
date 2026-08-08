"""
evaluate.py

Evaluates a trained DeltaBind-GNN checkpoint against experimental ddG, and
-- where available -- against Schrodinger's own published FEP+ predictions
on the same ligand pairs.

IMPORTANT: point --edges_csv at data/processed/edges_test.csv (written by
train.py), not edges_final.csv. Evaluating on the full edge set mixes in
edges the model trained on, which inflates every metric.

Reports, overall and per-target:
    RMSE, MAE, R^2, Spearman rho, Kendall tau

Usage:
    python -m src.evaluate --checkpoint results/best_model.pt \
        --edges_csv data/processed/edges_test.csv
"""

import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.data.dataset import LigandPairDataset, collate_pairs
from src.data.featurize import ELEMENTS, HYBRIDIZATIONS
from src.models.model import DeltaBindGNN

NODE_FEATURE_DIM = len(ELEMENTS) + len(HYBRIDIZATIONS) + 4


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "mae": float(mean_absolute_error(labels, preds)),
        "r2": float(r2_score(labels, preds)) if len(labels) > 1 else float("nan"),
        "spearman": float(spearmanr(labels, preds).correlation) if len(labels) > 1 else float("nan"),
        "kendall_tau": float(kendalltau(labels, preds).correlation) if len(labels) > 1 else float("nan"),
    }


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_fep, all_targets, all_edge_idx = [], [], [], [], []

    for batch in loader:
        batch["ligand_a"] = batch["ligand_a"].to(device)
        batch["ligand_b"] = batch["ligand_b"].to(device)
        batch["target_idx"] = batch["target_idx"].to(device)

        preds = model(batch).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].numpy().tolist())
        all_fep.extend(batch["fep_baselines"].numpy().tolist())
        all_targets.extend(batch["targets"])
        all_edge_idx.extend(batch["edge_indices"])

    return pd.DataFrame({
        "edge_idx": all_edge_idx,
        "target": all_targets,
        "pred_ddg": all_preds,
        "exp_ddg": all_labels,
        "fep_ddg": all_fep,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--edges_csv", default="data/processed/edges_test.csv")
    parser.add_argument("--sdf_dir", default="data/processed/sdf")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_interactions", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LigandPairDataset(args.edges_csv, args.sdf_dir)

    # NOTE: target vocabulary must be built from the SAME full edge set used
    # at train time (edges_final.csv), not just this eval subset -- otherwise
    # target indices won't line up with the trained embedding weights if this
    # eval file happens to be missing a target that appeared during training.
    full_edges = pd.read_csv("data/processed/edges_final.csv")
    unique_targets = sorted(full_edges["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    num_targets = len(unique_targets)
    collate_fn = partial(collate_pairs, target_to_idx=target_to_idx)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = DeltaBindGNN(node_feature_dim=NODE_FEATURE_DIM, num_targets=num_targets,
                         hidden_dim=args.hidden_dim, num_interactions=args.num_interactions).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results_df = run_inference(model, loader, device)
    results_df.to_csv("results/predictions.csv", index=False)

    print("=== Overall: DeltaBind-GNN vs Experiment ===")
    overall = compute_metrics(results_df["pred_ddg"].values, results_df["exp_ddg"].values)
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    fep_rows = results_df.dropna(subset=["fep_ddg"])
    if len(fep_rows) > 0:
        print("\n=== Overall: FEP+ vs Experiment (baseline) ===")
        fep_metrics = compute_metrics(fep_rows["fep_ddg"].values, fep_rows["exp_ddg"].values)
        for k, v in fep_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Per-target: DeltaBind-GNN vs Experiment ===")
    for target, group in results_df.groupby("target"):
        m = compute_metrics(group["pred_ddg"].values, group["exp_ddg"].values)
        print(f"  {target:15s} n={len(group):4d}  rmse={m['rmse']:.3f}  spearman={m['spearman']:.3f}")

    print("\nFull predictions saved to results/predictions.csv")


if __name__ == "__main__":
    main()