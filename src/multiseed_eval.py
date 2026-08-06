"""
multiseed_eval.py

Runs the full train + evaluate cycle across several ligand-holdout split
seeds and reports mean +/- std for each metric. A single seed's Spearman on
a ~300-edge test set is noisy enough to be misleading -- this is what
actually tells you whether "DeltaBind-GNN beats FEP+ on RMSE" is a real,
repeatable finding or an artifact of one lucky split.

Usage:
    python -m src.multiseed_eval --config configs/default.yaml --seeds 0 1 2 3 4
"""

import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LigandPairDataset, collate_pairs
from src.data.featurize import ELEMENTS, HYBRIDIZATIONS
from src.data.splits import ligand_holdout_split
from src.evaluate import compute_metrics, run_inference
from src.models.model import DeltaBindGNN
from src.train import evaluate_loss, train_one_epoch

NODE_FEATURE_DIM = len(ELEMENTS) + len(HYBRIDIZATIONS) + 4


def run_one_seed(seed, cfg, dataset, target_to_idx, num_targets, device):
    train_idx, val_idx, test_idx = ligand_holdout_split(
        dataset.edges, val_frac=cfg["train"]["val_split"],
        test_frac=cfg["train"]["test_split"], seed=seed,
    )

    collate_fn = partial(collate_pairs, target_to_idx=target_to_idx)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg["train"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    model = DeltaBindGNN(
        node_feature_dim=NODE_FEATURE_DIM,
        num_targets=num_targets,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_interactions=cfg["model"]["num_message_passing_layers"],
        head_hidden_dim=cfg["model"]["pairwise_head_hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(cfg["train"]["epochs"]):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    results_df = run_inference(model, test_loader, device)

    model_metrics = compute_metrics(results_df["pred_ddg"].values, results_df["exp_ddg"].values)
    fep_rows = results_df.dropna(subset=["fep_ddg"])
    fep_metrics = compute_metrics(fep_rows["fep_ddg"].values, fep_rows["exp_ddg"].values) if len(fep_rows) else None

    return model_metrics, fep_metrics, len(test_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--edges_csv", default="data/processed/edges_final.csv")
    parser.add_argument("--sdf_dir", default="data/processed/sdf")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LigandPairDataset(args.edges_csv, args.sdf_dir)

    unique_targets = sorted(dataset.edges["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    num_targets = len(unique_targets)

    model_runs, fep_runs = [], []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        model_metrics, fep_metrics, n_test = run_one_seed(seed, cfg, dataset, target_to_idx, num_targets, device)
        print(f"  n_test={n_test}  model: {model_metrics}")
        if fep_metrics:
            print(f"  fep+:  {fep_metrics}")
        model_runs.append(model_metrics)
        if fep_metrics:
            fep_runs.append(fep_metrics)

    print("\n=== Summary across seeds (mean +/- std) ===")
    for key in model_runs[0].keys():
        vals = [m[key] for m in model_runs]
        print(f"  DeltaBind-GNN {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if fep_runs:
        for key in fep_runs[0].keys():
            vals = [m[key] for m in fep_runs]
            print(f"  FEP+          {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()