"""
kfold_eval.py

Trains k models, one per ligand-fold (see ligand_kfold_split in
src/data/splits.py), and pools their out-of-fold test predictions into one
evaluation. This directly targets the variance problem seen in
multiseed_eval.py: single-split test sets were ~32-60 edges out of ~3,400
because most of the 81 targets are small, so Spearman/Kendall swung wildly
across seeds. Pooling out-of-fold predictions from k=5 folds evaluates on
close to the FULL dataset (every edge appears in test exactly once) while
still never letting a model see a ligand's identity in both train and its
own test fold.

Usage:
    python -m src.kfold_eval --config configs/default.yaml --k 5
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from functools import partial
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LigandPairDataset, collate_pairs
from src.data.featurize import ELEMENTS, HYBRIDIZATIONS
from src.data.splits import ligand_kfold_split
from src.evaluate import compute_metrics, run_inference
from src.models.model import DeltaBindGNN
from src.train import evaluate_loss, train_one_epoch

NODE_FEATURE_DIM = len(ELEMENTS) + len(HYBRIDIZATIONS) + 4


def run_one_fold(fold, cfg, dataset, target_to_idx, num_targets, device, fold_num):
    collate_fn = partial(collate_pairs, target_to_idx=target_to_idx)
    train_loader = DataLoader(Subset(dataset, fold["train_idx"]), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, fold["val_idx"]), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, fold["test_idx"]), batch_size=cfg["train"]["batch_size"],
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
    torch.save(model.state_dict(), f"results/fold_{fold_num}_model.pt")

    return run_inference(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--edges_csv", default="data/processed/edges_final.csv")
    parser.add_argument("--sdf_dir", default="data/processed/sdf")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LigandPairDataset(args.edges_csv, args.sdf_dir)

    unique_targets = sorted(dataset.edges["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    num_targets = len(unique_targets)

    folds = ligand_kfold_split(dataset.edges, k=args.k, seed=args.seed)
    for i, fold in enumerate(folds):
        print(f"Fold {i}: {len(fold['train_idx'])} train / {len(fold['val_idx'])} val / "
              f"{len(fold['test_idx'])} test edges")

    all_oof_predictions = []
    for i, fold in enumerate(folds):
        print(f"\n=== Training fold {i+1}/{args.k} ===")
        fold_preds = run_one_fold(fold, cfg, dataset, target_to_idx, num_targets, device, i)
        all_oof_predictions.append(fold_preds)

    oof_df = pd.concat(all_oof_predictions, ignore_index=True)
    n_before_dedup = len(oof_df)
    oof_df = oof_df.drop_duplicates(subset="edge_idx", keep="first")
    n_dropped = n_before_dedup - len(oof_df)

    oof_df.to_csv("results/kfold_oof_predictions.csv", index=False)
    print(f"\nTotal test predictions across all folds: {n_before_dedup}")
    print(f"After deduping edges that legitimately appeared in >1 fold's test set: {len(oof_df)} "
          f"({n_dropped} duplicate predictions discarded, kept first occurrence per edge)")
    print(f"(out of {len(dataset)} total edges)")

    print("\n=== Pooled out-of-fold: DeltaBind-GNN vs Experiment ===")
    model_metrics = compute_metrics(oof_df["pred_ddg"].values, oof_df["exp_ddg"].values)
    for k, v in model_metrics.items():
        print(f"  {k}: {v:.4f}")

    fep_rows = oof_df.dropna(subset=["fep_ddg"])
    if len(fep_rows) > 0:
        print(f"\n=== Pooled out-of-fold: FEP+ vs Experiment (baseline, n={len(fep_rows)}) ===")
        fep_metrics = compute_metrics(fep_rows["fep_ddg"].values, fep_rows["exp_ddg"].values)
        for k, v in fep_metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()