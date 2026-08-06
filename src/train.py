"""
train.py

Trains DeltaBind-GNN on the ligand-pair dataset using a ligand-level holdout
split (see src/data/splits.py -- prevents the same ligand appearing in both
train and eval, which silently inflates results via memorization).

Combines three training signals:
  1. MSE loss on ddG (the base regression signal).
  2. A pairwise ranking loss within each target, so the model is explicitly
     pushed to get relative ordering right, not just magnitude.
  3. A thermodynamic cycle-consistency loss (closed triangles in a target's
     ligand map should sum to ~0 ddG around the loop) -- a physics-grounded
     regularizer that needs no extra labels.

Usage:
    python -m src.train --config configs/default.yaml
"""

import argparse
from functools import partial

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.cycle_loss import cycle_consistency_loss, find_target_cycles
from src.data.dataset import LigandPairDataset, collate_pairs
from src.data.featurize import ELEMENTS, HYBRIDIZATIONS
from src.data.splits import ligand_holdout_split
from src.models.model import DeltaBindGNN

# Must match the feature vector built in featurize.featurize_ligand():
# one-hot(element) + one-hot(hybridization) + [charge, degree, aromatic, numHs]
NODE_FEATURE_DIM = len(ELEMENTS) + len(HYBRIDIZATIONS) + 4

RANKING_LOSS_WEIGHT = 0.5
CYCLE_LOSS_WEIGHT = 0.3
CYCLES_PER_EPOCH = 20


def pairwise_ranking_loss(preds, labels, targets, margin=0.1):
    """For edges within the same target, penalize disagreements in relative
    order between predicted and true ddG. Pure MSE has no notion of ranking,
    which is why Spearman/Kendall can lag behind RMSE without this term."""
    loss = 0.0
    count = 0
    targets = list(targets)
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if targets[i] != targets[j]:
                continue
            true_diff = labels[i] - labels[j]
            pred_diff = preds[i] - preds[j]
            if true_diff.abs() < 1e-3:
                continue
            sign = true_diff.sign()
            loss = loss + torch.clamp(margin - sign * pred_diff, min=0)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=preds.device)
    return loss / count


def combined_loss(preds, labels, targets, mse_fn, ranking_weight=RANKING_LOSS_WEIGHT):
    mse = mse_fn(preds, labels)
    rank_loss = pairwise_ranking_loss(preds, labels, targets)
    return mse + ranking_weight * rank_loss, mse, rank_loss


def train_one_epoch(model, loader, optimizer, loss_fn, device, ranking_weight=RANKING_LOSS_WEIGHT):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch["ligand_a"] = batch["ligand_a"].to(device)
        batch["ligand_b"] = batch["ligand_b"].to(device)
        batch["target_idx"] = batch["target_idx"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model(batch)
        loss, mse, rank_loss = combined_loss(preds, labels, batch["targets"], loss_fn, ranking_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device, ranking_weight=RANKING_LOSS_WEIGHT):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch["ligand_a"] = batch["ligand_a"].to(device)
        batch["ligand_b"] = batch["ligand_b"].to(device)
        batch["target_idx"] = batch["target_idx"].to(device)
        labels = batch["labels"].to(device)

        preds = model(batch)
        loss, _, _ = combined_loss(preds, labels, batch["targets"], loss_fn, ranking_weight)
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def run_cycle_loss_step(model, target_cycles, optimizer, device, epoch):
    """One extra gradient step per epoch using the cycle-consistency loss.
    Kept separate from the main batch loop since each cycle costs 3x the
    forward passes of a normal edge -- not worth doing every batch."""
    if not target_cycles:
        return 0.0

    cyc_loss = cycle_consistency_loss(model, target_cycles, device,
                                       max_cycles_per_call=CYCLES_PER_EPOCH, seed=epoch)
    if not cyc_loss.requires_grad:
        return cyc_loss.item()

    optimizer.zero_grad()
    (CYCLE_LOSS_WEIGHT * cyc_loss).backward()
    optimizer.step()
    return cyc_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--edges_csv", default="data/processed/edges_final.csv")
    parser.add_argument("--sdf_dir", default="data/processed/sdf")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = LigandPairDataset(args.edges_csv, args.sdf_dir)

    unique_targets = sorted(dataset.edges["target"].unique())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    num_targets = len(unique_targets)
    print(f"{num_targets} unique targets: {unique_targets}")
    collate_fn = partial(collate_pairs, target_to_idx=target_to_idx)

    train_idx, val_idx, test_idx = ligand_holdout_split(
        dataset.edges, val_frac=cfg["train"]["val_split"],
        test_frac=cfg["train"]["test_split"], seed=cfg["train"]["seed"],
    )
    dropped = len(dataset) - len(train_idx) - len(val_idx) - len(test_idx)
    print(f"Ligand-holdout split: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(test_idx)} test edges (dropped {dropped} straddling edges)")

    test_edges_df = dataset.edges.loc[test_idx]
    test_edges_df.to_csv("data/processed/edges_test.csv", index=False)
    print(f"Held-out test edges saved to data/processed/edges_test.csv ({len(test_edges_df)} edges)")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)

    target_cycles = find_target_cycles(dataset.edges)
    print(f"Found {len(target_cycles)} closed triangles across all targets for cycle-consistency training")

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

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        cyc_loss = run_cycle_loss_step(model, target_cycles, optimizer, device, epoch)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1:03d} | train_loss {train_loss:.4f} | "
              f"cycle_loss {cyc_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best val loss: {best_val_loss:.4f}. Model saved to results/best_model.pt")


if __name__ == "__main__":
    main()