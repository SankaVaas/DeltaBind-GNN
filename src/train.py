"""
train.py

Trains DeltaBind-GNN on the ligand-pair dataset with an 80/15/15-style
train/val/test split (configurable), using MSE loss on ddG (kcal/mol).

Usage:
    python -m src.train --config configs/default.yaml
"""

import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import LigandPairDataset, collate_pairs
from src.models.model import DeltaBindGNN


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch["ligand_a"] = batch["ligand_a"].to(device)
        batch["ligand_b"] = batch["ligand_b"].to(device)
        labels = batch["labels"].to(device)
        if batch["pocket"] is not None:
            batch["pocket"] = batch["pocket"].to(device)

        optimizer.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch["ligand_a"] = batch["ligand_a"].to(device)
        batch["ligand_b"] = batch["ligand_b"].to(device)
        labels = batch["labels"].to(device)
        if batch["pocket"] is not None:
            batch["pocket"] = batch["pocket"].to(device)

        preds = model(batch)
        loss = loss_fn(preds, labels)
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


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

    val_frac = cfg["train"]["val_split"]
    test_frac = cfg["train"]["test_split"]
    n = len(dataset)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(cfg["train"]["seed"])
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               collate_fn=collate_pairs)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             collate_fn=collate_pairs)

    model = DeltaBindGNN(
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
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

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
