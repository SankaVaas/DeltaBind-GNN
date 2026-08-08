"""
dataset.py

PyTorch Geometric Dataset over "edges": pairs of congeneric ligands (A, B)
bound to the same target, labeled with experimental ddG = dG(B) - dG(A).

Each __getitem__ returns a pair of graphs (ligand A, ligand B) plus a shared
pocket context vector for the target, and the scalar ddG label. The model
consumes both graphs through a shared encoder (Siamese) and predicts the
difference.
"""

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from src.data.featurize import featurize_ligand


class LigandPairDataset(Dataset):
    def __init__(self, edges_csv_path: str, sdf_dir: str, pocket_features: dict = None,
                 transform=None, pre_transform=None):
        """
        Args:
            edges_csv_path: path to the finalized edge table (target, ligand_A_id,
                ligand_B_id, sdf paths, exp_ddg, fep_pred_ddg, source).
            sdf_dir: root directory containing per-ligand SDF files referenced
                by the edge table.
            pocket_features: optional dict[target_name] -> fixed-size pocket
                embedding (precomputed, e.g. pooled CA coordinates or a
                learned embedding). If None, pocket conditioning is skipped
                and the model relies on ligand structure alone.
        """
        super().__init__(None, transform, pre_transform)
        self.edges = pd.read_csv(edges_csv_path)
        self.sdf_dir = sdf_dir
        self.pocket_features = pocket_features or {}

    def len(self):
        return len(self.edges)

    def get(self, idx):
        row = self.edges.iloc[idx]

        graph_a = featurize_ligand(row["ligand_A_sdf"], mol_id=row["ligand_A_id"])
        graph_b = featurize_ligand(row["ligand_B_sdf"], mol_id=row["ligand_B_id"])

        data_a = Data(
            x=torch.tensor(graph_a.node_features),
            pos=torch.tensor(graph_a.positions),
            edge_index=torch.tensor(graph_a.edge_index),
        )
        data_b = Data(
            x=torch.tensor(graph_b.node_features),
            pos=torch.tensor(graph_b.positions),
            edge_index=torch.tensor(graph_b.edge_index),
        )

        pocket_vec = self.pocket_features.get(row["target"], None)
        pocket_tensor = torch.tensor(pocket_vec, dtype=torch.float32) if pocket_vec is not None else None

        label = torch.tensor([row["exp_ddg"]], dtype=torch.float32)
        fep_baseline = torch.tensor(
            [row["fep_pred_ddg"]] if "fep_pred_ddg" in row and not pd.isna(row["fep_pred_ddg"]) else [float("nan")],
            dtype=torch.float32,
        )

        return {
            "ligand_a": data_a,
            "ligand_b": data_b,
            "pocket": pocket_tensor,
            "target": row["target"],
            "label": label,
            "fep_baseline": fep_baseline,
            "edge_idx": idx,
        }


def collate_pairs(batch, target_to_idx=None):
    """Custom collate: batches ligand_a and ligand_b graphs separately via PyG's Batch,
    keeps labels/baselines/targets as simple tensors/lists.

    target_to_idx: dict mapping target name -> integer index, used to build
    the target_idx tensor consumed by DeltaBindGNN's per-target embedding.
    Built once in train.py from the full dataset's unique targets.

    edge_idx: each item's original row index in edges_final.csv, carried
    through so kfold_eval.py can dedupe an edge that legitimately appears
    in more than one fold's test set (see splits.py docstring) without
    double-counting it in the pooled metrics.
    """
    from torch_geometric.data import Batch

    ligand_a_batch = Batch.from_data_list([b["ligand_a"] for b in batch])
    ligand_b_batch = Batch.from_data_list([b["ligand_b"] for b in batch])
    labels = torch.cat([b["label"] for b in batch])
    fep_baselines = torch.cat([b["fep_baseline"] for b in batch])
    targets = [b["target"] for b in batch]
    edge_indices = [b["edge_idx"] for b in batch]

    target_idx = None
    if target_to_idx is not None:
        target_idx = torch.tensor([target_to_idx[t] for t in targets], dtype=torch.long)

    pockets = None
    if batch[0]["pocket"] is not None:
        pockets = torch.stack([b["pocket"] for b in batch])

    return {
        "ligand_a": ligand_a_batch,
        "ligand_b": ligand_b_batch,
        "target_idx": target_idx,
        "pocket": pockets,
        "targets": targets,
        "labels": labels,
        "fep_baselines": fep_baselines,
        "edge_indices": edge_indices,
    }