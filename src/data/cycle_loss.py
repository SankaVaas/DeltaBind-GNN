"""
cycle_loss.py

FEP maps often contain closed triangles: ligand A -> B, B -> C, and A -> C
all measured/predicted. Thermodynamics requires these to be self-consistent:
ddG(A->B) + ddG(B->C) + ddG(C->A) = 0 (going around a closed loop costs no
net free energy). FEP+ itself uses this as an internal quality metric.

This gives the model a physics-grounded regularizer that doesn't need any
extra labels — just structural consistency — and tends to help ranking
metrics specifically, since inconsistent triangles are usually what wreck
Kendall tau.

Usage: call find_target_cycles() once after loading edges_final.csv, then
call cycle_consistency_loss() periodically during training (e.g. once per
epoch, not every batch -- forwarding 3 extra molecules per cycle adds up).
"""

import itertools
import random

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from src.data.featurize import featurize_ligand


def find_target_cycles(edges_df, max_cycles_per_target=50, seed=42):
    """Find closed triangles (3-cycles) in each target's ligand graph.

    Returns a list of (target, ligand_A_id, ligand_B_id, ligand_C_id,
    sdf_A, sdf_B, sdf_C) tuples.
    """
    rng = random.Random(seed)
    cycles = []

    for target, group in edges_df.groupby("target"):
        g = nx.Graph()
        sdf_lookup = {}
        for _, row in group.iterrows():
            a, b = row["ligand_A_id"], row["ligand_B_id"]
            g.add_edge(a, b)
            sdf_lookup[a] = row["ligand_A_sdf"]
            sdf_lookup[b] = row["ligand_B_sdf"]

        triangles = [
            (a, b, c) for a, b, c in itertools.combinations(g.nodes, 3)
            if g.has_edge(a, b) and g.has_edge(b, c) and g.has_edge(a, c)
        ]
        rng.shuffle(triangles)

        for a, b, c in triangles[:max_cycles_per_target]:
            cycles.append((target, a, b, c, sdf_lookup[a], sdf_lookup[b], sdf_lookup[c]))

    return cycles


def _to_pyg_data(sdf_path):
    g = featurize_ligand(sdf_path)
    return Data(
        x=torch.tensor(g.node_features),
        pos=torch.tensor(g.positions),
        edge_index=torch.tensor(g.edge_index),
    )


def cycle_consistency_loss(model, cycles, device, max_cycles_per_call=20, seed=None):
    """Samples up to `max_cycles_per_call` cycles and penalizes deviation
    from ddG(A->B) + ddG(B->C) + ddG(C->A) != 0.

    Keep this a small sample per call (not the full cycle list) -- it's
    called every epoch alongside normal training, and each cycle costs 3x
    the forward passes of a normal edge.
    """
    if not cycles:
        return torch.tensor(0.0, device=device)

    rng = random.Random(seed)
    sample = rng.sample(cycles, min(max_cycles_per_call, len(cycles)))

    total_loss = 0.0
    for target, a_id, b_id, c_id, sdf_a, sdf_b, sdf_c in sample:
        data_a = Batch.from_data_list([_to_pyg_data(sdf_a)]).to(device)
        data_b = Batch.from_data_list([_to_pyg_data(sdf_b)]).to(device)
        data_c = Batch.from_data_list([_to_pyg_data(sdf_c)]).to(device)

        emb_a = model.encoder(data_a.x, data_a.pos, data_a.batch)
        emb_b = model.encoder(data_b.x, data_b.pos, data_b.batch)
        emb_c = model.encoder(data_c.x, data_c.pos, data_c.batch)

        ddg_ab = model.head(emb_a, emb_b)
        ddg_bc = model.head(emb_b, emb_c)
        ddg_ca = model.head(emb_c, emb_a)

        cycle_sum = ddg_ab + ddg_bc + ddg_ca
        total_loss = total_loss + cycle_sum.pow(2).mean()

    return total_loss / len(sample)