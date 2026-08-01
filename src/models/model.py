"""
model.py

DeltaBind-GNN model: a shared GNN encoder embeds each ligand (conditioned
optionally on a pocket context vector), and a pairwise head predicts
ddG = dG(B) - dG(A) from the two embeddings.

Baseline encoder: a simple SchNet-style continuous-filter message passing
network operating on 3D coordinates + atom features. This is intentionally
the simplest thing that respects 3D geometry; an E(3)-equivariant encoder
(EGNN / e3nn) is the natural upgrade once the pipeline is validated
end-to-end.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import SchNet


class LigandEncoder(nn.Module):
    """Wraps PyG's SchNet as a per-atom -> pooled molecule embedding encoder.

    Using PyG's built-in SchNet implementation for the baseline rather than
    hand-rolling message passing, so the project's novelty sits in the
    pairwise/pocket-conditioning framing and evaluation methodology, not in
    reimplementing a standard encoder from scratch.
    """

    def __init__(self, hidden_dim: int = 128, num_interactions: int = 4,
                 num_gaussians: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="mean",
        )
        # Drop SchNet's final scalar-output layer; we want the pooled embedding.
        self.embedding_dim = hidden_dim

    def forward(self, z_or_x, pos, batch):
        """
        z_or_x: atomic numbers if using SchNet's built-in embedding, OR
                precomputed node features (see note below).
        pos: [num_atoms, 3]
        batch: [num_atoms] graph index per atom
        """
        # NOTE: PyG's SchNet expects integer atomic numbers (z) for its
        # internal embedding layer. Our featurize.py currently produces
        # one-hot + property node features (richer than plain element type).
        # For the first working baseline, pass atomic-number-equivalent ints
        # derived from the one-hot element channel; swap in a custom
        # message-passing encoder later to use the full feature vector.
        h = self.schnet.embedding(z_or_x)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, pos, batch, edge_index=None)
        pooled = self.schnet.readout(h, batch)
        return pooled


class PairwiseDeltaHead(nn.Module):
    """Predicts ddG from a pair of ligand embeddings (+ optional pocket vector)."""

    def __init__(self, embedding_dim: int, pocket_dim: int = 0, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        input_dim = embedding_dim + pocket_dim  # applied to the embedding *difference*
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, pocket: torch.Tensor = None):
        diff = emb_b - emb_a
        if pocket is not None:
            diff = torch.cat([diff, pocket], dim=-1)
        return self.mlp(diff).squeeze(-1)


class DeltaBindGNN(nn.Module):
    """Full model: shared encoder (Siamese) + pairwise ddG head."""

    def __init__(self, hidden_dim: int = 128, num_interactions: int = 4,
                 pocket_dim: int = 0, head_hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = LigandEncoder(hidden_dim=hidden_dim, num_interactions=num_interactions)
        self.head = PairwiseDeltaHead(
            embedding_dim=self.encoder.embedding_dim,
            pocket_dim=pocket_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch):
        emb_a = self.encoder(batch["ligand_a"].x, batch["ligand_a"].pos, batch["ligand_a"].batch)
        emb_b = self.encoder(batch["ligand_b"].x, batch["ligand_b"].pos, batch["ligand_b"].batch)
        return self.head(emb_a, emb_b, batch.get("pocket"))
