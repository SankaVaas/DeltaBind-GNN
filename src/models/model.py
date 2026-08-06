"""
model.py

DeltaBind-GNN model: a shared GNN encoder embeds each ligand (conditioned
optionally on a pocket context vector), and a pairwise head predicts
ddG = dG(B) - dG(A) from the two embeddings.

Encoder: a custom SchNet-style continuous-filter message passing network
that consumes our featurizer's actual node feature vectors directly (one-hot
element/hybridization + atomic properties), rather than relying on a
library encoder's built-in atomic-number embedding. Interatomic distances
are expanded with Gaussian radial basis functions and passed through a
filter-generating network, following Schutt et al. 2017 (SchNet) — this is
a from-scratch implementation of that idea sized for our feature space,
not a wrapper around a black-box encoder.

An E(3)-equivariant encoder (EGNN / e3nn) is the natural v2 upgrade once
this baseline is validated end-to-end.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def radius_graph_manual(pos: torch.Tensor, batch: torch.Tensor, cutoff: float,
                         loop: bool = False) -> torch.Tensor:
    """Dependency-free radius graph: connects atoms within `cutoff` distance,
    restricted to the same molecule (same `batch` index).

    PyG's built-in `radius_graph` requires the compiled `pyg-lib`/`torch-cluster`
    backend, which is fragile to install in Colab. Ligands here are small
    (tens of atoms), so a brute-force O(n^2) pairwise distance computation per
    forward pass is fast enough and removes that dependency entirely.
    """
    dist = torch.cdist(pos, pos)                       # [n, n]
    same_molecule = batch.unsqueeze(0) == batch.unsqueeze(1)  # [n, n]
    within_cutoff = dist <= cutoff
    mask = within_cutoff & same_molecule
    if not loop:
        mask.fill_diagonal_(False)
    row, col = mask.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0)


class GaussianSmearing(nn.Module):
    """Expands scalar interatomic distances into a Gaussian radial basis,
    the standard trick (from SchNet) for giving a neural net a smooth,
    differentiable handle on continuous distances rather than a raw scalar."""

    def __init__(self, start: float = 0.0, stop: float = 10.0, num_gaussians: int = 50):
        super().__init__()
        offsets = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offsets[1] - offsets[0]).item() ** 2
        self.register_buffer("offsets", offsets)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.view(-1, 1) - self.offsets.view(1, -1)
        return torch.exp(self.coeff * diff.pow(2))


class CFConv(nn.Module):
    """Continuous-filter convolution: messages between connected atoms are
    weighted by a filter network conditioned on their 3D distance, so the
    layer respects geometry rather than only bond connectivity."""

    def __init__(self, hidden_dim: int, num_gaussians: int = 50):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lin_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # row = target node, col = source node
        W = self.filter_net(edge_rbf)                 # [num_edges, hidden_dim]
        messages = self.lin_in(x[col]) * W             # [num_edges, hidden_dim]
        aggregated = scatter(messages, row, dim=0, dim_size=x.size(0), reduce="sum")
        return self.lin_out(aggregated)


class LigandEncoder(nn.Module):
    """Custom continuous-filter GNN encoder: raw node features -> pooled
    molecule embedding, conditioned on 3D geometry via a radius graph."""

    def __init__(self, node_feature_dim: int, hidden_dim: int = 128,
                 num_interactions: int = 4, num_gaussians: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.embedding_dim = hidden_dim
        self.cutoff = cutoff

        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)
        self.rbf = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            CFConv(hidden_dim, num_gaussians) for _ in range(num_interactions)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x:    [num_atoms, node_feature_dim]  raw features from featurize.py
        pos:  [num_atoms, 3]                 3D coordinates
        batch:[num_atoms]                    graph index per atom (from PyG Batch)
        """
        h = self.node_embed(x)

        # Build a distance-based graph (covers non-bonded neighbors too, unlike
        # the covalent-bond-only edge_index from featurize.py — geometry, not
        # just connectivity, is what should drive message passing here).
        edge_index = radius_graph_manual(pos, batch, cutoff=self.cutoff, loop=False)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_rbf = self.rbf(dist)

        for conv in self.interactions:
            h = h + conv(h, edge_index, edge_rbf)

        pooled = scatter(h, batch, dim=0, reduce="mean")
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
    """Full model: shared encoder (Siamese) + pairwise ddG head, conditioned
    on a learned per-target embedding (cheap proxy for pocket identity until
    real pocket-geometry features are wired in via extract_pocket_residues)."""

    def __init__(self, node_feature_dim: int, num_targets: int, hidden_dim: int = 128,
                 num_interactions: int = 4, target_embed_dim: int = 16,
                 head_hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = LigandEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_interactions=num_interactions,
        )
        self.target_embedding = nn.Embedding(num_targets, target_embed_dim)
        self.head = PairwiseDeltaHead(
            embedding_dim=self.encoder.embedding_dim,
            pocket_dim=target_embed_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch):
        emb_a = self.encoder(batch["ligand_a"].x, batch["ligand_a"].pos, batch["ligand_a"].batch)
        emb_b = self.encoder(batch["ligand_b"].x, batch["ligand_b"].pos, batch["ligand_b"].batch)
        pocket_vec = self.target_embedding(batch["target_idx"])
        return self.head(emb_a, emb_b, pocket_vec)