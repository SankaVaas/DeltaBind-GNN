"""
featurize.py

Converts a ligand's 3D bound pose (SDF/mol2/PDB) into a graph representation
suitable for GNN input:

  - Node features per atom: element (one-hot), formal charge, hybridization,
    aromaticity, degree, 3D coordinates (kept separate for equivariant models).
  - Edge index: bonds from RDKit's bond table (covalent graph). A radius-graph
    fallback can be added later for non-covalent pocket-contact edges.

Also extracts pocket residue context: protein residues within
`pocket_cutoff_angstrom` of any ligand atom, so the encoder can condition
each ligand embedding on the local binding environment (important since we
need pocket features to cancel between paired ligands in the same target).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]


def _one_hot(value, choices: List[str]) -> List[float]:
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


@dataclass
class LigandGraph:
    node_features: np.ndarray   # [num_atoms, node_feature_dim]
    positions: np.ndarray       # [num_atoms, 3]
    edge_index: np.ndarray      # [2, num_edges]
    mol_id: str


def featurize_ligand(sdf_path: str, mol_id: Optional[str] = None) -> LigandGraph:
    """Load a single ligand (first conformer) from an SDF file and build its graph."""
    if Chem is None:
        raise ImportError("RDKit is required. Install via `pip install rdkit`.")

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not parse a valid molecule from {sdf_path}")

    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)  # fallback; ideally the SDF already has 3D coords

    conf = mol.GetConformer()
    node_feats = []
    positions = []

    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        hyb = str(atom.GetHybridization())
        feats = (
            _one_hot(elem, ELEMENTS)
            + _one_hot(hyb, HYBRIDIZATIONS)
            + [
                float(atom.GetFormalCharge()),
                float(atom.GetDegree()),
                float(atom.GetIsAromatic()),
                float(atom.GetTotalNumHs()),
            ]
        )
        node_feats.append(feats)
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # undirected -> store both directions

    return LigandGraph(
        node_features=np.array(node_feats, dtype=np.float32),
        positions=np.array(positions, dtype=np.float32),
        edge_index=np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64),
        mol_id=mol_id or sdf_path,
    )


def extract_pocket_residues(protein_pdb_path: str, ligand_positions: np.ndarray,
                             cutoff_angstrom: float = 5.0):
    """Return CA coordinates of residues within `cutoff_angstrom` of any ligand atom.

    Uses Biopython for PDB parsing. Returned as an [num_pocket_residues, 3] array
    of CA positions, which can be pooled into a fixed-size pocket embedding or
    used as additional graph nodes connected to the ligand graph.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_pdb_path)

    pocket_ca_coords = []
    for residue in structure.get_residues():
        if "CA" not in residue:
            continue
        ca = residue["CA"].coord
        dists = np.linalg.norm(ligand_positions - ca, axis=1)
        if dists.min() <= cutoff_angstrom:
            pocket_ca_coords.append(ca)

    return np.array(pocket_ca_coords, dtype=np.float32) if pocket_ca_coords else np.zeros((0, 3), dtype=np.float32)
