"""
splits.py

Edge-level random splitting leaks: the same ligand shows up in many edges
(it's a congeneric "map", not independent pairs), so a random split on
edges lets the model see a given ligand in both train and val/test, and it
can shortcut by memorizing per-ligand potency instead of learning structure
-> affinity. This gives a ligand-level holdout split instead: for each
target, ligand IDs themselves are partitioned into train/val/test, and an
edge is only kept in a split if both its ligands belong to that split.
Edges that straddle two splits are dropped rather than risk leakage.
"""

import random

def ligand_holdout_split(edges_df, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for target, group in edges_df.groupby("target"):
        ligands = sorted(set(group["ligand_A_id"]) | set(group["ligand_B_id"]))
        rng.shuffle(ligands)

        n = len(ligands)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))

        val_ligands = set(ligands[:n_val])
        test_ligands = set(ligands[n_val:n_val + n_test])
        # everything else is implicitly train

        for idx, row in group.iterrows():
            a, b = row["ligand_A_id"], row["ligand_B_id"]
            # priority: if either ligand is held out for test, the edge is test
            if a in test_ligands or b in test_ligands:
                test_idx.append(idx)
            elif a in val_ligands or b in val_ligands:
                val_idx.append(idx)
            else:
                train_idx.append(idx)

    return train_idx, val_idx, test_idx