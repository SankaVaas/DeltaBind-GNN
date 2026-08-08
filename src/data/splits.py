"""
splits.py

Edge-level random splitting leaks: the same ligand shows up in many edges
(it's a congeneric "map", not independent pairs), so a random split on
edges lets the model see a given ligand in both train and val/test, and it
can shortcut by memorizing per-ligand potency instead of learning structure
-> affinity. This gives a ligand-level holdout split instead: for each
target, ligand IDs themselves are partitioned into train/val/test groups.
An edge goes to test if EITHER of its ligands is a held-out test ligand
(requiring BOTH was tried first and dropped ~45% of edges as "straddling"
pairs, leaving too few test edges per target to trust); an edge goes to val
similarly, and everything else is train. This still guarantees no test/val
ligand's identity was seen during training.

Also provides a k-fold version: instead of one random split (whose test set
is small and swings a lot across seeds -- see multiseed_eval.py results),
partition each target's ligands into k folds and rotate which fold is held
out. Every edge ends up in exactly one test fold across the k runs, so
pooling out-of-fold predictions from all k models gives one evaluation over
nearly the full dataset instead of k independent ~50-edge subsamples.
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


def ligand_kfold_split(edges_df, k=5, val_frac_of_train=0.15, seed=42):
    """Partitions each target's ligands into k roughly-equal folds. For fold
    i: test_idx = edges where EITHER ligand belongs to fold i's ligand set;
    train_idx = edges where NEITHER ligand belongs to fold i's ligand set.
    This is the same leakage-safe rule as ligand_holdout_split, just applied
    per rotating fold instead of a single train/val/test split.

    IMPORTANT: an edge whose two ligands land in two different folds (e.g.
    ligand A in fold 2, ligand B in fold 4) will legitimately appear in BOTH
    fold 2's test set and fold 4's test set -- that's correct and safe (the
    model in each of those runs has genuinely never seen that specific
    ligand's fold-mate during training), but it means naively concatenating
    all folds' test predictions double-counts that edge. Don't dedupe here;
    dedupe at the pooling stage in kfold_eval.py using each edge's original
    dataframe index, keeping one prediction per edge. (An earlier version
    tried to avoid this by assigning each edge a single "canonical" fold up
    front -- that broke the leakage guarantee instead, since a ligand's
    fold-mates across different edges could straddle its own train/test
    boundary. Fix leakage in the split; fix double-counting in the pool.)

    Returns a list of k dicts, each with 'train_idx', 'val_idx', 'test_idx'
    (all lists of original edges_df index values).
    """
    rng = random.Random(seed)
    per_target_folds = {}  # target -> list of k ligand sets

    for target, group in edges_df.groupby("target"):
        ligands = sorted(set(group["ligand_A_id"]) | set(group["ligand_B_id"]))
        rng.shuffle(ligands)
        folds = [set() for _ in range(k)]
        for i, lig in enumerate(ligands):
            folds[i % k].add(lig)
        per_target_folds[target] = folds

    results = []
    for fold_i in range(k):
        train_idx, test_idx = [], []

        for target, group in edges_df.groupby("target"):
            test_ligands = per_target_folds[target][fold_i]

            for idx, row in group.iterrows():
                a, b = row["ligand_A_id"], row["ligand_B_id"]
                if a in test_ligands or b in test_ligands:
                    test_idx.append(idx)
                else:
                    train_idx.append(idx)

        # carve a val set out of this fold's train ligands (not touching test)
        train_rng = random.Random(seed * 1000 + fold_i)
        final_train_idx, val_idx = [], []
        train_edge_rows = edges_df.loc[train_idx]

        for target, group in train_edge_rows.groupby("target"):
            remaining_ligands = sorted(set(group["ligand_A_id"]) | set(group["ligand_B_id"]))
            train_rng.shuffle(remaining_ligands)
            n_val = max(1, int(len(remaining_ligands) * val_frac_of_train)) if remaining_ligands else 0
            val_ligands = set(remaining_ligands[:n_val])

            for idx, row in group.iterrows():
                a, b = row["ligand_A_id"], row["ligand_B_id"]
                if a in val_ligands or b in val_ligands:
                    val_idx.append(idx)
                else:
                    final_train_idx.append(idx)

        results.append({"train_idx": final_train_idx, "val_idx": val_idx, "test_idx": test_idx})

    return results