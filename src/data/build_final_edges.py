"""
build_final_edges.py

Converts the raw Merck/Schindler edge CSV (columns: Ligand1, Ligand2, Exp.,
FEP, target, ...) into the schema LigandPairDataset expects, AND splits each
target's multi-molecule `ligands.sdf` into individual per-ligand SDF files
so ligand_A_sdf / ligand_B_sdf point at real, loadable files.

Expected raw layout per target (Merck/Schindler benchmark):
    data/raw/merck_fep_benchmark/<target>/ligands.sdf   (all ligands, one target)
    data/raw/merck_fep_benchmark/<target>/protein.pdb

Each molecule inside ligands.sdf is expected to carry an identifying name/
title (via RDKit's mol.GetProp('_Name')) matching the Ligand1/Ligand2 IDs
in the CSV. Run the inspection step first (this script prints what it finds)
before trusting the output — SDF title fields are not perfectly standardized
across benchmark releases.

Usage:
    python -m src.data.build_final_edges \
        --edges_raw data/processed/edges_raw.csv \
        --raw_dir data/raw/merck_fep_benchmark \
        --out_sdf_dir data/processed/sdf \
        --out_csv data/processed/edges_final.csv
"""

import argparse
import os

import pandas as pd
from rdkit import Chem


def split_target_sdf(target: str, raw_dir: str, out_sdf_dir: str) -> dict:
    """Split one target's ligands.sdf into per-ligand files.

    Returns a dict mapping ligand_id (str, as found in the SDF title) -> filepath.
    """
    sdf_path = os.path.join(raw_dir, target, "ligands.sdf")
    if not os.path.exists(sdf_path):
        print(f"[warn] no ligands.sdf found for target '{target}' at {sdf_path}")
        return {}

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    out_dir = os.path.join(out_sdf_dir, target)
    os.makedirs(out_dir, exist_ok=True)

    id_to_path = {}
    for mol in suppl:
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else None
        if not name:
            continue

        out_path = os.path.join(out_dir, f"{name}.sdf")
        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()
        id_to_path[name] = out_path

    return id_to_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges_raw", default="data/processed/edges_raw.csv")
    parser.add_argument("--raw_dir", default="data/raw/merck_fep_benchmark")
    parser.add_argument("--out_sdf_dir", default="data/processed/sdf")
    parser.add_argument("--out_csv", default="data/processed/edges_final.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.edges_raw)

    # Column mapping from the Merck CSV format to our schema
    df = df.rename(columns={
        "Ligand1": "ligand_A_id",
        "Ligand2": "ligand_B_id",
        "Exp.": "exp_ddg",
        "FEP": "fep_pred_ddg",
    })
    df["ligand_A_id"] = df["ligand_A_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["ligand_B_id"] = df["ligand_B_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    final_rows = []
    unresolved = []

    for target, group in df.groupby("target"):
        id_to_path = split_target_sdf(target, args.raw_dir, args.out_sdf_dir)
        print(f"[{target}] split {len(id_to_path)} ligands from ligands.sdf. "
              f"Example IDs found: {list(id_to_path.keys())[:5]}")

        protein_pdb = os.path.join(args.raw_dir, target, "protein.pdb")

        for _, row in group.iterrows():
            a_id, b_id = row["ligand_A_id"], row["ligand_B_id"]
            a_path = id_to_path.get(a_id)
            b_path = id_to_path.get(b_id)

            if a_path is None or b_path is None:
                unresolved.append((target, a_id, b_id))
                continue

            final_rows.append({
                "target": target,
                "ligand_A_id": a_id,
                "ligand_B_id": b_id,
                "ligand_A_sdf": a_path,
                "ligand_B_sdf": b_path,
                "protein_pdb": protein_pdb,
                "exp_ddg": row["exp_ddg"],
                "fep_pred_ddg": row.get("fep_pred_ddg"),
                "source": "merck",
            })

    final_df = pd.DataFrame(final_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    final_df.to_csv(args.out_csv, index=False)

    print(f"\nWrote {len(final_df)} resolved edges to {args.out_csv}")
    if unresolved:
        print(f"[warn] {len(unresolved)} edges could not be resolved to SDF files "
              f"(ID mismatch between CSV and SDF titles). First few:")
        for t, a, b in unresolved[:10]:
            print(f"    target={t}  ligand_A_id={a}  ligand_B_id={b}")
        print("If this list is large, the SDF title format doesn't match the CSV "
              "ligand IDs directly — inspect one ligands.sdf manually "
              "(e.g. print mol.GetProp('_Name') for the first few mols) and adjust "
              "the ID parsing above.")


if __name__ == "__main__":
    main()
