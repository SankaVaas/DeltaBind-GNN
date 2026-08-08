"""
build_schrodinger_edges.py

Parses fep_benchmark_inputs/structure_inputs/<group>/<subset>_edges.csv
(+ matching _ligands.sdf, _protein.pdb) into the same edge schema used by
build_final_edges.py for Merck, so both sources can be concatenated into one
training set.

Confirmed columns (via inspect_schrodinger_edges.py): 'Ligand 1', 'Ligand 2',
'ddG (kcal/mol)'. Ligand names in edges.csv match SDF molecule titles
exactly here -- no ID-mismatch step needed, unlike Merck's SHP2 slash issue.

Each <group>/<subset> pair is treated as its own "target" (e.g.
"macrocycles__hsp90_3hvd_custcore"), since each subset is its own distinct
protein structure/pocket -- collapsing subsets within a group together would
incorrectly merge unrelated pockets under one target.

Note: this dataset doesn't carry a per-edge FEP+ prediction in the same file
(those live separately under 21_4_results/edge_predictions/ with different
naming) -- fep_pred_ddg is left NaN for these rows. That's fine: they're
still valid training edges, they just won't contribute to the FEP+ baseline
comparison in evaluate.py (which already drops NaN fep_ddg rows).

Usage:
    python -m src.data.build_schrodinger_edges \
        --raw_dir data/raw/schrodinger_fep_benchmark \
        --out_sdf_dir data/processed/sdf_schrodinger \
        --out_csv data/processed/edges_schrodinger.csv
"""

import argparse
import glob
import os
import re

import pandas as pd
from rdkit import Chem


def sanitize_filename(name: str) -> str:
    """Ligand titles here also contain spaces/commas (e.g. '8, 3VHA'),
    invalid or awkward in filenames -- same fix as Merck's SHP099 slash case."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name).strip())


def split_ligands_sdf(sdf_path: str, out_dir: str) -> dict:
    """Split one subset's ligands.sdf into per-ligand files.
    Returns dict mapping the ORIGINAL ligand title -> filepath, so lookups
    against edges.csv (which uses original titles) still work."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    os.makedirs(out_dir, exist_ok=True)

    id_to_path = {}
    for mol in suppl:
        if mol is None or not mol.HasProp("_Name"):
            continue
        name = mol.GetProp("_Name").strip()
        if not name:
            continue

        safe_name = sanitize_filename(name)
        out_path = os.path.join(out_dir, f"{safe_name}.sdf")
        try:
            writer = Chem.SDWriter(out_path)
            writer.write(mol)
            writer.close()
            id_to_path[name] = out_path
        except Exception as e:
            print(f"    [warn] could not write ligand '{name}': {e}")

    return id_to_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/schrodinger_fep_benchmark")
    parser.add_argument("--out_sdf_dir", default="data/processed/sdf_schrodinger")
    parser.add_argument("--out_csv", default="data/processed/edges_schrodinger.csv")
    args = parser.parse_args()

    structure_root = os.path.join(args.raw_dir, "fep_benchmark_inputs", "structure_inputs")
    edge_csvs = glob.glob(os.path.join(structure_root, "**", "*_edges.csv"), recursive=True)
    print(f"Found {len(edge_csvs)} subsets to process")

    final_rows = []
    skipped_nan = 0
    skipped_unresolved = 0

    for edges_path in edge_csvs:
        group = os.path.basename(os.path.dirname(edges_path))
        subset_name = os.path.basename(edges_path).replace("_edges.csv", "")
        target = f"{group}__{subset_name}"

        sdf_path = edges_path.replace("_edges.csv", "_ligands.sdf")
        pdb_path = edges_path.replace("_edges.csv", "_protein.pdb")

        if not os.path.exists(sdf_path):
            print(f"[skip] {target}: no matching ligands.sdf")
            continue

        id_to_path = split_ligands_sdf(sdf_path, os.path.join(args.out_sdf_dir, target))

        try:
            df = pd.read_csv(edges_path)
        except Exception as e:
            print(f"[skip] {target}: could not read edges.csv ({e})")
            continue

        # Column names aren't fully consistent across all 92 files (e.g.
        # charge_annhil/egfr uses 'Ligand1'/'Exp. ddG (kcal/mol)' instead of
        # 'Ligand 1'/'ddG (kcal/mol)'). Normalize known variants rather than
        # dropping the whole subset.
        column_aliases = {
            "Ligand1": "Ligand 1",
            "Ligand2": "Ligand 2",
            "Exp. ddG (kcal/mol)": "ddG (kcal/mol)",
            "Exp ddG (kcal/mol)": "ddG (kcal/mol)",
        }
        df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})

        required_cols = {"Ligand 1", "Ligand 2", "ddG (kcal/mol)"}
        if not required_cols.issubset(df.columns):
            print(f"[skip] {target}: unexpected columns {list(df.columns)} "
                  f"(expected {required_cols})")
            continue

        for _, row in df.iterrows():
            a_name, b_name, ddg = row["Ligand 1"], row["Ligand 2"], row["ddG (kcal/mol)"]

            if pd.isna(ddg):
                skipped_nan += 1
                continue

            a_path = id_to_path.get(str(a_name).strip())
            b_path = id_to_path.get(str(b_name).strip())
            if a_path is None or b_path is None:
                skipped_unresolved += 1
                continue

            final_rows.append({
                "target": target,
                "ligand_A_id": a_name,
                "ligand_B_id": b_name,
                "ligand_A_sdf": a_path,
                "ligand_B_sdf": b_path,
                "protein_pdb": pdb_path if os.path.exists(pdb_path) else None,
                "exp_ddg": ddg,
                "fep_pred_ddg": None,   # not available in this file; see module docstring
                "source": "schrodinger",
            })

    out_df = pd.DataFrame(final_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\nWrote {len(out_df)} resolved edges to {args.out_csv}")
    print(f"Skipped {skipped_nan} edges with NaN ddG, {skipped_unresolved} with unresolved ligand IDs")
    print(f"Unique targets/subsets: {out_df['target'].nunique() if len(out_df) else 0}")


if __name__ == "__main__":
    main()