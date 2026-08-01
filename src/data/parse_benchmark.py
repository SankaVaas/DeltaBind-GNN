"""
parse_benchmark.py

Unifies the two raw benchmark sources (Schrodinger public FEP+ benchmark and
the Merck/Schindler FEP benchmark) into a single edge table:

    target | ligand_A_id | ligand_B_id | ligand_A_sdf | ligand_B_sdf |
    protein_pdb | exp_ddg | fep_pred_ddg (if available) | source

Each row is one "edge": a pair of congeneric ligands bound to the same
target, with experimental relative binding free energy (ddG, kcal/mol) as
the training label, and the FEP+ prediction (where published) kept as a
baseline to compare against at evaluation time.

NOTE: the exact file layout inside each cloned repo can shift between
versions. This script is written defensively: it globs for the expected
file types (CSV summary files, per-target SDF/mol2 ligand files, PDB
protein files) rather than hardcoding exact paths. Run this after
`data/download_data.sh` and inspect data/processed/edges.csv before
trusting it blindly — always sanity check row counts against the paper
numbers (Schrodinger JACS set / Merck: 8 targets, 264 ligands, 550 edges).
"""

import argparse
import glob
import os

import pandas as pd


def find_files(root: str, pattern: str):
    return glob.glob(os.path.join(root, "**", pattern), recursive=True)


def parse_merck_benchmark(raw_dir: str) -> pd.DataFrame:
    """Parse the Merck/Schindler fep-benchmark repo structure.

    Expected layout (per target subfolder): ligand SDF files + a results/
    CSV with experimental and FEP-predicted ddG per edge.
    """
    merck_root = os.path.join(raw_dir, "merck_fep_benchmark")
    rows = []

    csv_files = find_files(merck_root, "*.csv")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        # Heuristics: look for columns that look like ligand pair + ddG data.
        has_pair_cols = any("ligand" in c or "edge" in c for c in cols_lower)
        has_ddg_col = any("ddg" in c or "exp" in c for c in cols_lower)
        if not (has_pair_cols and has_ddg_col):
            continue

        target = os.path.basename(os.path.dirname(csv_path))
        df["target"] = target
        df["source"] = "merck"
        df["_source_csv"] = csv_path
        rows.append(df)

    if not rows:
        print(f"[warn] No usable CSVs found under {merck_root}. "
              f"Inspect the repo structure manually and adjust parse_merck_benchmark().")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)


def parse_schrodinger_benchmark(raw_dir: str) -> pd.DataFrame:
    """Parse the Schrodinger public_binding_free_energy_benchmark repo.

    Expected layout: fep_benchmark_inputs/ (structures) and output CSVs with
    Schrodinger's own FEP+ predictions alongside experimental values.
    """
    schrod_root = os.path.join(raw_dir, "schrodinger_fep_benchmark")
    rows = []

    csv_files = find_files(schrod_root, "*.csv")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        has_ddg_col = any("ddg" in c or "dg" in c for c in cols_lower)
        if not has_ddg_col:
            continue

        # target name is usually encoded in the parent directory name
        target = os.path.basename(os.path.dirname(csv_path))
        df["target"] = target
        df["source"] = "schrodinger"
        df["_source_csv"] = csv_path
        rows.append(df)

    if not rows:
        print(f"[warn] No usable CSVs found under {schrod_root}. "
              f"Inspect the repo structure manually and adjust parse_schrodinger_benchmark().")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    merck_df = parse_merck_benchmark(args.raw_dir)
    schrod_df = parse_schrodinger_benchmark(args.raw_dir)

    combined = pd.concat([merck_df, schrod_df], ignore_index=True, sort=False)
    out_path = os.path.join(args.out_dir, "edges_raw.csv")
    combined.to_csv(out_path, index=False)

    print(f"Wrote {len(combined)} raw rows to {out_path}")
    print("Targets found:", sorted(combined["target"].dropna().unique().tolist()) if len(combined) else "none")
    print("\nNEXT STEP: open this CSV, confirm which columns hold ligand IDs, "
          "structure file paths, experimental ddG, and FEP+ predicted ddG, "
          "then finalize the column mapping in this script (search for '_source_csv').")


if __name__ == "__main__":
    main()
