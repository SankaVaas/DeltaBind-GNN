"""
inspect_schrodinger_edges.py

We know fep_benchmark_inputs/structure_inputs/<group>/<subset>_edges.csv is
the real per-edge file (co-located with _ligands.sdf and _protein.pdb), but
haven't seen its columns yet -- this samples a few directly so the parser
isn't written against a guess.
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/schrodinger_fep_benchmark")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    edges_dir = os.path.join(args.raw_dir, "fep_benchmark_inputs", "structure_inputs")
    edge_csvs = glob.glob(os.path.join(edges_dir, "**", "*_edges.csv"), recursive=True)
    print(f"Found {len(edge_csvs)} *_edges.csv files total\n")

    for path in edge_csvs[: args.n_samples]:
        print(f"--- {path} ---")
        df = pd.read_csv(path)
        print("columns:", list(df.columns))
        print(df.head(8))
        print()

    # Also cross-check: do the ligand IDs in one edges.csv actually match
    # the molecule titles inside the matching ligands.sdf?
    if edge_csvs:
        sample_edges = edge_csvs[0]
        sdf_path = sample_edges.replace("_edges.csv", "_ligands.sdf")
        print(f"--- Cross-check: ligand names in {sdf_path} ---")
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        names = [m.GetProp("_Name") for m in suppl if m is not None and m.HasProp("_Name")]
        print(f"{len(names)} ligand names found, first 10: {names[:10]}")


if __name__ == "__main__":
    main()