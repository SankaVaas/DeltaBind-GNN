"""
inspect_schrodinger.py

Run this BEFORE writing a parser for the Schrodinger benchmark repo — we
don't yet know its actual file layout (unlike Merck, which we already
reverse-engineered). This just prints structure so we can write a correct
parser in one shot instead of guessing.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/schrodinger_fep_benchmark")
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    print(f"=== Directory tree of {args.raw_dir} (depth {args.max_depth}) ===")
    base_depth = args.raw_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(args.raw_dir):
        depth = root.count(os.sep) - base_depth
        if depth > args.max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files)[:10]:
            print(f"{indent}  {f}")
        if len(files) > 10:
            print(f"{indent}  ... ({len(files) - 10} more files)")

    print("\n=== Sample of any CSV files found (first 3, first 5 rows each) ===")
    import glob
    import pandas as pd
    csvs = glob.glob(os.path.join(args.raw_dir, "**", "*.csv"), recursive=True)
    for csv_path in csvs[:3]:
        print(f"\n--- {csv_path} ---")
        try:
            df = pd.read_csv(csv_path)
            print("columns:", list(df.columns))
            print(df.head())
        except Exception as e:
            print(f"could not read: {e}")

    print("\n=== Any SDF/MOL2/PDB structure files found ===")
    for ext in ["sdf", "mol2", "pdb", "mae"]:
        matches = glob.glob(os.path.join(args.raw_dir, "**", f"*.{ext}"), recursive=True)
        print(f"  .{ext}: {len(matches)} files. Example: {matches[0] if matches else 'none'}")


if __name__ == "__main__":
    main()