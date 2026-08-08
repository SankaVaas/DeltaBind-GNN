import pandas as pd
from src.evaluate import compute_metrics

oof_df = pd.read_csv("results/kfold_oof_predictions.csv")

# restrict to exactly the edges where FEP+ has a prediction, for both methods
fair_df = oof_df.dropna(subset=["fep_ddg"])
print(f"Fair comparison set: {len(fair_df)} edges (same subset for both methods)")

print("\n=== DeltaBind-GNN on the SAME subset FEP+ was evaluated on ===")
model_fair = compute_metrics(fair_df["pred_ddg"].values, fair_df["exp_ddg"].values)
for k, v in model_fair.items():
    print(f"  {k}: {v:.4f}")

print("\n=== FEP+ (for reference, same numbers as before) ===")
fep_fair = compute_metrics(fair_df["fep_ddg"].values, fair_df["exp_ddg"].values)
for k, v in fep_fair.items():
    print(f"  {k}: {v:.4f}")