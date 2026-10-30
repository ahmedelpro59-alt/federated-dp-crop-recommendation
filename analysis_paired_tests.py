"""
Post-hoc analysis of results/results_raw.csv produced by fl_dp_experiment.py.

1) Paired change in accuracy of each clipping/DP configuration relative to plain FedAvg
   (pooled over the three partitions and five seeds; 15 pairs per configuration).
2) Paired t-tests (two-sided, n = 5 seeds) comparing each non-IID partition with IID
   for every method.

Usage:
    python analysis_paired_tests.py results/results_raw.csv
"""
import sys
import pandas as pd
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/results_raw.csv"
raw = pd.read_csv(path)
fl = raw[raw.setting != "Centralized"]

methods = ["FedAvg", "FedAvg+Clip (z=0)", "FedAvg+DP (z=0.1)",
           "FedAvg+DP (z=0.5)", "FedAvg+DP (z=1.0)", "FedAvg+DP (z=2.0)"]
settings = ["IID", "Non-IID (alpha=0.5)", "Non-IID (alpha=0.1)"]

print("=== 1) Paired change vs plain FedAvg (percentage points) ===")
piv = fl.pivot_table(index=["seed", "setting"], columns="method", values="accuracy")
print(f"{'configuration':22s} {'mean':>8s} {'SD':>6s}   lower/equal/higher")
for m in methods[1:]:
    diff = (piv[m] - piv["FedAvg"]) * 100
    print(f"{m:22s} {diff.mean():8.2f} {diff.std(ddof=1):6.2f}   "
          f"{(diff < 0).sum()}/{(diff == 0).sum()}/{(diff > 0).sum()}")

print("\n=== 2) Paired t-tests: non-IID vs IID within seeds (n = 5) ===")
print(f"{'method':22s} {'setting':22s} {'mean diff':>10s} {'p-value':>8s}")
for m in methods:
    iid = fl[(fl.method == m) & (fl.setting == "IID")].sort_values("seed").accuracy.values * 100
    for s in settings[1:]:
        v = fl[(fl.method == m) & (fl.setting == s)].sort_values("seed").accuracy.values * 100
        d = v - iid
        p = stats.ttest_1samp(d, 0)[1] if d.std() > 0 else 1.0
        print(f"{m:22s} {s:22s} {d.mean():10.2f} {p:8.3f}")
