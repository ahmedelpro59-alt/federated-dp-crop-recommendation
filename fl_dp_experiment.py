"""
Federated Learning (FedAvg) with and without Differential Privacy noise
on a crop-recommendation tabular dataset (Kaggle: Crop Recommender with Soil Nutrients).

Usage (Colab or local):
    python fl_dp_experiment.py path/to/dataset.csv
Assumptions: all columns are numeric features except the LAST column = crop label.

Outputs (in ./results):
    results_table.csv, fig_privacy_vs_accuracy.png, fig_iid_vs_noniid.png
Note: DP here is the Gaussian mechanism on clipped client updates, reported by the
noise multiplier z. Formal (epsilon, delta) accounting is NOT computed.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# ---------------- settings ----------------
SEEDS = [0, 1, 2, 3, 4]
N_CLIENTS = 5
ROUNDS = 50
LOCAL_EPOCHS = 2
LR = 0.1
BATCH = 32
CLIP = 1.0                       # L2 clipping norm for client updates
NOISE_MULTIPLIERS = [0.0, 0.1, 0.5, 1.0, 2.0]   # 0.0 = clipping only, no noise
SETTINGS = {"IID": None, "Non-IID (alpha=0.5)": 0.5, "Non-IID (alpha=0.1)": 0.1}
OUT = "results"


# ---------------- model: multinomial logistic regression ----------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gradient(W, X, y):
    P = softmax(X @ W)
    P[np.arange(len(y)), y] -= 1.0
    return X.T @ P / len(y)


def local_train(W, X, y, epochs, lr, batch, rng):
    W = W.copy()
    n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            W -= lr * gradient(W, X[b], y[b])
    return W


def evaluate(W, X, y):
    pred = (X @ W).argmax(axis=1)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


# ---------------- data partitioning ----------------
def partition(y, k, alpha, rng, min_size=10):
    """alpha=None -> IID split; otherwise Dirichlet(alpha) label-skew split."""
    n = len(y)
    if alpha is None:
        return np.array_split(rng.permutation(n), k)
    while True:
        parts = [[] for _ in range(k)]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet([alpha] * k)
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]


# ---------------- federated training ----------------
def federated(Xtr, ytr, Xte, yte, n_classes, alpha, seed, clip=None, z=0.0):
    rng = np.random.default_rng(seed)
    parts = partition(ytr, N_CLIENTS, alpha, rng)
    W = np.zeros((Xtr.shape[1], n_classes))
    sizes = np.array([len(p) for p in parts], dtype=float)
    for _ in range(ROUNDS):
        deltas = []
        for p in parts:
            Wl = local_train(W, Xtr[p], ytr[p], LOCAL_EPOCHS, LR, BATCH, rng)
            d = Wl - W
            if clip is not None:
                norm = np.linalg.norm(d)
                d = d * min(1.0, clip / (norm + 1e-12))
            deltas.append(d)
        if clip is None:                      # plain FedAvg (weighted by data size)
            agg = sum(w * d for w, d in zip(sizes / sizes.sum(), deltas))
        else:                                 # DP-FedAvg (uniform mean + Gaussian noise)
            agg = np.mean(deltas, axis=0)
            if z > 0:
                agg = agg + rng.normal(0, z * clip / N_CLIENTS, size=agg.shape)
        W = W + agg
    return evaluate(W, Xte, yte)


def centralized(Xtr, ytr, Xte, yte, n_classes, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((Xtr.shape[1], n_classes))
    W = local_train(W, Xtr, ytr, ROUNDS * LOCAL_EPOCHS, LR, BATCH, rng)
    return evaluate(W, Xte, yte)


# ---------------- main ----------------
def main(path):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("Classes:", df.iloc[:, -1].nunique())
    X_raw = df.iloc[:, :-1].select_dtypes(include=[np.number]).values.astype(float)
    y_raw = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    n_classes = len(np.unique(y_raw))

    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=seed)
        sc = StandardScaler().fit(Xtr)
        Xtr = np.hstack([sc.transform(Xtr), np.ones((len(Xtr), 1))])   # bias column
        Xte = np.hstack([sc.transform(Xte), np.ones((len(Xte), 1))])

        acc, f1 = centralized(Xtr, ytr, Xte, yte, n_classes, seed)
        rows.append(dict(seed=seed, setting="Centralized", method="Centralized",
                         accuracy=acc, f1_macro=f1))

        for sname, alpha in SETTINGS.items():
            acc, f1 = federated(Xtr, ytr, Xte, yte, n_classes, alpha, seed)
            rows.append(dict(seed=seed, setting=sname, method="FedAvg",
                             accuracy=acc, f1_macro=f1))
            for z in NOISE_MULTIPLIERS:
                acc, f1 = federated(Xtr, ytr, Xte, yte, n_classes, alpha, seed,
                                    clip=CLIP, z=z)
                label = "FedAvg+Clip (z=0)" if z == 0 else f"FedAvg+DP (z={z})"
                rows.append(dict(seed=seed, setting=sname, method=label,
                                 accuracy=acc, f1_macro=f1))
        print(f"seed {seed} done")

    res = pd.DataFrame(rows)
    summary = (res.groupby(["setting", "method"])[["accuracy", "f1_macro"]]
                  .agg(["mean", "std"]).round(4))
    summary.to_csv(f"{OUT}/results_table.csv")
    res.to_csv(f"{OUT}/results_raw.csv", index=False)
    print(summary)

    # Figure 1: privacy (noise) vs accuracy for each data setting
    plt.figure(figsize=(7, 4.5))
    for sname in SETTINGS:
        means, stds = [], []
        for z in NOISE_MULTIPLIERS:
            m = "FedAvg+Clip (z=0)" if z == 0 else f"FedAvg+DP (z={z})"
            v = res[(res.setting == sname) & (res.method == m)].accuracy
            means.append(v.mean())
            stds.append(v.std())
        plt.errorbar(NOISE_MULTIPLIERS, means, yerr=stds, marker="o", capsize=3, label=sname)
    cen = res[res.method == "Centralized"].accuracy.mean()
    plt.axhline(cen, ls="--", color="gray", label="Centralized")
    plt.xlabel("Noise multiplier z (higher = stronger privacy)")
    plt.ylabel("Test accuracy")
    plt.title("Privacy-accuracy trade-off in federated crop recommendation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_privacy_vs_accuracy.png", dpi=200)
    plt.close()

    # Figure 2: IID vs non-IID for plain FedAvg
    plt.figure(figsize=(6, 4))
    names = ["Centralized"] + list(SETTINGS)
    vals, errs = [], []
    for n in names:
        v = res[res.method == "Centralized"].accuracy if n == "Centralized" else \
            res[(res.setting == n) & (res.method == "FedAvg")].accuracy
        vals.append(v.mean())
        errs.append(v.std())
    plt.bar(range(len(names)), vals, yerr=errs, capsize=4)
    plt.xticks(range(len(names)), names, rotation=15)
    plt.ylabel("Test accuracy")
    plt.title("Effect of data heterogeneity (FedAvg)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_iid_vs_noniid.png", dpi=200)
    plt.close()
    print(f"\nSaved results to ./{OUT}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python fl_dp_experiment.py dataset.csv")
    main(sys.argv[1])
