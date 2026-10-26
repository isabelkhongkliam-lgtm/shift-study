"""
Reliability of Tabular ML Models Under Distribution Shift
Accuracy + Calibration (ECE, Brier, NLL) + Conformal coverage, ID vs OOD.

    python shift_study.py                         # full run (3 datasets, 5 seeds)
    python shift_study.py --synthetic --seeds 2   # quick test, no internet needed

Outputs (in ./results): all_runs.csv, summary.csv, ece_<dataset>.png, coverage.png
"""
import argparse, os, warnings
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_classification
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

# shift_col: rows are sorted by this column; lowest 80% = source, top 20% = shifted (OOD).
# shift_col=None -> original row order (Electricity is time-ordered = natural temporal shift).
DATASETS = {
    "adult":       dict(kw=dict(name="adult", version=2), shift_col="age"),
    "bank":        dict(kw=dict(data_id=1461),            shift_col="V1"),   # V1 = age
    "electricity": dict(kw=dict(data_id=151),             shift_col=None),
}
ALPHA = 0.1  # conformal target: 90% coverage


# ---------- data ----------
@lru_cache(maxsize=None)
def _fetch(key):
    kw = DATASETS[key]["kw"]
    try:
        return fetch_openml(**kw, as_frame=True, parser="auto")
    except Exception:
        return fetch_openml(**kw, as_frame=True, parser="liac-arff")


def load_openml(key, max_rows, seed):
    cfg = DATASETS[key]
    d = _fetch(key)
    X, y = d.data.copy(), LabelEncoder().fit_transform(d.target)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object)
    shift_col = cfg["shift_col"]
    if shift_col is not None and shift_col not in X.columns:
        shift_col = X.select_dtypes("number").columns[0]
    if len(X) > max_rows:  # subsample but keep original order
        idx = np.sort(np.random.RandomState(seed).choice(len(X), max_rows, replace=False))
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y, shift_col


def load_synthetic(seed):
    Xn, y = make_classification(n_samples=20000, n_features=12, n_informative=6,
                                flip_y=0.05, random_state=seed)
    X = pd.DataFrame(Xn, columns=[f"f{i}" for i in range(12)])
    X["cat"] = np.where(X["f1"] > 0, "a", "b").astype(object)
    return X, y, "f0"


def make_splits(X, y, shift_col, seed):
    n, rng = len(X), np.random.RandomState(seed)
    if shift_col is None:
        order = np.arange(n)
    else:
        v = pd.to_numeric(X[shift_col], errors="coerce").fillna(0).values + rng.uniform(0, 1e-6, n)
        order = np.argsort(v)
    cut = int(0.8 * n)
    src, ood = order[:cut], order[cut:]
    tr, rest = train_test_split(src, test_size=0.4, random_state=seed)      # 60% train
    cal, idt = train_test_split(rest, test_size=0.625, random_state=seed)   # 15% calib, 25% ID test
    return tr, cal, idt, ood


def make_pre(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ])


# ---------- models ----------
def get_models(seed):
    m = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
        "MLP": MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     n_jobs=-1, random_state=seed, eval_metric="logloss")
    except ImportError:
        print("xgboost not installed -> skipping XGBoost (pip install xgboost)")
    return m


# ---------- metrics ----------
def ece(p, y, bins=15):
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def metrics(p, y):
    return dict(acc=(p.argmax(1) == y).mean(), auc=roc_auc_score(y, p[:, 1]),
                ece=ece(p, y), brier=brier_score_loss(y, p[:, 1]),
                nll=log_loss(y, np.clip(p, 1e-6, 1), labels=[0, 1]))


def isotonic(p_cal, y_cal, p_test):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_cal[:, 1], y_cal)
    q = iso.predict(p_test[:, 1])
    return np.c_[1 - q, q]


def conformal(p_cal, y_cal, p_test, y_test, alpha=ALPHA):
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]          # nonconformity score
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = p_test >= 1 - q
    return sets[np.arange(len(y_test)), y_test].mean(), sets.sum(1).mean()


# ---------- main ----------
def run(args):
    os.makedirs(args.out, exist_ok=True)
    rows = []
    keys = ["synthetic"] if args.synthetic else args.datasets
    for key in keys:
        print(f"\n=== Dataset: {key} ===")
        for seed in range(args.seeds):
            X, y, sc = load_synthetic(seed) if args.synthetic else load_openml(key, args.max_rows, seed)
            tr, cal, idt, ood = make_splits(X, y, sc, seed)
            pre = make_pre(X).fit(X.iloc[tr])
            Z = {k: pre.transform(X.iloc[i]) for k, i in
                 dict(tr=tr, cal=cal, ID=idt, OOD=ood).items()}
            Y = dict(tr=y[tr], cal=y[cal], ID=y[idt], OOD=y[ood])
            for mname, model in get_models(seed).items():
                model.fit(Z["tr"], Y["tr"])
                P = {k: model.predict_proba(Z[k]) for k in ["cal", "ID", "OOD"]}
                for split in ["ID", "OOD"]:
                    r = metrics(P[split], Y[split])
                    cov, size = conformal(P["cal"], Y["cal"], P[split], Y[split])
                    rows.append(dict(dataset=key, seed=seed, model=mname, calib="raw",
                                     split=split, coverage=cov, set_size=size, **r))
                    p_iso = isotonic(P["cal"], Y["cal"], P[split])
                    rows.append(dict(dataset=key, seed=seed, model=mname, calib="isotonic",
                                     split=split, coverage=np.nan, set_size=np.nan,
                                     **metrics(p_iso, Y[split])))
            print(f"  seed {seed} done")
        pd.DataFrame(rows).to_csv(f"{args.out}/partial_{key}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}/all_runs.csv", index=False)
    cols = ["acc", "auc", "ece", "brier", "nll", "coverage", "set_size"]
    summ = df.groupby(["dataset", "model", "calib", "split"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{args.out}/summary.csv")
    print("\n--- Summary (raw models, mean over seeds) ---")
    print(df[df.calib == "raw"].groupby(["dataset", "model", "split"])[cols].mean().round(3).to_string())
    make_plots(df, args.out)
    print(f"\nSaved tables and figures to ./{args.out}/")


def make_plots(df, out):
    raw = df[df.calib == "raw"]
    for ds, g in raw.groupby("dataset"):
        m = g.groupby(["model", "split"])["ece"].agg(["mean", "std"]).unstack()
        fig, ax = plt.subplots(figsize=(7, 4))
        x, w = np.arange(len(m)), 0.38
        ax.bar(x - w / 2, m["mean"]["ID"], w, yerr=m["std"]["ID"], label="In-distribution", capsize=3)
        ax.bar(x + w / 2, m["mean"]["OOD"], w, yerr=m["std"]["OOD"], label="Shifted (OOD)", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(m.index); ax.set_ylabel("ECE (lower is better)")
        ax.set_title(f"Calibration error under shift - {ds}"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{out}/ece_{ds}.png", dpi=200); plt.close()
    cov = raw[raw.split == "OOD"].groupby(["dataset", "model"])["coverage"].mean().unstack()
    ax = cov.plot(kind="bar", figsize=(8, 4))
    ax.axhline(1 - ALPHA, color="k", ls="--", label=f"target {1 - ALPHA:.0%}")
    ax.set_ylabel("Conformal coverage on OOD data"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Does conformal coverage hold under shift?"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{out}/coverage.png", dpi=200); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=30000)
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test")
    ap.add_argument("--out", default="results")
    run(ap.parse_args())