import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from load_2a import load_subject_2a  # must exist in your project

# -----------------------------
# Feature extraction (multi-domain)
# -----------------------------
from scipy.signal import welch
import pywt

def hjorth_params(x: np.ndarray):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return activity, mobility, complexity

def extract_time_features(epoch: np.ndarray):
    feats = []
    for ch in epoch:
        mean = np.mean(ch)
        std = np.std(ch)
        var = np.var(ch)
        rms = np.sqrt(np.mean(ch**2))
        act, mob, comp = hjorth_params(ch)
        feats.extend([mean, std, var, rms, act, mob, comp])
    return np.array(feats, dtype=np.float32)

def bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def extract_freq_features(epoch: np.ndarray, sfreq: float = 250.0):
    bands = [(1,4), (4,8), (8,13), (13,30), (30,40)]
    feats = []
    for ch in epoch:
        freqs, psd = welch(ch, fs=sfreq, nperseg=int(sfreq), noverlap=int(sfreq // 2))
        total = np.trapz(psd, freqs) + 1e-12
        for a, b in bands:
            bp = bandpower(psd, freqs, a, b)
            feats.append(bp / total)  # relative power
    return np.array(feats, dtype=np.float32)

def extract_tfr_features(epoch: np.ndarray):
    feats = []
    for ch in epoch:
        coeffs = pywt.wavedec(ch, 'db4', level=4)
        energies = [np.sum(c**2) for c in coeffs]
        feats.extend(energies)
    return np.array(feats, dtype=np.float32)

def extract_multi_domain(epoch: np.ndarray, sfreq: float = 250.0):
    return np.concatenate([
        extract_time_features(epoch),
        extract_freq_features(epoch, sfreq),
        extract_tfr_features(epoch)
    ])

# -----------------------------
# CSP (OVR) - fit within train folds only
# -----------------------------
from scipy.linalg import eigh

def cov_norm(X: np.ndarray):
    c = X @ X.T
    return c / (np.trace(c) + 1e-12)

def fit_csp_ovr(X: np.ndarray, y: np.ndarray, n_comp: int = 2):
    classes = np.unique(y)
    W_list = []
    for cls in classes:
        X1 = X[y == cls]
        X0 = X[y != cls]
        C1 = np.mean([cov_norm(e) for e in X1], axis=0)
        C0 = np.mean([cov_norm(e) for e in X0], axis=0)
        evals, evecs = eigh(C1, C1 + C0)
        ix = np.argsort(evals)[::-1]
        evecs = evecs[:, ix]
        W = np.concatenate([evecs[:, :n_comp], evecs[:, -n_comp:]], axis=1)
        W_list.append(W)
    return W_list

def transform_csp_features(W_list, epoch: np.ndarray):
    feats = []
    for W in W_list:
        Z = W.T @ epoch
        var = np.var(Z, axis=1)
        var = var / (np.sum(var) + 1e-12)
        feats.extend(np.log(var + 1e-12))
    return np.array(feats, dtype=np.float32)

# -----------------------------
# Baseline run (NO feature selection)
# -----------------------------
def run_subject_baseline(data_dir: str, subj: int, outer_splits: int = 5, seed: int = 0):
    X, y, _, sf, _ = load_subject_2a(data_dir, subj=subj, session="T")

    # Precompute multi-domain features once per subject
    t0 = time.perf_counter()
    X_md = np.vstack([extract_multi_domain(ep, sfreq=sf) for ep in X])
    md_ms_per_trial = (time.perf_counter() - t0) / len(X) * 1000.0

    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_acc, fold_f1, fold_kap = [], [], []
    fold_infer_ms = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        Xtr_eeg, Xte_eeg = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # CSP fit on train fold only
        W_list = fit_csp_ovr(Xtr_eeg, ytr, n_comp=2)
        Xtr_csp = np.vstack([transform_csp_features(W_list, ep) for ep in Xtr_eeg])
        Xte_csp = np.vstack([transform_csp_features(W_list, ep) for ep in Xte_eeg])

        # Combine multi-domain + CSP
        Xtr_all = np.hstack([X_md[tr_idx], Xtr_csp])
        Xte_all = np.hstack([X_md[te_idx], Xte_csp])

        # Scale (fit on train only)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_all)
        Xte_s = scaler.transform(Xte_all)

        clf = SVC(C=2.0, kernel="rbf", gamma="scale")
        clf.fit(Xtr_s, ytr)

        t1 = time.perf_counter()
        pred = clf.predict(Xte_s)
        infer_ms = (time.perf_counter() - t1) / len(yte) * 1000.0

        acc = accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")
        kap = cohen_kappa_score(yte, pred)

        fold_acc.append(acc)
        fold_f1.append(f1m)
        fold_kap.append(kap)
        fold_infer_ms.append(infer_ms)

        print(
            f"[BASELINE] Subject {subj} | fold {fold_i}/{outer_splits} | "
            f"acc={acc:.3f} f1={f1m:.3f} kappa={kap:.3f} infer_ms={infer_ms:.6f}"
        )

    return {
        "subject": subj,
        "n_trials": int(len(y)),
        "outer_splits": int(outer_splits),
        "cv_acc_mean": float(np.mean(fold_acc)),
        "cv_acc_std": float(np.std(fold_acc)),
        "cv_macroF1_mean": float(np.mean(fold_f1)),
        "cv_macroF1_std": float(np.std(fold_f1)),
        "cv_kappa_mean": float(np.mean(fold_kap)),
        "cv_kappa_std": float(np.std(fold_kap)),
        "feature_extract_md_ms_per_trial": float(md_ms_per_trial),
        "inference_ms_per_trial_mean": float(np.mean(fold_infer_ms)),
        "inference_ms_per_trial_std": float(np.std(fold_infer_ms)),
    }

def main():
    data_dir = r"D:\Articles\1-ongoing\2-IEEE\BCICIV2a"
    outer_splits = 5
    seed = 0

    all_res = []
    for subj in range(1, 10):
        r = run_subject_baseline(data_dir, subj, outer_splits=outer_splits, seed=seed)
        all_res.append(r)

    df = pd.DataFrame(all_res)
    df.to_csv("baseline_Tonly_per_subject.csv", index=False)

    summary = df[[
        "cv_acc_mean", "cv_macroF1_mean", "cv_kappa_mean",
        "feature_extract_md_ms_per_trial",
        "inference_ms_per_trial_mean"
    ]].agg(["mean", "std"])
    summary.to_csv("baseline_Tonly_summary.csv")

    # Figure: per-subject Macro-F1
    plt.figure()
    plt.bar(df["subject"].astype(str), df["cv_macroF1_mean"])
    plt.xlabel("Subject")
    plt.ylabel("CV Macro F1-score (mean)")
    plt.title("Baseline T-only Performance (Macro-F1)")
    plt.grid(True, axis="y")
    plt.savefig("Figure_Baseline_Tonly_PerSubject_MacroF1.png", dpi=300)

    print("\nSaved:")
    print(" - baseline_Tonly_per_subject.csv")
    print(" - baseline_Tonly_summary.csv")
    print(" - Figure_Baseline_Tonly_PerSubject_MacroF1.png")

if __name__ == "__main__":
    main()
