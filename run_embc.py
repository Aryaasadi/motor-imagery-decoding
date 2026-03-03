import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from load_2a import load_subject_2a

# -----------------------------
# Feature extraction (multi-domain)
# -----------------------------
from scipy.signal import welch
import pywt

def hjorth_params(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return activity, mobility, complexity

def extract_time_features(epoch):
    feats = []
    for ch in epoch:
        mean = np.mean(ch)
        std = np.std(ch)
        var = np.var(ch)
        rms = np.sqrt(np.mean(ch**2))
        act, mob, comp = hjorth_params(ch)
        feats.extend([mean, std, var, rms, act, mob, comp])
    return np.array(feats, dtype=np.float32)

def bandpower(psd, freqs, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def extract_freq_features(epoch, sfreq=250):
    bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
    feats = []
    for ch in epoch:
        freqs, psd = welch(ch, fs=sfreq, nperseg=int(sfreq), noverlap=int(sfreq//2))
        total = np.trapz(psd, freqs) + 1e-12
        for (a,b) in bands:
            bp = bandpower(psd, freqs, a, b)
            feats.append(bp/total)  # relative power
    return np.array(feats, dtype=np.float32)

def extract_tfr_features(epoch):
    feats = []
    for ch in epoch:
        coeffs = pywt.wavedec(ch, 'db4', level=4)
        energies = [np.sum(c**2) for c in coeffs]
        feats.extend(energies)
    return np.array(feats, dtype=np.float32)

def extract_multi_domain(epoch, sfreq=250):
    return np.concatenate([
        extract_time_features(epoch),
        extract_freq_features(epoch, sfreq),
        extract_tfr_features(epoch)
    ])

# -----------------------------
# CSP (OVR) - must be fit within outer-train only
# -----------------------------
from scipy.linalg import eigh

def cov_norm(X):
    c = X @ X.T
    return c / (np.trace(c) + 1e-12)

def fit_csp_ovr(X, y, n_comp=2):
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

def transform_csp_features(W_list, epoch):
    feats = []
    for W in W_list:
        Z = W.T @ epoch
        var = np.var(Z, axis=1)
        var = var / (np.sum(var) + 1e-12)
        feats.extend(np.log(var + 1e-12))
    return np.array(feats, dtype=np.float32)

# -----------------------------
# Feature selection + stability
# -----------------------------
def select_topk_rf(X, y, k=100, seed=0):
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:k]
    return idx

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / (len(a | b) + 1e-12)

# -----------------------------
# Nested CV on T
# -----------------------------
def run_subject(data_dir, subj, k_list, outer_splits=5, inner_splits=5, seed=0):
    X, y, _, sf, _ = load_subject_2a(data_dir, subj=subj, session="T")

    # Precompute multi-domain features once (per-trial only, safe)
    t0 = time.perf_counter()
    X_md = np.vstack([extract_multi_domain(ep, sfreq=sf) for ep in X])
    md_ms_per_trial = (time.perf_counter() - t0) / len(X) * 1000.0

    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_acc, fold_f1, fold_kap = [], [], []
    fold_bestk, fold_stab, fold_infer = [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        Xtr_eeg, Xte_eeg = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # Fit CSP on outer-train only
        W_list = fit_csp_ovr(Xtr_eeg, ytr, n_comp=2)
        Xtr_csp = np.vstack([transform_csp_features(W_list, ep) for ep in Xtr_eeg])
        Xte_csp = np.vstack([transform_csp_features(W_list, ep) for ep in Xte_eeg])

        # Combine features
        Xtr_all = np.hstack([X_md[tr_idx], Xtr_csp])
        Xte_all = np.hstack([X_md[te_idx], Xte_csp])

        # Scale on outer-train only
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_all)
        Xte_s = scaler.transform(Xte_all)

        # Inner CV to choose k + stability (on outer-train only)
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

        best_k, best_f1 = None, -1
        stability_at_k = {}

        for k in k_list:
            inner_f1 = []
            selected_sets = []

            for in_tr, in_va in inner.split(Xtr_s, ytr):
                Xin_tr, Xin_va = Xtr_s[in_tr], Xtr_s[in_va]
                yin_tr, yin_va = ytr[in_tr], ytr[in_va]

                idx = select_topk_rf(Xin_tr, yin_tr, k=k, seed=seed)
                selected_sets.append(idx)

                clf = SVC(C=2.0, kernel="rbf", gamma="scale")
                clf.fit(Xin_tr[:, idx], yin_tr)
                pred = clf.predict(Xin_va[:, idx])
                inner_f1.append(f1_score(yin_va, pred, average="macro"))

            mean_f1 = float(np.mean(inner_f1))

            js = []
            for i in range(len(selected_sets)):
                for j in range(i + 1, len(selected_sets)):
                    js.append(jaccard(selected_sets[i], selected_sets[j]))
            stability_at_k[k] = float(np.mean(js)) if js else 0.0

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_k = k

        # Train final on full outer-train with best_k
        idx_final = select_topk_rf(Xtr_s, ytr, k=best_k, seed=seed)

        clf = SVC(C=2.0, kernel="rbf", gamma="scale")
        clf.fit(Xtr_s[:, idx_final], ytr)

        t1 = time.perf_counter()
        pred = clf.predict(Xte_s[:, idx_final])
        infer_ms = (time.perf_counter() - t1) / len(yte) * 1000.0

        acc = accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")
        kap = cohen_kappa_score(yte, pred)

        fold_acc.append(acc)
        fold_f1.append(f1m)
        fold_kap.append(kap)
        fold_bestk.append(best_k)
        fold_stab.append(stability_at_k[best_k])
        fold_infer.append(infer_ms)

        print(f"Subject {subj} | fold {fold_i}/{outer_splits} | acc={acc:.3f} f1={f1m:.3f} kappa={kap:.3f} best_k={best_k}")

    return {
        "subject": subj,
        "n_trials": int(len(y)),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "cv_acc_mean": float(np.mean(fold_acc)),
        "cv_acc_std": float(np.std(fold_acc)),
        "cv_macroF1_mean": float(np.mean(fold_f1)),
        "cv_macroF1_std": float(np.std(fold_f1)),
        "cv_kappa_mean": float(np.mean(fold_kap)),
        "cv_kappa_std": float(np.std(fold_kap)),
        "best_k_mean": float(np.mean(fold_bestk)),
        "best_k_mode": int(pd.Series(fold_bestk).mode().iloc[0]),
        "stability_mean_at_best_k": float(np.mean(fold_stab)),
        "feature_extract_md_ms_per_trial": float(md_ms_per_trial),
        "inference_ms_per_trial_mean": float(np.mean(fold_infer)),
    }


if __name__ == "__main__":
    data_dir = r"D:\Articles\1-ongoing\2-IEEE\BCICIV2a"
    k_list = [20, 40, 60, 80, 100, 120, 150, 200, 300, 400]

    all_res = []
    for subj in range(1, 10):
        r = run_subject(data_dir, subj, k_list, outer_splits=5, inner_splits=5, seed=0)
        all_res.append(r)

    df = pd.DataFrame(all_res)
    df.to_csv("embc_Tonly_results_per_subject.csv", index=False)

    summary = df[[
        "cv_acc_mean", "cv_macroF1_mean", "cv_kappa_mean",
        "best_k_mean", "stability_mean_at_best_k",
        "feature_extract_md_ms_per_trial", "inference_ms_per_trial_mean"
    ]].agg(["mean", "std"])
    summary.to_csv("embc_Tonly_results_summary.csv")

    plt.figure()
    plt.bar(df["subject"].astype(str), df["cv_macroF1_mean"])
    plt.xlabel("Subject")
    plt.ylabel("CV Macro F1-score (mean)")
    plt.title("T-only Nested CV Performance (Macro-F1)")
    plt.grid(True, axis="y")
    plt.savefig("Figure_Tonly_PerSubject_MacroF1.png", dpi=300)

    print("\nSaved:")
    print(" - embc_Tonly_results_per_subject.csv")
    print(" - embc_Tonly_results_summary.csv")
    print(" - Figure_Tonly_PerSubject_MacroF1.png")
