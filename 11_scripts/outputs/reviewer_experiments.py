#!/usr/bin/env python3
"""
Reviewer-response experiments for the second-round review.
Runs on the public repo data (OpenCap drop-jump metadata + REHAB24-6 per-rep biomarkers
and smoothed trajectories). Produces printed results + CSV outputs in scripts/outputs/.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

REPO = Path("/home/z/my-project/repo")
OUT = Path("/home/z/my-project/scripts/outputs")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

SQUAT_CSV = REPO / "14_rehab24_outputs/biomarkers_per_rep/rehab24_squat_per_rep_biomarkers.csv"
LUNGE_CSV = REPO / "15_rehab24_lunge_outputs/biomarkers_per_rep/rehab24_lunge_per_rep_biomarkers.csv"
SQUAT_TRAJ = REPO / "14_rehab24_outputs/smoothed_per_rep"
LUNGE_TRAJ = REPO / "15_rehab24_lunge_outputs/smoothed_per_rep"

results = {}

print("=" * 78)
print("E1. VERIFICATION OF THE UNCERTAINTY CHAIN (Reviewer Comments 2 & 6)")
print("=" * 78)
agree = pd.read_csv(REPO / "16_opencap_dropjump_outputs/metadata/phase6_agreement_final.csv")
rows = []
for _, r in agree.iterrows():
    if r["biomarker"].startswith("#5"):
        continue
    width = r["loa_upper"] - r["loa_lower"]
    sd_total = width / 3.92
    var_total = sd_total**2
    rows.append({"biomarker": r["biomarker"], "bias": r["bias(video-ik)"],
                 "loa_lower": r["loa_lower"], "loa_upper": r["loa_upper"],
                 "width": width, "sd_total": sd_total, "var_total": var_total})
e1 = pd.DataFrame(rows)
e1["var_paper"] = [104.9867, 37.4132, 150.8291, 4346.2534]
print(e1.round(4).to_string(index=False))

# static turnaround (same-instant) errors
spe = pd.read_csv(REPO / "16_opencap_dropjump_outputs/metadata/static_peak_error.csv")
d_static = spe["error"].values
bias_static = d_static.mean()
sd_static = d_static.std(ddof=1)
lo_static, hi_static = bias_static - 1.96 * sd_static, bias_static + 1.96 * sd_static
print(f"\nStatic-turnaround same-instant errors (n={len(d_static)}):")
print(f"  bias = {bias_static:+.4f} deg, SD = {sd_static:.4f} deg, 95% LoA [{lo_static:.2f}, {hi_static:.2f}]")
sd_peak2peak = e1.loc[e1.biomarker.str.contains("peak_landing"), "sd_total"].iloc[0]
print(f"  peak-to-peak agreement SD (biomarker #2) = {sd_peak2peak:.4f} deg")
extra_var = sd_static**2 - sd_peak2peak**2
print(f"  Extra same-instant variance = {extra_var:.2f} deg^2 (SD {np.sqrt(extra_var):.2f} deg)")
print("  -> sub-frame timing component even at the turnaround (dynamic, NOT transferred)")

# splits from paper Table 5
splits = {"#2 peak_landing_flexion": 1.00, "#1 contact_flexion": 0.90,
          "#3 landing_rom": 139.7047/150.8291, "#6 loading_rate": 0.10}
nf_table = []
inv_vars = {}
for _, r in e1.iterrows():
    name = r["biomarker"]
    f = splits[name]
    var_proj = r["var_total"] * f if name != "#3 landing_rom" else 139.7047
    var_proj = round(var_proj, 4)
    nf = 1.96 * np.sqrt(var_proj)
    inv_vars[name] = 1.0 / var_proj
    nf_table.append({"biomarker": name, "var_total": round(r["var_total"], 4),
                     "share_static": f if name != "#3 landing_rom" else round(139.7047/150.8291, 4),
                     "var_proj": var_proj, "sigma_proj": round(np.sqrt(var_proj), 4),
                     "NF_1.96sigma": round(nf, 4)})
nf_df = pd.DataFrame(nf_table)
wsum = sum(inv_vars.values())
nf_df["raw_w"] = nf_df["var_proj"].map(lambda v: round(1.0 / v, 5))
nf_df["norm_w_pct"] = nf_df["var_proj"].map(lambda v: round(100 * (1.0 / v) / wsum, 2))
print("\nTable 5 verification (NF = 1.96 x sigma_proj):")
print(nf_df.to_string(index=False))
results["E1"] = {"static_bias": bias_static, "static_sd": sd_static,
                 "static_loa": [lo_static, hi_static], "sd_peak2peak": sd_peak2peak,
                 "extra_var": extra_var, "nf_table": nf_table}

print()
print("=" * 78)
print("E2. TEMPORAL DOWNSAMPLING SENSITIVITY 30->15 Hz (Comment 1)")
print("=" * 78)

def load_traj(df_bio, traj_dir):
    out = []
    for _, row in df_bio.iterrows():
        f = traj_dir / f"{row['video_id']}_rep_{int(row['rep_number']):02d}_smoothed.csv"
        if not f.is_file():
            continue
        t = pd.read_csv(f)
        y = t["knee_angle_smoothed"].interpolate().ffill().bfill().values
        if len(y) < 12:
            continue
        out.append({"subject_id": int(row["subject_id"]), "video_id": row["video_id"],
                    "rep": int(row["rep_number"]), "label": int(row["correctness_label"]),
                    "status": row["phase_identification_status"], "y": y})
    return out

def biomarkers_from(y, fps):
    """Included-angle convention: peak flexion = min, extension = max."""
    peak = float(np.min(y)); ext = float(np.max(y)); rom = ext - peak
    ipk = int(np.argmin(y))
    v = np.diff(y)  # deg/frame
    if ipk > 0:
        v_des = v[:ipk]
        peak_des = float(np.min(v_des)) if len(v_des) else np.nan
        mean_des = float(np.mean(v_des)) if len(v_des) else np.nan
    else:
        peak_des, mean_des = np.nan, np.nan
    return peak, rom, peak_des * fps, mean_des * fps  # velocities in deg/s

sq_bio = pd.read_csv(SQUAT_CSV)
lg_bio = pd.read_csv(LUNGE_CSV)
sq_traj = load_traj(sq_bio, SQUAT_TRAJ)
lg_traj = load_traj(lg_bio, LUNGE_TRAJ)
print(f"Loaded trajectories: squats n={len(sq_traj)}, lunges n={len(lg_traj)}")

def downsample_sensitivity(trajs, name):
    recs = []
    turn_vels = []
    for t in trajs:
        if t["status"] != "ok" or len(t["y"]) < 12:
            continue
        y30 = t["y"]
        y15 = y30[::2]
        p30, r30, pd30, md30 = biomarkers_from(y30, 30.0)
        p15, r15, pd15, md15 = biomarkers_from(y15, 15.0)
        ipk = int(np.argmin(y30))
        lo, hi = max(0, ipk - 1), min(len(y30) - 1, ipk + 1)
        v_turn = np.abs(np.diff(y30[lo:hi + 1])) * 30.0
        if len(v_turn):
            turn_vels.append(np.max(v_turn))
        recs.append({"subject_id": t["subject_id"], "rep": t["rep"], "label": t["label"],
                     "d_peak": abs(p15 - p30), "d_rom": abs(r15 - r30),
                     "d_peakvel": abs(pd15 - pd30), "d_meanvel": abs(md15 - md30),
                     "peak30": p30, "peak15": p15, "rom30": r30, "rom15": r15,
                     "peakvel30": pd30, "peakvel15": pd15, "meanvel30": md30, "meanvel15": md15})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"E2_downsample_{name}.csv", index=False)
    summ = {c: {"median": float(df[c].median()), "p95": float(df[c].quantile(0.95)), "max": float(df[c].max())}
            for c in ["d_peak", "d_rom", "d_peakvel", "d_meanvel"]}
    print(f"\n{name}: |30Hz - 15Hz| change (n={len(df)} reps)")
    for c, s in summ.items():
        print(f"  {c:11s}: median {s['median']:.2f}, p95 {s['p95']:.2f}, max {s['max']:.2f}")
    tv = np.array(turn_vels)
    print(f"  max |ang. vel| within +/-1 frame of turnaround: median {np.median(tv):.2f} deg/s, p95 {np.percentile(tv,95):.2f} deg/s")
    return df, summ, {"turn_med": float(np.median(tv)), "turn_p95": float(np.percentile(tv, 95))}

sq_ds, sq_sum, sq_turn = downsample_sensitivity(sq_traj, "squat")
lg_ds, lg_sum, lg_turn = downsample_sensitivity(lg_traj, "lunge")
results["E2"] = {"squat": sq_sum, "lunge": lg_sum, "turnaround_vel": {"squat": sq_turn, "lunge": lg_turn}}

print()
print("=" * 78)
print("E3. UNCERTAINTY PROPAGATION + THRESHOLD SENSITIVITY (Comment 2)")
print("=" * 78)
SIGMA_MEAS = {"peak": sd_peak2peak,
              "rom": e1.loc[e1.biomarker.str.contains("landing_rom"), "sd_total"].iloc[0],
              "vel": float(np.sqrt(434.6253))}
NF_OPS = {"peak_flexion_deg": 11.9885, "rom_deg": 23.1666,
          "mean_descent_velocity_deg_per_frame": 40.8615}
K = 2

def subject_baseline(df_sub, metric, supervised=True):
    df_sub = df_sub.sort_values("rep_number")
    ok = df_sub[df_sub["phase_identification_status"] == "ok"]
    if supervised:
        base_rows = ok[ok["correctness_label"] == 1].head(K)
    else:
        base_rows = ok.head(K)
    if len(base_rows) < K:
        return None, None
    return float(base_rows[metric].astype(float).mean()), base_rows

def evaluate_rule(df, metric, floor, transform, supervised):
    recs = []
    for sub, g in df.groupby("subject_id"):
        mu, base_rows = subject_baseline(g, metric, supervised)
        if mu is None:
            continue
        base_idx = set(base_rows["rep_number"].tolist())
        for _, r in g.iterrows():
            if r["phase_identification_status"] != "ok" or r["rep_number"] in base_idx:
                continue
            if pd.isna(r[metric]):
                continue
            stat = transform(float(r[metric]), mu)
            recs.append({"subject_id": sub, "rep": int(r["rep_number"]),
                         "label": int(r["correctness_label"]), "stat": stat,
                         "flag": int(stat > floor)})
    return pd.DataFrame(recs)

def depth_stat(x, mu):
    return mu - x

def rom_stat(x, mu):
    return x - mu

def vel_stat(x, mu):
    return abs(x) * 30.0 - abs(mu) * 30.0

def summarise(recs, name):
    if len(recs) == 0:
        return None
    cor = recs[recs.label == 1]; inc = recs[recs.label == 0]
    fpr = cor.flag.mean() if len(cor) else np.nan
    sens = inc.flag.mean() if len(inc) else np.nan
    bal = (fpr + sens) / 2 if not (np.isnan(fpr) or np.isnan(sens)) else np.nan
    return {"rule": name, "n": len(recs), "FPR_correct": round(float(fpr), 4),
            "flag_rate_incorrect": round(float(sens), 4),
            "balanced": round(float(bal), 4),
            "n_correct": len(cor), "n_incorrect": len(inc)}

def null_stats(df, metric, transform, supervised=True):
    recs = evaluate_rule(df, metric, 1e9, transform, supervised)
    cor = recs[recs.label == 1]
    return cor["stat"].values, recs

prop_rows = []
for ex_name, df in [("squat", sq_bio), ("lunge", lg_bio)]:
    for metric, tr, sig in [("peak_flexion_deg", depth_stat, SIGMA_MEAS["peak"]),
                            ("rom_deg", rom_stat, SIGMA_MEAS["rom"]),
                            ("mean_descent_velocity_deg_per_frame", vel_stat, SIGMA_MEAS["vel"])]:
        d_null, recs = null_stats(df, metric, tr, supervised=True)
        if len(d_null) < 5:
            continue
        sd_emp = d_null.std(ddof=1)
        q95_emp = np.percentile(d_null, 95)
        w = []
        for sub, g in df[(df.phase_identification_status == "ok") & (df.correctness_label == 1)].groupby("subject_id"):
            v = g[metric].astype(float).values
            v = v[~pd.isna(v)]
            if metric == "mean_descent_velocity_deg_per_frame":
                v = np.abs(v) * 30.0
            if len(v) >= 3:
                w.append(v.var(ddof=1))
        var_within_video = float(np.median(w)) if w else np.nan
        var_bio = max(var_within_video - sig**2, 0.0)
        nf_op = NF_OPS[metric]
        nf_p1 = 1.96 * sig * np.sqrt(1 + 1.0 / K)
        nf_p2 = 1.96 * np.sqrt((1 + 1.0 / K) * (var_bio + sig**2))
        row = {"exercise": ex_name, "metric": metric, "n_null": len(d_null),
               "emp_SD_D": round(float(sd_emp), 3), "emp_q95_D": round(float(q95_emp), 3),
               "sigma_meas": round(sig, 3),
               "var_within_video": round(var_within_video, 3), "var_bio": round(var_bio, 3),
               "NF_operating": nf_op,
               "NF_prop_meas_only": round(nf_p1, 3), "NF_prop_with_bio": round(nf_p2, 3)}
        for tag, fl in [("op", nf_op), ("p1", nf_p1), ("p2", nf_p2), ("emp", float(q95_emp))]:
            rr = evaluate_rule(df, metric, fl, tr, supervised=True)
            s = summarise(rr, tag)
            row[f"FPR_{tag}"] = s["FPR_correct"] if s else np.nan
            row[f"sens_{tag}"] = s["flag_rate_incorrect"] if s else np.nan
            row[f"bal_{tag}"] = s["balanced"] if s else np.nan
        prop_rows.append(row)
        print(json.dumps(row, default=float))
prop_df = pd.DataFrame(prop_rows)
prop_df.to_csv(OUT / "E3_threshold_sensitivity.csv", index=False)
results["E3"] = prop_rows

print()
print("=" * 78)
print("E4. SUPERVISED vs LABEL-FREE BASELINE (Comment 3)")
print("=" * 78)
lab_rows = []
for ex_name, df in [("squat", sq_bio), ("lunge", lg_bio)]:
    for sup in [True, False]:
        for metric, tr, fl in [("peak_flexion_deg", depth_stat, NF_OPS["peak_flexion_deg"]),
                               ("rom_deg", rom_stat, NF_OPS["rom_deg"]),
                               ("mean_descent_velocity_deg_per_frame", vel_stat, NF_OPS["mean_descent_velocity_deg_per_frame"])]:
            rr = evaluate_rule(df, metric, fl, tr, supervised=sup)
            s = summarise(rr, f"{'supervised' if sup else 'label-free'}|{metric}")
            if s:
                s["exercise"] = ex_name
                lab_rows.append(s)
        allrecs = []
        for sub, g in df.groupby("subject_id"):
            mu_p, br = subject_baseline(g, "peak_flexion_deg", sup)
            mu_r, _ = subject_baseline(g, "rom_deg", sup)
            mu_v, _ = subject_baseline(g, "mean_descent_velocity_deg_per_frame", sup)
            if mu_p is None or mu_r is None or mu_v is None:
                continue
            bidx = set(br["rep_number"].tolist())
            for _, r in g.iterrows():
                if r["phase_identification_status"] != "ok" or r["rep_number"] in bidx or pd.isna(r["peak_flexion_deg"]):
                    continue
                flag = (depth_stat(r["peak_flexion_deg"], mu_p) > NF_OPS["peak_flexion_deg"]) or \
                       (rom_stat(r["rom_deg"], mu_r) > NF_OPS["rom_deg"]) or \
                       (vel_stat(r["mean_descent_velocity_deg_per_frame"], mu_v) > NF_OPS["mean_descent_velocity_deg_per_frame"])
                allrecs.append({"subject_id": sub, "rep": r["rep_number"],
                                "label": int(r["correctness_label"]), "flag": int(flag)})
        s = summarise(pd.DataFrame(allrecs), f"{'supervised' if sup else 'label-free'}|OR-3-rules")
        if s:
            s["exercise"] = ex_name
            lab_rows.append(s)
lab_df = pd.DataFrame(lab_rows)
lab_df.to_csv(OUT / "E4_supervised_vs_labelfree.csv", index=False)
print(lab_df.to_string(index=False))
results["E4"] = lab_rows

print()
print("=" * 78)
print("E5. CONFUSION MATRICES, PER-PARTICIPANT RESULTS, BOOTSTRAP VARIANTS (Comment 5)")
print("=" * 78)

def run_peak_loso(df_bio, exercise):
    d = df_bio if exercise == "squat" else df_bio[df_bio.phase_identification_status == "ok"]
    d = d.dropna(subset=["peak_flexion_deg"]).copy()
    subjects = sorted(d.subject_id.unique())
    pooled = []
    for sub in subjects:
        tr = d[d.subject_id != sub]; te = d[d.subject_id == sub]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(tr[["peak_flexion_deg"]].values)
        Xte = scaler.transform(te[["peak_flexion_deg"]].values)
        clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=42)
        clf.fit(Xtr, tr.correctness_label.values)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob > 0.5).astype(int)
        for lbl, pr in zip(te.correctness_label.values, pred):
            pooled.append({"subject_id": sub, "label": int(lbl), "pred": int(pr)})
    pooled = pd.DataFrame(pooled)
    y, p = pooled.label.values, pooled.pred.values
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    bal = balanced_accuracy_score(y, p)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    per_part = []
    for sub, g in pooled.groupby("subject_id"):
        yy, pp = g.label.values, g.pred.values
        b = balanced_accuracy_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
        tnc, fpc, fnc, tpc = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
        per_part.append({"subject_id": int(sub), "n": len(g), "TP": int(tpc), "FP": int(fpc),
                         "TN": int(tnc), "FN": int(fnc), "bal_acc": None if np.isnan(b) else round(float(b), 4)})
    return pooled, {"exercise": exercise, "n": len(pooled), "TP": int(tp), "FP": int(fp),
                    "TN": int(tn), "FN": int(fn), "sensitivity": round(float(sens), 4),
                    "specificity": round(float(spec), 4),
                    "balanced_accuracy": round(float(bal), 4)}, pd.DataFrame(per_part)

boot_rows = []
for ex_name, df in [("squat", sq_bio), ("lunge", lg_bio)]:
    pooled, glob, per_part = run_peak_loso(df, ex_name)
    print(f"\n{ex_name}: pooled LOSO peak-flexion baseline: {glob}")
    print(per_part.to_string(index=False))
    per_part.to_csv(OUT / f"E5_per_participant_{ex_name}.csv", index=False)
    pooled.to_csv(OUT / f"E5_oof_predictions_{ex_name}.csv", index=False)

    subs = pooled.subject_id.unique()
    n_sub = len(subs)
    blocks = {s: g[["label", "pred"]].values for s, g in pooled.groupby("subject_id")}
    bas, skipped = [], 0
    for b in range(5000):
        pick = rng.choice(subs, size=n_sub, replace=True)
        yy = np.concatenate([blocks[s][:, 0] for s in pick])
        pp = np.concatenate([blocks[s][:, 1] for s in pick])
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        bas.append(balanced_accuracy_score(yy, pp))
    ciA = np.percentile(bas, [2.5, 97.5])
    print(f"  Bootstrap A (OOF-prediction resampling, 5000): mean {np.mean(bas):.4f}, 95% CI [{ciA[0]:.4f}, {ciA[1]:.4f}], skipped {skipped}")

    d = df if ex_name == "squat" else df[df.phase_identification_status == "ok"]
    d = d.dropna(subset=["peak_flexion_deg"]).copy()
    all_subjects = sorted(d.subject_id.unique())

    def loso_on(subjects_sel):
        pools = []
        for sub in subjects_sel:
            tr = d[d.subject_id.isin(subjects_sel) & (d.subject_id != sub)]
            te = d[d.subject_id == sub]
            if len(tr) == 0 or tr.correctness_label.nunique() < 2:
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(tr[["peak_flexion_deg"]].values)
            Xte = sc.transform(te[["peak_flexion_deg"]].values)
            clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=42)
            clf.fit(Xtr, tr.correctness_label.values)
            pr = (clf.predict_proba(Xte)[:, 1] > 0.5).astype(int)
            pools.append(np.column_stack([te.correctness_label.values, pr]))
        if not pools:
            return np.nan
        arr = np.vstack(pools)
        if len(np.unique(arr[:, 0])) < 2:
            return np.nan
        return balanced_accuracy_score(arr[:, 0], arr[:, 1])

    bbs, skippedB = [], 0
    for b in range(2000):
        pick = rng.choice(all_subjects, size=n_sub, replace=True)
        val = loso_on(list(pick))
        if np.isnan(val):
            skippedB += 1
            continue
        bbs.append(val)
    ciB = np.percentile(bbs, [2.5, 97.5])
    print(f"  Bootstrap B (FULL-REFIT LOSO, 2000): mean {np.mean(bbs):.4f}, 95% CI [{ciB[0]:.4f}, {ciB[1]:.4f}], skipped {skippedB}")
    boot_rows.append({"exercise": ex_name, "point": glob["balanced_accuracy"],
                      "ciA_low": ciA[0], "ciA_high": ciA[1], "ciB_low": ciB[0], "ciB_high": ciB[1]})
pd.DataFrame(boot_rows).to_csv(OUT / "E5_bootstrap_variants.csv", index=False)
results["E5"] = boot_rows

print()
print("=" * 78)
print("E6. SUBJECT-IDENTITY ENCODING DIAGNOSTIC (Comment 5c: LSTM explanation)")
print("=" * 78)

def identity_diagnostic(trajs, name):
    reps = [t for t in trajs if t["status"] == "ok" and len(t["y"]) >= 100]
    L = min(len(t["y"]) for t in reps)
    Yraw = np.array([t["y"][:L] for t in reps])
    subs = [t["subject_id"] for t in reps]
    Y_A = Yraw - Yraw[:, [0]]
    rng_b = np.ptp(Yraw, axis=1, keepdims=True); rng_b[rng_b < 1e-5] = 1
    Y_B = (Yraw - Yraw.min(axis=1, keepdims=True)) / rng_b
    out = {}
    usub = sorted(set(subs))
    for tag, Y in [("raw", Yraw), ("SchemeA", Y_A), ("SchemeB", Y_B)]:
        counts = {s: sum(1 for x in subs if x == s) for s in usub}
        cents = {s: Y[[i for i, ss in enumerate(subs) if ss == s]].mean(axis=0) for s in usub}
        correct = 0
        for i in range(len(Y)):
            dists = {}
            for s in usub:
                c_loo = (cents[s] * counts[s] - Y[i]) / max(counts[s] - 1, 1)
                dists[s] = np.linalg.norm(c_loo - Y[i])
            pred = min(dists, key=dists.get)
            correct += int(pred == subs[i])
        gm = Y.mean(axis=0)
        bet = sum(counts[s] * (cents[s] - gm) ** 2 for s in usub).sum()
        tot = ((Y - gm) ** 2).sum()
        out[tag] = {"identity_acc": round(correct / len(Y), 4),
                    "chance": round(1 / len(usub), 4),
                    "between_subject_var_share": round(float(bet / tot), 4)}
        print(f"  {name} {tag}: identity acc {out[tag]['identity_acc']:.3f} "
              f"(chance {out[tag]['chance']:.3f}), between-subject var share {out[tag]['between_subject_var_share']:.3f}")
    dfb = pd.DataFrame([{"subject_id": t["subject_id"], "peak": float(np.min(t["y"]))} for t in reps])
    gm = dfb.peak.mean()
    bet = sum(len(g) * (g.peak.mean() - gm) ** 2 for _, g in dfb.groupby("subject_id"))
    tot = ((dfb.peak - gm) ** 2).sum()
    out["peak_between_share"] = round(float(bet / tot), 4)
    print(f"  {name} peak-flexion between-subject variance share: {out['peak_between_share']:.3f}")
    return out

res6 = {"squat": identity_diagnostic(sq_traj, "squat"),
        "lunge": identity_diagnostic(lg_traj, "lunge")}
results["E6"] = res6

print()
print("=" * 78)
print("E7. EXCLUSION IMPACT & EFFECT-SIZE STABILITY (Comment 4)")
print("=" * 78)
lg = lg_bio.copy()
valid = lg[lg.phase_identification_status == "ok"].copy()
rej = lg[lg.phase_identification_status != "ok"].copy()
vc, vi = int((valid.correctness_label == 1).sum()), int((valid.correctness_label == 0).sum())
rc, ri = int((rej.correctness_label == 1).sum()), int((rej.correctness_label == 0).sum())
print(f"valid {len(valid)} (correct {vc}, incorrect {vi}); rejected {len(rej)} (correct {rc}, incorrect {ri})")
or_stat = stats.fisher_exact([[vc, vi], [rc, ri]])
print(f"Fisher exact OR={or_stat[0]:.4f}, p={or_stat[1]:.4f}")

def cohens_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / s

d_full = cohens_d(valid[valid.correctness_label == 1].peak_flexion_deg.values,
                  valid[valid.correctness_label == 0].peak_flexion_deg.values)
pm042 = lg[lg.video_id == "PM_042"]
sub42 = int(pm042.subject_id.iloc[0]) if len(pm042) else None
v_no42 = valid[valid.subject_id != sub42] if sub42 is not None else valid
d_no42 = cohens_d(v_no42[v_no42.correctness_label == 1].peak_flexion_deg.values,
                  v_no42[v_no42.correctness_label == 0].peak_flexion_deg.values)
print(f"peak-flexion d (61 valid) = {d_full:.4f}; without PM_042 single rep = {d_no42:.4f}")
los_d = []
for sub in sorted(valid.subject_id.unique()):
    vv = valid[valid.subject_id != sub]
    dd = cohens_d(vv[vv.correctness_label == 1].peak_flexion_deg.values,
                  vv[vv.correctness_label == 0].peak_flexion_deg.values)
    los_d.append({"excluded_subject": int(sub), "d": round(float(dd), 4)})
los_df = pd.DataFrame(los_d)
print(los_df.to_string(index=False))
print(f"LOSO-d range: [{los_df.d.min():.3f}, {los_df.d.max():.3f}]")
los_df.to_csv(OUT / "E7_lunge_LOSO_effect_sizes.csv", index=False)
results["E7"] = {"fisher_OR": float(or_stat[0]), "fisher_p": float(or_stat[1]),
                 "d_full": float(d_full), "d_noPM042": float(d_no42),
                 "los_range": [float(los_df.d.min()), float(los_df.d.max())]}

with open(OUT / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nAll experiments complete. Outputs in", OUT)
