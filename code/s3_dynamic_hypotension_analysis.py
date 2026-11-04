#!/usr/bin/env python3
"""S3 analysis for the VitalDB dynamic hypotension trajectory manuscript.

The script is intentionally dependency-light: pandas/numpy plus the Python
standard library. It reuses the previously verified VitalDB MAP cache and
selected-MAP table, then writes all new outputs into this manuscript run.
"""

from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vitaldb_open"
OLD_RESULTS = ROOT / ".workflow" / "academic-research-suite" / "vitaldb-bja" / "results"
RUN = ROOT / ".workflow" / "academic-research-suite" / "vitaldb-ja-dynamic-hypotension"
RESULTS = RUN / "results"
FIGURES = RESULTS / "figures"
CACHE = DATA / "map_tracks"

PHYS_MIN = 20.0
PHYS_MAX = 200.0
MAX_INTERVAL_SECONDS = 300.0
MIN_COVERAGE = 0.70
MIN_OP_DURATION_MIN = 30.0
PRIMARY_THRESHOLD = 65
SEVERE_THRESHOLDS = [55, 60]
RNG_SEED = 20260703


@dataclass(frozen=True)
class ModelSpec:
    name: str
    columns: list[str]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def read_track(path: Path) -> pd.DataFrame:
    rows: list[tuple[float, float]] = []
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "Time":
            return pd.DataFrame(columns=["time", "map"])
        for row in reader:
            if len(row) < 2:
                continue
            try:
                t = float(row[0])
                v = float(row[1])
            except ValueError:
                continue
            if PHYS_MIN <= v <= PHYS_MAX:
                rows.append((t, v))
    if not rows:
        return pd.DataFrame(columns=["time", "map"])
    out = pd.DataFrame(rows, columns=["time", "map"])
    return out.sort_values("time").reset_index(drop=True)


def window_track(track: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    if track.empty or end <= start:
        return track.iloc[0:0].copy()
    return track[(track["time"] >= start) & (track["time"] <= end)].copy().reset_index(drop=True)


def weighted_time_arrays(track: pd.DataFrame, end: float) -> tuple[np.ndarray, np.ndarray]:
    times = track["time"].to_numpy(dtype=float)
    values = track["map"].to_numpy(dtype=float)
    next_times = np.r_[times[1:], end]
    intervals = np.clip(next_times - times, 0, MAX_INTERVAL_SECONDS)
    valid = intervals > 0
    return values[valid], intervals[valid]


def episode_features(values: np.ndarray, intervals: np.ndarray, threshold: float) -> dict[str, float]:
    below = values < threshold
    count = 0
    longest = 0.0
    current = 0.0
    for is_below, interval in zip(below, intervals):
        if is_below:
            if current == 0:
                count += 1
            current += float(interval)
            longest = max(longest, current)
        else:
            current = 0.0
    return {
        f"episode_count_lt_{int(threshold)}": float(count),
        f"longest_episode_min_lt_{int(threshold)}": longest / 60.0,
    }


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(values.mean())


def trajectory_features(case: pd.Series, track: pd.DataFrame) -> dict[str, float | int | str]:
    start = float(case["opstart"])
    end = float(case["opend"])
    duration_seconds = end - start
    base: dict[str, float | int | str] = {
        "caseid": int(case["caseid"]),
        "duration_min": duration_seconds / 60.0 if duration_seconds > 0 else np.nan,
        "valid_bp_points": 0,
        "valid_bp_seconds": 0.0,
        "bp_coverage": 0.0,
    }
    tw = window_track(track, start, end)
    if tw.empty or duration_seconds <= 0:
        return base

    values, intervals = weighted_time_arrays(tw, end)
    if len(values) == 0:
        return base

    valid_seconds = float(intervals.sum())
    weights = intervals / valid_seconds
    base.update(
        {
            "valid_bp_points": int(len(values)),
            "valid_bp_seconds": valid_seconds,
            "bp_coverage": valid_seconds / duration_seconds,
            "mean_map": float(np.sum(values * weights)),
            "min_map": float(np.min(values)),
            "sd_map": float(np.sqrt(np.sum(((values - np.sum(values * weights)) ** 2) * weights))),
            "cv_map": float(np.std(values) / np.mean(values)) if np.mean(values) else np.nan,
        }
    )

    q_means: list[float] = []
    rel_time = (tw["time"].to_numpy(dtype=float) - start) / duration_seconds
    raw_values = tw["map"].to_numpy(dtype=float)
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        mask = (rel_time >= lo) & (rel_time <= hi if i == 9 else rel_time < hi)
        q_means.append(float(np.mean(raw_values[mask])) if np.any(mask) else np.nan)
    for i, value in enumerate(q_means, start=1):
        base[f"map_decile_{i}"] = value

    q = np.asarray(q_means, dtype=float)
    if np.isfinite(q).sum() >= 6:
        x = np.linspace(0, 1, len(q))
        mask = np.isfinite(q)
        base["slope_norm"] = float(np.polyfit(x[mask], q[mask], 1)[0])
        base["early_map"] = safe_nanmean(q[:3])
        base["middle_map"] = safe_nanmean(q[3:7])
        base["late_map"] = safe_nanmean(q[7:])
        base["recovery_from_nadir"] = safe_nanmean(q[7:]) - float(np.nanmin(q))
        base["early_to_late_change"] = safe_nanmean(q[7:]) - safe_nanmean(q[:3])

    for threshold in [PRIMARY_THRESHOLD] + SEVERE_THRESHOLDS:
        deficits = np.maximum(0.0, threshold - values)
        below = values < threshold
        below_seconds = float(intervals[below].sum())
        aut = float(np.sum(deficits * intervals) / 60.0)
        base[f"any_map_lt_{threshold}"] = int(below_seconds > 0)
        base[f"minutes_map_lt_{threshold}"] = below_seconds / 60.0
        base[f"pct_time_lt_{threshold}"] = below_seconds / valid_seconds if valid_seconds else np.nan
        base[f"aut_map_lt_{threshold}"] = aut
        base[f"twa_map_lt_{threshold}"] = aut / (valid_seconds / 60.0) if valid_seconds else np.nan
        base[f"auc_per_hour_lt_{threshold}"] = aut / (valid_seconds / 3600.0) if valid_seconds else np.nan
        base.update(episode_features(values, intervals, threshold))

    return base


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(DATA / "cases.csv")
    cases = cases.rename(columns={cases.columns[0]: "caseid"})
    selected = pd.read_csv(OLD_RESULTS / "s3_selected_map_tracks.csv")
    return cases, selected


def build_feature_table(cases: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_map = selected.set_index("caseid")
    rows: list[dict[str, float | int | str]] = []
    eligible_cases = cases[cases["caseid"].isin(selected_map.index)].copy()
    for idx, case in eligible_cases.iterrows():
        caseid = int(case["caseid"])
        meta = selected_map.loc[caseid]
        path = CACHE / f"{meta['tid']}.csv.gz"
        row: dict[str, float | int | str] = {
            "caseid": caseid,
            "map_tname": str(meta["tname"]),
            "map_source_type": str(meta["source_type"]),
            "track_exists": int(path.exists()),
        }
        if path.exists():
            row.update(trajectory_features(case, read_track(path)))
        rows.append(row)
        if (idx + 1) % 500 == 0:
            print(f"feature_progress {idx + 1}/{len(eligible_cases)}", flush=True)
    return pd.DataFrame(rows)


def prepare_analysis_frame(cases: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    df = cases.merge(features, on="caseid", how="inner")
    df["icu_use"] = (pd.to_numeric(df["icu_days"], errors="coerce") > 0).astype(int)
    df["op_duration_min"] = (pd.to_numeric(df["opend"], errors="coerce") - pd.to_numeric(df["opstart"], errors="coerce")) / 60.0
    df["male"] = (df["sex"] == "M").astype(int)
    df["general_ane"] = (df["ane_type"] == "General").astype(int)
    df["source_invasive"] = (df["map_source_type"] == "invasive").astype(int)
    dept = pd.get_dummies(df["department"], prefix="dept", dummy_na=False)
    if "dept_General surgery" in dept.columns:
        dept = dept.drop(columns=["dept_General surgery"])
    return pd.concat([df, dept], axis=1)


def analysis_cohort(df: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    out = df[
        (df["general_ane"] == 1)
        & (df["bp_coverage"] >= MIN_COVERAGE)
        & (df["op_duration_min"] >= MIN_OP_DURATION_MIN)
    ].copy()
    if source:
        out = out[out["map_source_type"] == source].copy()
    return out.reset_index(drop=True)


def standardize_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    X = np.where(np.isfinite(X), X, mu)
    return (X - mu) / sd, mu, sd


def kmeans(Z: np.ndarray, k: int, seed: int = RNG_SEED, iterations: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k)
    centers = Z[rng.choice(len(Z), size=k, replace=False)].copy()
    for _ in range(iterations):
        dist = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new_centers = np.vstack(
            [Z[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    dist = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1), centers


def centroid_silhouette(Z: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    own = np.sqrt(((Z - centers[labels]) ** 2).sum(axis=1))
    dist = np.sqrt(((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    dist[np.arange(len(Z)), labels] = np.inf
    other = dist.min(axis=1)
    return float(np.mean((other - own) / np.maximum(other, own)))


def assign_phenotype_names(profile: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    severe = profile["auc_per_hour_lt_65"].idxmax()
    high = profile["mean_map"].idxmax()
    recovery = profile["recovery_from_nadir"].idxmax()
    used = set()
    names[int(severe)] = "sustained_severe_hypotension_burden"
    used.add(int(severe))
    if int(high) not in used:
        names[int(high)] = "preserved_high_map"
        used.add(int(high))
    if int(recovery) not in used:
        names[int(recovery)] = "early_low_map_with_recovery"
        used.add(int(recovery))
    for cluster in profile.index:
        if int(cluster) not in names:
            names[int(cluster)] = "moderate_low_burden_or_variable_map"
    return names


def derive_phenotypes(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "mean_map",
        "min_map",
        "sd_map",
        "auc_per_hour_lt_65",
        "pct_time_lt_65",
        "early_map",
        "middle_map",
        "late_map",
        "slope_norm",
        "recovery_from_nadir",
        "episode_count_lt_65",
        "longest_episode_min_lt_65",
    ]
    complete = cohort.dropna(subset=feature_cols).copy().reset_index(drop=True)
    Z, _, _ = standardize_matrix(complete, feature_cols)

    selection_rows = []
    for k in [3, 4, 5]:
        labels, centers = kmeans(Z, k)
        selection_rows.append(
            {
                "k": k,
                "centroid_silhouette_approx": centroid_silhouette(Z, labels, centers),
                "min_cluster_size": int(np.bincount(labels).min()),
                "max_cluster_size": int(np.bincount(labels).max()),
                "cluster_sizes": ";".join(map(str, np.bincount(labels).tolist())),
            }
        )
    selection = pd.DataFrame(selection_rows)

    labels, _ = kmeans(Z, 4)
    complete["trajectory_cluster"] = labels
    profile = complete.groupby("trajectory_cluster")[feature_cols + ["icu_use"]].mean()
    names = assign_phenotype_names(profile)
    complete["trajectory_phenotype"] = complete["trajectory_cluster"].map(names)
    profile = complete.groupby(["trajectory_cluster", "trajectory_phenotype"])[
        feature_cols + ["icu_use"]
    ].agg(["mean", "std", "count"])
    profile.columns = ["_".join(col).strip() for col in profile.columns.to_flat_index()]
    profile = profile.reset_index()
    return complete, selection, profile


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = y_true[mask].astype(int)
    s = score[mask].astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    unique, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / counts)[inv]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.1,
    steps: int = 4000,
    lr: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xs = (np.where(np.isfinite(X), X, mu) - mu) / sd
    Xd = np.c_[np.ones(len(Xs)), Xs]
    beta = np.zeros(Xd.shape[1])
    for _ in range(steps):
        z = np.clip(Xd @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xd.T @ (p - y) / len(y)
        grad[1:] += lam * beta[1:] / len(y)
        beta -= lr * grad
    return beta, mu, sd


def predict_logistic(X: np.ndarray, beta: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xs = (np.where(np.isfinite(X), X, mu) - mu) / sd
    Xd = np.c_[np.ones(len(Xs)), Xs]
    return 1.0 / (1.0 + np.exp(-np.clip(Xd @ beta, -30, 30)))


def bootstrap_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def base_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "age",
        "male",
        "bmi",
        "asa",
        "emop",
        "op_duration_min",
        "preop_hb",
        "preop_cr",
        "preop_alb",
        "source_invasive",
    ]
    cols.extend([c for c in df.columns if c.startswith("dept_")])
    return cols


def model_specs(df: pd.DataFrame) -> list[ModelSpec]:
    base = base_columns(df)
    phenotype_cols = [c for c in df.columns if c.startswith("phenotype_")]
    return [
        ModelSpec("base_covariates", base),
        ModelSpec("base_plus_traditional_ioh", base + ["minutes_map_lt_65", "twa_map_lt_65"]),
        ModelSpec("base_plus_trajectory_phenotype", base + phenotype_cols),
        ModelSpec("base_plus_ioh_and_trajectory", base + ["minutes_map_lt_65", "twa_map_lt_65"] + phenotype_cols),
    ]


def model_comparison(phenotype_frame: pd.DataFrame, source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = phenotype_frame.copy()
    if source:
        df = df[df["map_source_type"] == source].copy()
    dummies = pd.get_dummies(df["trajectory_phenotype"], prefix="phenotype", drop_first=True)
    df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    y = df["icu_use"].to_numpy(dtype=int)
    rows = []
    predictions = pd.DataFrame({"caseid": df["caseid"].to_numpy(), "icu_use": y, "source": source or "all"})
    base_auc = None
    for spec in model_specs(df):
        beta, mu, sd = fit_logistic_coefficients(df[spec.columns].to_numpy(dtype=float), y)
        pred = predict_logistic(df[spec.columns].to_numpy(dtype=float), beta, mu, sd)
        auc = auc_score(y, pred)
        if base_auc is None:
            base_auc = auc
        rows.append(
            {
                "source": source or "all",
                "model": spec.name,
                "n": int(len(df)),
                "events": int(y.sum()),
                "auc": auc,
                "delta_auc_vs_base": auc - base_auc,
                "brier": float(np.mean((pred - y) ** 2)),
                "observed_event_rate": float(np.mean(y)),
                "predicted_event_rate": float(np.mean(pred)),
            }
        )
        predictions[spec.name] = pred
    return pd.DataFrame(rows), predictions


def adjusted_phenotype_or(phenotype_frame: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    df = phenotype_frame.copy().reset_index(drop=True)
    reference = "preserved_high_map"
    if reference not in set(df["trajectory_phenotype"]):
        reference = sorted(df["trajectory_phenotype"].unique())[0]
    phenotype_levels = [p for p in sorted(df["trajectory_phenotype"].unique()) if p != reference]
    dummies = pd.DataFrame(
        {
            f"phenotype_{level}": (df["trajectory_phenotype"] == level).astype(int)
            for level in phenotype_levels
        }
    )
    model_df = pd.concat([df, dummies], axis=1)
    phenotype_cols = list(dummies.columns)
    cols = base_columns(model_df) + phenotype_cols
    y = model_df["icu_use"].to_numpy(dtype=int)
    beta, _, _ = fit_logistic_coefficients(model_df[cols].to_numpy(dtype=float), y)
    coef = dict(zip(["intercept"] + cols, beta))
    rng = np.random.default_rng(RNG_SEED)
    boot: dict[str, list[float]] = {col: [] for col in phenotype_cols}
    for _ in range(n_boot):
        idx = rng.integers(0, len(model_df), len(model_df))
        part = model_df.iloc[idx]
        if part["icu_use"].nunique() < 2:
            continue
        b, _, _ = fit_logistic_coefficients(part[cols].to_numpy(dtype=float), part["icu_use"].to_numpy(dtype=int), steps=1800)
        for col, value in zip(cols, b[1:]):
            if col in boot:
                boot[col].append(float(np.exp(value)))
    rows = []
    rows.append({"term": "reference", "phenotype": reference, "adjusted_or": 1.0, "ci_low": np.nan, "ci_high": np.nan})
    for col in phenotype_cols:
        phenotype = col.replace("phenotype_", "")
        or_value = float(np.exp(coef[col]))
        low, high = bootstrap_ci(boot[col])
        rows.append({"term": col, "phenotype": phenotype, "adjusted_or": or_value, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def calibration_deciles(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    df = predictions[["icu_use", model]].rename(columns={model: "pred"}).sort_values("pred").reset_index(drop=True)
    df["bin"] = pd.qcut(df.index, q=10, labels=False, duplicates="drop") + 1
    rows = []
    for bin_id, part in df.groupby("bin", sort=True):
        rows.append(
            {
                "model": model,
                "bin": int(bin_id),
                "n": int(len(part)),
                "predicted_mean": float(part["pred"].mean()),
                "observed_rate": float(part["icu_use"].mean()),
                "events": int(part["icu_use"].sum()),
            }
        )
    return pd.DataFrame(rows)


def cohort_flow(cases: pd.DataFrame, features: pd.DataFrame, frame: pd.DataFrame, phenotype_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": "VitalDB cases", "n": int(len(cases))},
            {"step": "Cases with selected MAP track", "n": int(features["track_exists"].sum())},
            {"step": "General anesthesia, MAP coverage >=70%, operation >=30 min", "n": int(len(frame))},
            {"step": "Complete trajectory feature phenotype cohort", "n": int(len(phenotype_frame))},
            {"step": "ICU use events in phenotype cohort", "n": int(phenotype_frame["icu_use"].sum())},
        ]
    )


def write_svg_trajectory(profile_frame: pd.DataFrame, out_path: Path) -> None:
    decile_cols = [f"map_decile_{i}" for i in range(1, 11)]
    prof = profile_frame.groupby("trajectory_phenotype")[decile_cols].mean()
    width, height = 780, 460
    left, top, right, bottom = 70, 40, 30, 60
    vals = prof.to_numpy(dtype=float)
    ymin = max(40, math.floor(np.nanmin(vals) / 10) * 10)
    ymax = min(130, math.ceil(np.nanmax(vals) / 10) * 10)
    colors = ["#2A6F97", "#B23A48", "#2F7D32", "#7C4D9E", "#C47F00"]

    def xscale(i: int) -> float:
        return left + i * (width - left - right) / 9

    def yscale(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / (ymax - ymin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="700">Mean MAP trajectory by phenotype</text>',
    ]
    for y in range(int(ymin), int(ymax) + 1, 10):
        yy = yscale(y)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y}</text>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#111827"/>')
    for i in range(10):
        xx = xscale(i)
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="11">{i+1}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="12">Normalized operative time decile</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">MAP, mmHg</text>')

    for idx, (name, row) in enumerate(prof.iterrows()):
        points = " ".join(f"{xscale(i):.1f},{yscale(float(row[col])):.1f}" for i, col in enumerate(decile_cols))
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{xscale(9):.1f}" cy="{yscale(float(row[decile_cols[-1]])):.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{width-right-180}" y="{52+idx*22}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def write_svg_model_performance(model_table: pd.DataFrame, out_path: Path) -> None:
    df = model_table[model_table["source"].eq("all")].copy()
    width, height = 760, 420
    left, top, right, bottom = 230, 40, 40, 60
    xmin = max(0.5, math.floor(df["auc"].min() * 100) / 100 - 0.01)
    xmax = min(1.0, math.ceil(df["auc"].max() * 100) / 100 + 0.01)

    def xscale(v: float) -> float:
        return left + (v - xmin) * (width - left - right) / (xmax - xmin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="700">ICU risk model discrimination</text>',
    ]
    for tick in np.linspace(xmin, xmax, 5):
        xx = xscale(float(tick))
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.2f}</text>')
    for i, row in enumerate(df.itertuples(index=False)):
        y = top + 38 + i * 72
        x = xscale(float(row.auc))
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="12">{row.model}</text>')
        parts.append(f'<line x1="{left}" x2="{x:.1f}" y1="{y}" y2="{y}" stroke="#93c5fd" stroke-width="12"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#1d4ed8"/>')
        parts.append(f'<text x="{x+10:.1f}" y="{y+5}" font-family="Arial" font-size="12">AUROC {row.auc:.3f}</text>')
    parts.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="12">AUROC</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main() -> None:
    ensure_dirs()
    cases, selected = load_inputs()
    features = build_feature_table(cases, selected)
    features.to_csv(RESULTS / "s3_trajectory_features.csv", index=False)

    frame = prepare_analysis_frame(cases, features)
    cohort = analysis_cohort(frame)
    phenotype_frame, cluster_selection, cluster_profile = derive_phenotypes(cohort)

    cohort.to_csv(RESULTS / "s3_analysis_cohort.csv", index=False)
    phenotype_frame.to_csv(RESULTS / "s3_phenotype_frame.csv", index=False)
    cluster_selection.to_csv(RESULTS / "s3_cluster_selection.csv", index=False)
    cluster_profile.to_csv(RESULTS / "s3_cluster_profile.csv", index=False)
    cohort_flow(cases, features, cohort, phenotype_frame).to_csv(RESULTS / "s3_cohort_flow.csv", index=False)

    model_tables = []
    prediction_tables = []
    for source in [None, "invasive", "nibp"]:
        table, preds = model_comparison(phenotype_frame, source=source)
        model_tables.append(table)
        prediction_tables.append(preds)
    model_table = pd.concat(model_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    model_table.to_csv(RESULTS / "s3_model_comparison_icu.csv", index=False)
    predictions.to_csv(RESULTS / "s3_model_predictions_icu.csv", index=False)

    adjusted_phenotype_or(phenotype_frame).to_csv(RESULTS / "s3_adjusted_phenotype_or.csv", index=False)
    calibration_deciles(prediction_tables[0], "base_plus_ioh_and_trajectory").to_csv(
        RESULTS / "s3_calibration_deciles_icu.csv", index=False
    )

    write_svg_trajectory(phenotype_frame, FIGURES / "Figure_2_map_trajectory_phenotypes.svg")
    write_svg_model_performance(model_table, FIGURES / "Figure_5_model_performance.svg")

    summary = {
        "source_cases": int(len(cases)),
        "tracks_with_selected_map": int(features["track_exists"].sum()),
        "analysis_cohort_n": int(len(cohort)),
        "phenotype_cohort_n": int(len(phenotype_frame)),
        "icu_events": int(phenotype_frame["icu_use"].sum()),
        "output_directory": str(RESULTS),
    }
    pd.DataFrame([summary]).to_csv(RESULTS / "s3_execution_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
