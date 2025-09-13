
#!/usr/bin/env python3
"""
Plot & aggregate results from inference_bench_sb.py

Usage:
  python plot_inference_results.py \
    --outdir plots \
    --baseline "Bottleneck" \
    bn_results.json s4a_results.json conf_results.json

What it does:
- Loads multiple results.json files
- Infers a label for each run from meta["adapter"] (fallback to filename stem)
- Concatenates rows into a dataframe
- Computes %-slower vs baseline for each (batch_size, seconds)
- Writes a tidy CSV summary and a baseline-normalized CSV
- Saves charts:
    1) Latency (mean ms) vs batch size (one figure per seconds)
    2) Throughput (items/s) vs batch size (one figure per seconds)
    3) Bar chart of geometric-mean % slower vs baseline across all conditions
"""

import argparse, os, json, math
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt

def _infer_label_and_insertion(path: Path, rec: Dict[str, Any]):
    meta = rec.get("meta", {})
    label = meta.get("adapter") or meta.get("adapter_type")
    insertion = meta.get("insertion")
    if not label:
        proj = meta.get("projection_size")
        label = path.stem
        if proj is not None:
            label = f"{label} (p={proj})"
    label_full = f"{label} [{insertion}]" if insertion else label
    return str(label_full), (insertion or "unknown")

def load_results(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, "r") as f:
            rec = json.load(f)
        label, insertion = _infer_label_and_insertion(p, rec)
        meta = rec.get("meta", {})
        for run in rec.get("runs", []):
            s = run["summary"]
            rows.append({
                "label": label,
            "insertion": insertion,
                "file": str(p),
                "device": rec.get("device"),
                "precision": rec.get("precision"),
                "batch_size": run["batch_size"],
                "seconds": run["seconds"],
                "mean_ms": s["mean_ms"],
                "p50_ms": s["p50_ms"],
                "p90_ms": s["p90_ms"],
                "items_per_s": run.get("throughput_items_per_s"),
                "audio_s_per_s": run.get("throughput_audio_seconds_per_s"),
            })
    return pd.DataFrame(rows)

def save_latency_plots(df: pd.DataFrame, outdir: Path):
    for sec, sub in df.groupby("seconds"):
        fig = plt.figure()
        for label, g in sub.groupby("label"):
            g2 = g.sort_values("batch_size")
            plt.plot(g2["batch_size"], g2["mean_ms"], marker="o", label=label)
        plt.xlabel("Batch size")
        plt.ylabel("Latency (mean ms)")
        plt.title(f"Latency vs Batch Size (seconds={sec})")
        plt.legend()
        fig.savefig(outdir / f"latency_sec{sec}.png", bbox_inches="tight")
        plt.close(fig)

def save_throughput_plots(df: pd.DataFrame, outdir: Path):
    for sec, sub in df.groupby("seconds"):
        fig = plt.figure()
        for label, g in sub.groupby("label"):
            g2 = g.sort_values("batch_size")
            plt.plot(g2["batch_size"], g2["items_per_s"], marker="o", label=label)
        plt.xlabel("Batch size")
        plt.ylabel("Throughput (items/s)")
        plt.title(f"Throughput vs Batch Size (seconds={sec})")
        plt.legend()
        fig.savefig(outdir / f"throughput_sec{sec}.png", bbox_inches="tight")
        plt.close(fig)

def geometric_mean_ratio(series):
    # geometric mean of ratios: exp(mean(log(r)))
    import numpy as np
    s = pd.Series(series).replace([0, None], pd.NA).dropna()
    if len(s) == 0:
        return float("nan")
    arr = s.to_numpy(dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float("nan")
    return float(np.exp(np.log(arr).mean()))

def save_geomean_bar(df: pd.DataFrame, baseline: str, outdir: Path):
    # compute ratio per (seconds,batch) then take geometric mean per label
    labels = sorted(df["label"].unique())
    ratios = {}
    for label in labels:
        if label == baseline: 
            continue
        merged = pd.merge(
            df[df["label"] == label],
            df[df["label"] == baseline],
            on=["seconds","batch_size"],
            suffixes=("_cand","_base")
        )
        # ratio of mean_ms (cand/base)
        merged["ratio"] = merged["mean_ms_cand"] / merged["mean_ms_base"]
        ratios[label] = geometric_mean_ratio(merged["ratio"])

    if not ratios:
        return

    # Convert to %-slower
    perc = {k: (v - 1.0) * 100.0 for k, v in ratios.items() if v == v}
    order = sorted(perc.items(), key=lambda kv: kv[1], reverse=True)
    labels_plot = [k for k,_ in order]
    values = [v for _,v in order]

    fig = plt.figure()
    plt.bar(labels_plot, values)
    plt.ylabel(f"% slower vs {baseline} (geometric mean across all settings)")
    plt.title("Overall normalized latency")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(outdir / "geomean_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--baseline", type=str, default=None, help="adapter label to normalize against (e.g., 'Bottleneck')")
    ap.add_argument("jsons", nargs="+", help="results.json files from inference_bench_sb.py")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = [Path(j) for j in args.jsons]
    df = load_results(paths)
    df = df.sort_values(["seconds","batch_size","label"])

    # Save tidy CSV
    csv_path = outdir / "summary.csv"
    df.to_csv(csv_path, index=False)

    # Plots
    save_latency_plots(df, outdir)
    save_throughput_plots(df, outdir)

    if args.baseline:
        # Save normalized CSV per setting (percent slower vs baseline)
        base = df[df["label"] == args.baseline]
        merged = pd.merge(df, base, on=["seconds","batch_size"], suffixes=("","_base"))
        merged["pct_slower_vs_baseline"] = (merged["mean_ms"] / merged["mean_ms_base"] - 1.0) * 100.0
        merged.to_csv(outdir / "summary_vs_baseline.csv", index=False)
        save_geomean_bar(df, args.baseline, outdir)

    print(f"Wrote CSV and plots to {outdir}")

if __name__ == "__main__":
    main()
