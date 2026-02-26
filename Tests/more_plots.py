#!/usr/bin/env python3
"""
Multi-method training report analyzer (config-driven; small CLI only for backbone + min/max toggle).

Generates, for EACH method:
  1) train_acc_top1.png
  2) train_loss.png
  3) test_acc_top1.png
  4) test_loss.png
  5) all_four_metrics_grid.png
  6) params_vs_accuracy_*_classifier_params.png
  7) params_vs_accuracy_*_total_params.png
  + CSVs (parsed runs + per-epoch stats + summary)

Generates, ACROSS methods:
  8) all_methods_comparison_grid.png
  9) all_methods_params_vs_accuracy_classifier_params.png   (RAW scatter: all runs)
 10) all_methods_params_vs_accuracy_total_params.png        (RAW scatter: all runs)

Behavior notes
--------------
- Std band is always mean ± std, but CLIPPED to observed [min, max] at each epoch.
- Per-method 4-grid writes std annotations (σ) if ANNOTATE_STD_GRID=True.
- Cross-method comparison grid has NO std annotations.
- If highlight_report is provided, it is treated as an EXTRA RUN/EXPERIMENT (appended once if unique),
  not as a special replacement curve.
- Min/Max dashed lines are controlled ONLY by --show-minmax flag (not by highlight presence).
- Top1 is auto-normalized to percent if a report appears to be in [0,1] scale.
- Duplicate epoch entries inside one report are resolved by keeping the LAST occurrence.

Usage
-----
# EfficientNet (default), hide min/max:
python more_plots.py

# EfficientNet, show min/max:
python more_plots.py --show-minmax

# ResNet, show min/max:
python more_plots.py --backbone resnet --show-minmax
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CLI (only backbone selection + show-minmax)
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone",
        choices=["efficientnet", "resnet"],
        default="efficientnet",
        help="Choose which predefined config set to use (no more comment/uncomment).",
    )
    p.add_argument(
        "--show-minmax",
        action="store_true",
        default=True,
        help="If set, draw min/max dashed lines on per-method plots/grids (independent of highlight).",
    )
    return p.parse_args()


# ============================================================
# CONFIG (EDIT IF NEEDED)
# ============================================================

ANNOTATE_STD_INDIVIDUAL = True
ANNOTATE_STD_GRID = True   # write σ values in each method's all-4 grid

PARAM_ACCURACY_MODE = "final"  # "final" or "best"
PARAM_X_FIELDS = ["classifier_params", "total_params"]

# Per-method param plot: if False -> raw runs only (recommended)
PER_METHOD_PARAM_AGGREGATE = False

# Parsing / normalization behavior
AUTO_NORMALIZE_TOP1_TO_PERCENT = True
TOP1_NORMALIZE_THRESHOLD = 1.5  # if max(top1) <= this, treat as [0,1] and multiply by 100
DUPLICATE_EPOCH_KEEP = "last"   # "last" or "first"

# Path anchors (robust to CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Marker/color style for combined raw scatter (all methods in one image)
METHOD_SCATTER_STYLE = {
    "TRL": {"marker": "*", "color": "red", "size": 150},
    "HLGTL": {"marker": "o", "color": "green", "size": 70},   # circle
    "BASE": {"marker": "D", "color": "orange", "size": 75},
}


# -------------------------
# ResNet50 config
# -------------------------
BASE_OUTPUT_DIR_RESNET = Path("./analysis_outputs_resnet50")
COMBINED_PARAM_SCATTER_TITLE_RESNET = "Resnet50 TINY"

METHODS_RESNET = [
    {
        "name": "BASE",
        "report_patterns": [
            r"results/TEST_ID0025_SEED_exp_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0025_SEED_exp_*/model_stats/model_info.txt",
        ],
        "highlight_report": r"results/Test_ID025_EXP/accuracy_stats/report.txt",
    },
    {
        "name": "TRL",
        "report_patterns": [
            r"results/TEST_ID0028_SEED_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0028_SEED_*/model_stats/model_info.txt",
        ],
        "highlight_report": r"results/TEST_ID028/accuracy_stats/report.txt",
    },
    {
        "name": "HLGTL",
        "report_patterns": [
            r"results/TEST_ID0032_SEED_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0032_SEED_*/model_stats/model_info.txt",
        ],
        "highlight_report": r"results/TEST_ID032/accuracy_stats/report.txt",
    },
]

# -------------------------
# EfficientNet config
# -------------------------
BASE_OUTPUT_DIR_EFFICIENTNET = Path("./analysis_outputs_efficientnet")
COMBINED_PARAM_SCATTER_TITLE_EFFICIENTNET = "EfficientNet TINY"

METHODS_EFFICIENTNET = [
    {
        "name": "BASE",
        "report_patterns": [
            r"results/TEST_ID0025_efficient_net_better_SEED_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0025_efficient_net_better_SEED_*/model_stats/model_info.txt",
        ],
        # "highlight_report": r"results/Test_ID001_EfficientNetB4_25_base_EXP/accuracy_stats/report.txt",
    },
    {
        "name": "TRL",
        "report_patterns": [
            r"results/TEST_ID0028_efficient_net_better_SEED_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0028_efficient_net_better_SEED_*/model_stats/model_info.txt",
        ],
        # "highlight_report": r"results/Test_ID002_EfficientNetB4_28_tensorly/accuracy_stats/report.txt",
    },
    {
        "name": "HLGTL",
        "report_patterns": [
            r"results/TEST_ID0032_efficient_net_6_better_SEED_*/accuracy_stats/report.txt",
        ],
        "param_patterns": [
            r"results/TEST_ID0032_efficient_net_6_better_SEED_*/model_stats/model_info.txt",
        ],
        # "highlight_report": r"results/Test_ID006_EfficientNetB4_32_ourmethod_EXP/accuracy_stats/report.txt",
    },
]


# ============================================================
# Regex Parsing
# ============================================================

REPORT_LINE_RE = re.compile(
    r"^(Train|Test)\s+epoch\s+(\d+):[^\n\r]*?"
    r"top1=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%"
    r"[^\n\r]*?"
    r"loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)

SEED_RE = re.compile(r"Seed:\s*(\d+)", re.IGNORECASE)
TOTAL_PARAMS_RE = re.compile(r"total number of parameters:\s*([0-9]+)", re.IGNORECASE)
CLS_PARAMS_RE = re.compile(r"total number of classifier parameters:\s*([0-9]+)", re.IGNORECASE)


# ============================================================
# Utilities
# ============================================================

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "method"


def fmt_num(x: float) -> str:
    x = float(x)
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.5f}"


def natural_sort_key_path(p: Path):
    parts = re.split(r"(\d+)", str(p))
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def expand_paths(patterns: Iterable[str], base_dir: Path = REPO_ROOT) -> List[Path]:
    """Accept direct paths and/or glob patterns. Returns sorted unique existing files."""
    found: List[str] = []
    for p in patterns:
        p_str = str(p)
        if not Path(p_str).is_absolute():
            p_glob = str((base_dir / p_str).resolve())
        else:
            p_glob = p_str

        matches = glob.glob(p_glob, recursive=True)
        if matches:
            found.extend(matches)
        else:
            found.append(p_glob)

    out: List[Path] = []
    seen = set()
    for s in found:
        path = Path(s).expanduser().resolve()
        if path.exists() and path.is_file():
            sp = str(path)
            if sp not in seen:
                seen.add(sp)
                out.append(path)

    return sorted(out, key=natural_sort_key_path)


def append_unique_path(paths: List[Path], extra_path: Optional[Path]) -> List[Path]:
    """Append extra_path only if it exists and is not already in the list."""
    if extra_path is None:
        return list(paths)

    extra_resolved = extra_path.resolve()
    existing = {p.resolve() for p in paths}
    if extra_resolved in existing:
        return list(paths)
    return list(paths) + [extra_resolved]


def run_root_from_report_path(report_path: Path) -> Path:
    # .../RUN/accuracy_stats/report.txt -> RUN
    return report_path.parent.parent.resolve()


def run_root_from_meta_path(meta_path: Path) -> Path:
    # .../RUN/model_stats/model_info.txt -> RUN
    return meta_path.parent.parent.resolve()


def clipped_std_band(stats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean±std band, clipped to observed [min, max] at each epoch.
    This prevents impossible values (e.g., negative accuracy) from being shaded.
    """
    mean = stats["mean"].to_numpy(dtype=float)
    std = stats["std"].to_numpy(dtype=float)
    mn = stats["min"].to_numpy(dtype=float)
    mx = stats["max"].to_numpy(dtype=float)

    lower = mean - std
    upper = mean + std

    # Trim only exceeding parts
    lower = np.maximum(lower, mn)
    upper = np.minimum(upper, mx)

    # Safety
    upper = np.maximum(upper, lower)
    return lower, upper


# ============================================================
# Report Parsing
# ============================================================

def _auto_normalize_top1_percent(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    if df.empty or "top1" not in df.columns or not AUTO_NORMALIZE_TOP1_TO_PERCENT:
        return df

    top1_max = float(df["top1"].max())
    if np.isfinite(top1_max) and top1_max <= TOP1_NORMALIZE_THRESHOLD:
        df = df.copy()
        df["top1"] = df["top1"] * 100.0
        print(f"[INFO] Auto-normalized top1 to percent for: {report_path.name} (max={top1_max:.4f})")
    return df


def _dedupe_epochs_keep_last(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    if df.empty:
        return df

    dup_rows = int(df.duplicated(subset=["split", "epoch"], keep=False).sum())
    if dup_rows > 0:
        print(
            f"[INFO] Duplicate (split, epoch) rows detected in {report_path.name}: "
            f"{dup_rows} rows -> keeping {DUPLICATE_EPOCH_KEEP}"
        )

    keep_arg = "last" if DUPLICATE_EPOCH_KEEP.lower() == "last" else "first"
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    df = (
        df.sort_values("_row_order")
        .drop_duplicates(subset=["split", "epoch"], keep=keep_arg)
        .drop(columns=["_row_order"])
        .sort_values(["split", "epoch"])
        .reset_index(drop=True)
    )
    return df


def parse_report(report_path: Path) -> pd.DataFrame:
    """
    Parse one report.txt into rows: split, epoch, top1, loss.
    Supports lines with top2/top3/top4/top5/time, etc.
    """
    txt = report_path.read_text(encoding="utf-8", errors="ignore")
    rows = []

    for m in REPORT_LINE_RE.finditer(txt):
        split, epoch, top1, loss = m.groups()
        rows.append(
            {
                "split": split,
                "epoch": int(epoch),
                "top1": float(top1),
                "loss": float(loss),
            }
        )

    if not rows:
        preview = "\n".join(txt.splitlines()[:8])
        raise ValueError(
            f"No valid train/test epoch lines found in report: {report_path}\n"
            f"First lines preview:\n{preview}"
        )

    df = pd.DataFrame(rows)
    df = _auto_normalize_top1_percent(df, report_path)
    df = _dedupe_epochs_keep_last(df, report_path)
    return df


def load_reports(report_paths: List[Path]) -> pd.DataFrame:
    """Load multiple report files into one DataFrame with run metadata."""
    frames = []
    for idx, rp in enumerate(report_paths, start=1):
        df = parse_report(rp)
        run_name = run_root_from_report_path(rp).name or f"run_{idx}"
        df["run"] = idx
        df["run_name"] = run_name
        df["report_path"] = str(rp)
        frames.append(df)

    if not frames:
        raise ValueError("No report data loaded.")
    return pd.concat(frames, ignore_index=True)


# ============================================================
# Param File Parsing / Pairing
# ============================================================

def parse_param_info(meta_path: Path) -> dict:
    """
    Parse parameter info file containing e.g.
    total number of parameters:
    38253832
    total number of classifier parameters:
    14745800

    Seed line is optional.
    """
    txt = meta_path.read_text(encoding="utf-8", errors="ignore")

    seed_m = SEED_RE.search(txt)
    total_m = TOTAL_PARAMS_RE.search(txt)
    cls_m = CLS_PARAMS_RE.search(txt)

    if total_m is None:
        raise ValueError(f"Could not find total parameter count in: {meta_path}")

    return {
        "meta_path": str(meta_path),
        "seed": int(seed_m.group(1)) if seed_m else None,
        "total_params": int(total_m.group(1)),
        "classifier_params": int(cls_m.group(1)) if cls_m else None,
        "meta_dir_name": meta_path.parent.name,
    }


def pair_reports_and_meta(report_paths: List[Path], meta_paths: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Pair by run-root directory (preferred), fallback to sorted order.
    Expects same counts.
    """
    if len(report_paths) != len(meta_paths):
        raise ValueError(
            f"Number of report files ({len(report_paths)}) must match number of meta files ({len(meta_paths)})."
        )

    report_by_root = {str(run_root_from_report_path(rp)): rp for rp in report_paths}
    meta_by_root = {str(run_root_from_meta_path(mp)): mp for mp in meta_paths}

    common_roots = sorted(
        set(report_by_root).intersection(meta_by_root),
        key=lambda s: natural_sort_key_path(Path(s)),
    )

    if len(common_roots) == len(report_paths) == len(meta_paths):
        return [(report_by_root[root], meta_by_root[root]) for root in common_roots]

    print("[WARN] Could not perfectly pair report/meta by run-root. Using sorted-order pairing.")
    return list(
        zip(
            sorted(report_paths, key=natural_sort_key_path),
            sorted(meta_paths, key=natural_sort_key_path),
        )
    )


# ============================================================
# Stats
# ============================================================

def compute_epoch_stats(df_all: pd.DataFrame, split: str, metric_col: str) -> pd.DataFrame:
    """
    Compute per-epoch mean/std/var/min/max across runs.
    Uses sample std/var (ddof=1) when n>=2.
    """
    sub = df_all[df_all["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"No rows for split={split}")

    piv = sub.pivot_table(index="epoch", columns="run", values=metric_col, aggfunc="first").sort_index()

    stats = pd.DataFrame({
        "epoch": piv.index.to_numpy(),
        "mean": piv.mean(axis=1).to_numpy(),
        "std": piv.std(axis=1, ddof=1).to_numpy(),
        "var": piv.var(axis=1, ddof=1).to_numpy(),
        "min": piv.min(axis=1).to_numpy(),
        "max": piv.max(axis=1).to_numpy(),
        "n_runs_at_epoch": piv.count(axis=1).to_numpy(),
    })

    stats["std"] = stats["std"].fillna(0.0)
    stats["var"] = stats["var"].fillna(0.0)
    return stats


def build_summary_row(stats: pd.DataFrame, title: str) -> dict:
    final = stats.iloc[-1]
    max_std_idx = int(stats["std"].idxmax())
    max_var_idx = int(stats["var"].idxmax())

    return {
        "metric": title,
        "overall_min": float(stats["min"].min()),
        "overall_max": float(stats["max"].max()),
        "final_epoch": int(final["epoch"]),
        "final_epoch_mean": float(final["mean"]),
        "final_epoch_std": float(final["std"]),
        "final_epoch_var": float(final["var"]),
        "max_std_epoch": int(stats.loc[max_std_idx, "epoch"]),
        "max_std": float(stats.loc[max_std_idx, "std"]),
        "max_var_epoch": int(stats.loc[max_var_idx, "epoch"]),
        "max_var": float(stats.loc[max_var_idx, "var"]),
    }


# ============================================================
# Plotting (single metric)
# ============================================================

def plot_metric(
    stats: pd.DataFrame,
    out_path: Path,
    title: str,
    y_label: str,
    annotate_std: bool = True,
    annotation_fontsize: int = 7,
    show_minmax: bool = False,
) -> None:
    """
    Individual plot:
    - bold mean line
    - shaded ± std (CLIPPED to observed min/max)
    - min/max dashed (optional)
    - per-point annotation ONLY std (σ)
    """
    x = stats["epoch"].to_numpy()
    mean = stats["mean"].to_numpy()
    std = stats["std"].to_numpy()
    mn = stats["min"].to_numpy()
    mx = stats["max"].to_numpy()

    lower, upper = clipped_std_band(stats)

    fig, ax = plt.subplots(figsize=(13, 7))

    line, = ax.plot(x, mean, marker="o", linewidth=3.2, label=f"Mean (n≈{int(stats['n_runs_at_epoch'].max())})")
    ax.fill_between(x, lower, upper, alpha=0.22, color=line.get_color(), label="±1 Std")

    if show_minmax:
        ax.plot(x, mn, "--", linewidth=1.3, label="Min")
        ax.plot(x, mx, "--", linewidth=1.3, label="Max")

    if annotate_std:
        y_vis_min = float(np.nanmin(lower)) if len(lower) else 0.0
        y_vis_max = float(np.nanmax(upper)) if len(upper) else 1.0
        y_span = max(y_vis_max - y_vis_min, 1e-6)
        offset = 0.025 * y_span

        for xi, yi, si, yi_up in zip(x, mean, std, upper):
            y_text = min(yi + offset, yi_up + 0.04 * y_span)
            ax.text(
                xi,
                y_text,
                f"σ={fmt_num(si)}",
                ha="center",
                va="bottom",
                fontsize=annotation_fontsize,
            )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plotting (4 metrics in 1 grid for a method)
# ============================================================

def plot_four_metrics_grid(
    metric_plot_items: List[dict],
    out_path: Path,
    suptitle: str,
    annotate_std: bool = True,
    show_minmax: bool = False,
) -> None:
    """
    metric_plot_items: list of dicts with keys
      - stats
      - title
      - y_label

    Layout order expected from caller:
      top row: Train Loss, Test Loss
      bottom row: Train Accuracy, Test Accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    for ax, item in zip(axes, metric_plot_items):
        stats = item["stats"]
        title = item["title"]
        y_label = item["y_label"]

        x = stats["epoch"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = stats["std"].to_numpy()
        mn = stats["min"].to_numpy()
        mx = stats["max"].to_numpy()

        lower, upper = clipped_std_band(stats)

        line, = ax.plot(x, mean, marker="o", linewidth=2.6, label=f"Mean (n≈{int(stats['n_runs_at_epoch'].max())})")
        ax.fill_between(x, lower, upper, alpha=0.20, color=line.get_color(), label="±1 Std")

        if show_minmax:
            ax.plot(x, mn, "--", linewidth=1.0, label="Min")
            ax.plot(x, mx, "--", linewidth=1.0, label="Max")

        if annotate_std:
            y_vis_min = float(np.nanmin(lower)) if len(lower) else 0.0
            y_vis_max = float(np.nanmax(upper)) if len(upper) else 1.0
            y_span = max(y_vis_max - y_vis_min, 1e-6)
            offset = 0.02 * y_span

            for xi, yi, si, yi_up in zip(x, mean, std, upper):
                y_text = min(yi + offset, yi_up + 0.035 * y_span)
                ax.text(xi, y_text, f"σ={fmt_num(si)}", fontsize=6, ha="center", va="bottom")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=15)  # BASE / TRL / HLGTL
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Per-method Analysis
# ============================================================

def analyze_method_reports(
    method_name: str,
    report_paths: List[Path],
    out_dir: Path,
    annotate_std_individual: bool = True,
    annotate_std_grid: bool = True,
    show_minmax_in_method_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generates per-method outputs:
    - 4 individual metric plots
    - 1 combined 4-in-1 grid plot
    - CSVs
    Returns:
      summary_df, all_epoch_stats_long_df, metric_stats_map
    """
    if len(report_paths) == 0:
        raise ValueError(f"No report files found for method: {method_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = load_reports(report_paths)

    df_all.to_csv(out_dir / "parsed_runs_long.csv", index=False)

    metric_specs = [
        ("train_acc_top1", "Train", "top1", f"{method_name} | Train Accuracy (Top-1)", "Top-1 Accuracy"),
        ("train_loss", "Train", "loss", f"{method_name} | Train Loss", "Loss"),
        ("test_acc_top1", "Test", "top1", f"{method_name} | Test Accuracy (Top-1)", "Top-1 Accuracy"),
        ("test_loss", "Test", "loss", f"{method_name} | Test Loss", "Loss"),
    ]

    summary_rows = []
    all_epoch_stats = []
    metric_stats_map: Dict[str, pd.DataFrame] = {}
    grid_item_map: Dict[str, dict] = {}

    for key, split, col, title, ylab in metric_specs:
        stats = compute_epoch_stats(df_all, split=split, metric_col=col)
        stats["metric_key"] = key
        stats["metric_title"] = title
        all_epoch_stats.append(stats.copy())
        metric_stats_map[key] = stats.copy()

        stats[["epoch", "mean", "std", "var", "min", "max", "n_runs_at_epoch"]].to_csv(
            out_dir / f"{key}_epoch_stats.csv", index=False
        )

        plot_metric(
            stats=stats,
            out_path=out_dir / f"{key}.png",
            title=title,
            y_label=ylab,
            annotate_std=annotate_std_individual,
            show_minmax=show_minmax_in_method_plots,
        )

        grid_item_map[key] = {
            "stats": stats,
            "title": title.replace(f"{method_name} | ", ""),
            "y_label": ylab,
        }

        summary_rows.append(build_summary_row(stats, title))

    # Requested order:
    # top row: train loss | test loss
    # bottom row: train acc | test acc
    grid_order = ["train_loss", "test_loss", "train_acc_top1", "test_acc_top1"]
    metric_grid_items = [grid_item_map[k] for k in grid_order]

    plot_four_metrics_grid(
        metric_plot_items=metric_grid_items,
        out_path=out_dir / "all_four_metrics_grid.png",
        suptitle=method_name,
        annotate_std=annotate_std_grid,
        show_minmax=show_minmax_in_method_plots,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_stats.csv", index=False)

    all_epoch_stats_df = pd.concat(all_epoch_stats, ignore_index=True)
    all_epoch_stats_df.to_csv(out_dir / "all_epoch_stats_long.csv", index=False)

    return summary_df, all_epoch_stats_df, metric_stats_map


# ============================================================
# Parameter-vs-Accuracy (per method)
# ============================================================

def final_or_best_test_top1(df_report: pd.DataFrame, mode: str = "final") -> float:
    """Extract final or best Test top1 from one parsed report dataframe."""
    t = df_report[df_report["split"] == "Test"].sort_values("epoch")
    if t.empty:
        raise ValueError("No Test rows in report.")
    if mode == "best":
        return float(t["top1"].max())
    return float(t.iloc[-1]["top1"])


def build_param_vs_accuracy_plot(
    method_name: str,
    report_paths: List[Path],
    meta_paths: List[Path],
    out_dir: Path,
    accuracy_mode: str = "final",              # "final" or "best"
    aggregate_same_param_count: bool = True,
    x_param_col: str = "classifier_params",    # "classifier_params" or "total_params"
) -> pd.DataFrame:
    """
    Per-method parameter-vs-accuracy plot.
    x_param_col controls x-axis: classifier_params or total_params.
    """
    if x_param_col not in {"classifier_params", "total_params"}:
        raise ValueError("x_param_col must be 'classifier_params' or 'total_params'")

    pairs = pair_reports_and_meta(report_paths, meta_paths)

    rows = []
    for i, (rp, mp) in enumerate(pairs, start=1):
        rep_df = parse_report(rp)
        pinfo = parse_param_info(mp)
        test_acc = final_or_best_test_top1(rep_df, mode=accuracy_mode)

        rows.append({
            "idx": i,
            "run_name": run_root_from_report_path(rp).name,
            "report_path": str(rp),
            "meta_path": str(mp),
            "seed": pinfo.get("seed"),
            "total_params": pinfo["total_params"],
            "classifier_params": pinfo.get("classifier_params"),
            f"test_top1_{accuracy_mode}": test_acc,
        })

    df = pd.DataFrame(rows)
    ycol = f"test_top1_{accuracy_mode}"
    xcol = x_param_col

    df = df[df[xcol].notna()].copy()
    if df.empty:
        raise ValueError(f"No rows with valid '{xcol}' values for method {method_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"params_vs_accuracy_{accuracy_mode}_{xcol}.csv", index=False)

    x_label_map = {
        "classifier_params": "Classifier Parameters",
        "total_params": "Total Parameters",
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    if aggregate_same_param_count:
        agg = (
            df.groupby(xcol, dropna=False)[ycol]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .sort_values(xcol)
        )
        agg["std"] = agg["std"].fillna(0.0)

        ax.scatter(df[xcol], df[ycol], alpha=0.50, label="Runs")

        ax.errorbar(
            agg[xcol],
            agg["mean"],
            yerr=agg["std"],
            fmt="o-",
            linewidth=2.2,
            capsize=4,
            label="Mean ± Std (same param count)",
        )
    else:
        ax.scatter(df[xcol], df[ycol], label="Runs", alpha=0.9)
        ax.scatter(
            [float(df[xcol].mean())],
            [float(df[ycol].mean())],
            s=220,
            edgecolors="black",
            linewidths=1.2,
            label="Mean",
        )

    ax.set_title(f"{method_name} | {x_label_map[xcol]} vs Test Accuracy ({accuracy_mode.title()})")
    ax.set_xlabel(x_label_map[xcol])
    ax.set_ylabel(f"Top-1 Accuracy [{accuracy_mode}]")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_dir / f"params_vs_accuracy_{accuracy_mode}_{xcol}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return df


# ============================================================
# Cross-Method Comparison (4 metrics in one figure)
# ============================================================

def metric_key_pretty(metric_key: str) -> Tuple[str, str]:
    mapping = {
        "train_acc_top1": ("Train Accuracy (Top-1)", "Top-1 Accuracy"),
        "train_loss": ("Train Loss", "Loss"),
        "test_acc_top1": ("Test Accuracy (Top-1)", "Top-1 Accuracy"),
        "test_loss": ("Test Loss", "Loss"),
    }
    return mapping[metric_key]


def plot_all_methods_comparison_grid(
    methods_metric_stats: Dict[str, Dict[str, pd.DataFrame]],
    out_path: Path,
    shade_std: bool = True,
) -> None:
    """
    Create one 2x2 figure where each subplot is a metric,
    and all methods are overlaid (mean line + std band).
    No std text annotations here (per request).
    """
    metric_keys_order = ["train_loss", "test_loss", "train_acc_top1", "test_acc_top1"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    for ax, metric_key in zip(axes, metric_keys_order):
        title, ylab = metric_key_pretty(metric_key)

        for method_name, metric_map in methods_metric_stats.items():
            if metric_key not in metric_map:
                continue

            stats = metric_map[metric_key]
            x = stats["epoch"].to_numpy()
            mean = stats["mean"].to_numpy()
            lower, upper = clipped_std_band(stats)

            line, = ax.plot(x, mean, marker="o", linewidth=2.4, label=f"{method_name} mean")
            if shade_std:
                ax.fill_between(x, lower, upper, alpha=0.12, color=line.get_color())

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("All Methods Comparison | Mean ± Std Across Runs", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Cross-Method Parameter Comparison (RAW scatter, all runs)
# ============================================================

def plot_all_methods_params_vs_accuracy_raw_scatter(
    method_param_tables: Dict[str, pd.DataFrame],
    out_path: Path,
    accuracy_mode: str = "final",
    x_param_col: str = "classifier_params",
    title: Optional[str] = None,
) -> None:
    """
    Overlay raw parameter-vs-accuracy points for all methods (e.g., 45 points for 15x3)
    and also overlay a larger mean marker per method.

    x_param_col: "classifier_params" or "total_params"
    """
    if x_param_col not in {"classifier_params", "total_params"}:
        raise ValueError("x_param_col must be 'classifier_params' or 'total_params'")

    ycol = f"test_top1_{accuracy_mode}"
    xcol = x_param_col

    x_label_map = {
        "classifier_params": "Classifier Parameters",
        "total_params": "Total Parameters",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    any_plotted = False

    for method_name, df in method_param_tables.items():
        if df is None or df.empty or ycol not in df.columns or xcol not in df.columns:
            continue

        df_use = df[df[xcol].notna()].copy()
        if df_use.empty:
            continue

        any_plotted = True

        style = METHOD_SCATTER_STYLE.get(method_name, {"marker": "o", "color": None, "size": 65})
        marker = style["marker"]
        color = style["color"]
        size = style["size"]

        ax.scatter(
            df_use[xcol],
            df_use[ycol],
            s=size,
            marker=marker,
            c=color,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            label=method_name,
            zorder=3,
        )

        mx = float(df_use[xcol].mean())
        my = float(df_use[ycol].mean())
        ax.scatter(
            [mx], [my],
            s=size * 2.3,
            marker=marker,
            c=color,
            edgecolors="black",
            linewidths=1.5,
            alpha=1.0,
            zorder=5,
        )

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_title(title or "")
    ax.set_xlabel(x_label_map[xcol])
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.backbone == "resnet":
        BASE_OUTPUT_DIR = BASE_OUTPUT_DIR_RESNET
        METHODS = METHODS_RESNET
        COMBINED_PARAM_SCATTER_TITLE = COMBINED_PARAM_SCATTER_TITLE_RESNET
    else:
        BASE_OUTPUT_DIR = BASE_OUTPUT_DIR_EFFICIENTNET
        METHODS = METHODS_EFFICIENTNET
        COMBINED_PARAM_SCATTER_TITLE = COMBINED_PARAM_SCATTER_TITLE_EFFICIENTNET

    SHOW_MINMAX_LINES = bool(args.show_minmax)

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods_metric_stats: Dict[str, Dict[str, pd.DataFrame]] = {}
    methods_param_tables: Dict[str, pd.DataFrame] = {}

    any_method_processed = False

    print(f"[DEBUG] backbone   = {args.backbone}")
    print(f"[DEBUG] show_minmax= {SHOW_MINMAX_LINES}")
    print(f"[DEBUG] cwd        = {Path.cwd()}")
    print(f"[DEBUG] script_dir = {SCRIPT_DIR}")
    print(f"[DEBUG] repo_root  = {REPO_ROOT}")
    print(f"[DEBUG] out_root   = {BASE_OUTPUT_DIR.resolve()}")

    for cfg in METHODS:
        method_name = cfg["name"]
        method_slug = slugify(method_name)
        method_out_dir = BASE_OUTPUT_DIR / method_slug

        report_patterns = cfg.get("report_patterns", []) or []
        param_patterns = cfg.get("param_patterns", []) or []

        print(f"\n{'=' * 80}")
        print(f"[METHOD] {method_name}")
        print(f"{'=' * 80}")

        # Seed-run reports (base set)
        report_paths_seed = expand_paths(report_patterns)
        if len(report_paths_seed) == 0:
            print(f"[SKIP] No report files found for '{method_name}'. Check report_patterns.")
            continue

        # Optional extra experiment report (highlight) -> appended to METRIC analysis only
        highlight_report_path: Optional[Path] = None
        highlight_spec = cfg.get("highlight_report")
        if highlight_spec:
            highlight_matches = expand_paths([highlight_spec])
            if len(highlight_matches) == 0:
                print(f"[WARN] highlight_report not found for {method_name}: {highlight_spec}")
            else:
                if len(highlight_matches) > 1:
                    print(f"[WARN] highlight_report matched multiple files for {method_name}; using first one.")
                highlight_report_path = highlight_matches[0]

        report_paths_metrics = append_unique_path(report_paths_seed, highlight_report_path)

        any_method_processed = True
        print(f"[INFO] Found {len(report_paths_seed)} seed report file(s) for {method_name}")
        for p in report_paths_seed[:5]:
            print(f"  - {p}")
        if len(report_paths_seed) > 5:
            print(f"  ... (+{len(report_paths_seed) - 5} more)")

        if highlight_report_path is not None:
            if len(report_paths_metrics) > len(report_paths_seed):
                print(f"[INFO] Added highlight_report as external run for metric plots/stats:")
                print(f"       {highlight_report_path}")
                print(f"[INFO] Metric run count now: {len(report_paths_metrics)}")
            else:
                print(f"[INFO] highlight_report already included in report_patterns (no duplicate append).")

        # Per-method report analysis (4 individual + 1 grid + CSVs)
        summary_df, _, metric_stats_map = analyze_method_reports(
            method_name=method_name,
            report_paths=report_paths_metrics,
            out_dir=method_out_dir,
            annotate_std_individual=ANNOTATE_STD_INDIVIDUAL,
            annotate_std_grid=ANNOTATE_STD_GRID,
            show_minmax_in_method_plots=SHOW_MINMAX_LINES,
        )
        methods_metric_stats[method_name] = metric_stats_map

        print(f"[OK] Saved method plots + CSVs to: {method_out_dir}")
        print(summary_df.round(4).to_string(index=False))

        # Parameter plots (paired with param files)
        # These use seed reports only to keep 1:1 pairing with param files.
        if param_patterns:
            meta_paths = expand_paths(param_patterns)
            if len(meta_paths) == 0:
                print(f"[WARN] No param files found for '{method_name}' (patterns provided but no matches).")
            else:
                try:
                    df_for_method = None
                    for x_field in PARAM_X_FIELDS:
                        df_tmp = build_param_vs_accuracy_plot(
                            method_name=method_name,
                            report_paths=report_paths_seed,
                            meta_paths=meta_paths,
                            out_dir=method_out_dir / "param_vs_accuracy",
                            accuracy_mode=PARAM_ACCURACY_MODE,
                            aggregate_same_param_count=PER_METHOD_PARAM_AGGREGATE,
                            x_param_col=x_field,
                        )
                        df_for_method = df_tmp
                        print(f"[OK] Saved parameter-vs-accuracy plot for {method_name} using x={x_field}")

                    if df_for_method is not None:
                        methods_param_tables[method_name] = df_for_method

                except Exception as e:
                    print(f"[WARN] Failed parameter-vs-accuracy for {method_name}: {e}")
        else:
            print(f"[INFO] No param_patterns provided for '{method_name}' (skipping parameter plots).")

    if not any_method_processed:
        print("\n[EXIT] No methods processed. Check your CONFIG paths.")
        return

    # Cross-method 4-metric comparison (no std annotations)
    if len(methods_metric_stats) >= 2:
        combined_dir = BASE_OUTPUT_DIR / "all_methods_comparison"
        combined_dir.mkdir(parents=True, exist_ok=True)

        plot_all_methods_comparison_grid(
            methods_metric_stats=methods_metric_stats,
            out_path=combined_dir / "all_methods_comparison_grid.png",
            shade_std=True,
        )
        print(f"\n[OK] Saved all-methods comparison grid to: {combined_dir / 'all_methods_comparison_grid.png'}")
    else:
        print("\n[INFO] Only one method processed; skipping all-methods comparison grid.")

    # Cross-method parameter comparison (RAW scatter: all runs)
    valid_param_methods = {k: v for k, v in methods_param_tables.items() if v is not None and not v.empty}
    if len(valid_param_methods) >= 2:
        combined_dir = BASE_OUTPUT_DIR / "all_methods_comparison"
        combined_dir.mkdir(parents=True, exist_ok=True)

        for x_field in PARAM_X_FIELDS:
            plot_all_methods_params_vs_accuracy_raw_scatter(
                method_param_tables=valid_param_methods,
                out_path=combined_dir / f"all_methods_params_vs_accuracy_{x_field}.png",
                accuracy_mode=PARAM_ACCURACY_MODE,
                x_param_col=x_field,
                title=COMBINED_PARAM_SCATTER_TITLE,
            )
            print(
                f"[OK] Saved all-methods RAW parameter scatter ({x_field}) to: "
                f"{combined_dir / f'all_methods_params_vs_accuracy_{x_field}.png'}"
            )
    elif len(valid_param_methods) == 1:
        print("[INFO] Parameter plots exist for only one method; skipping cross-method parameter comparisons.")
    else:
        print("[INFO] No valid parameter tables found; skipping cross-method parameter comparisons.")

    print(f"\n[DONE] Outputs root: {BASE_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()