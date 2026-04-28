from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_condition(
    results: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    """
    Plot average metric by condition.

    # Inspired by: L10.3-bias-completion-task.ipynb — aggregation of outputs into comparable summaries.
    """
    summary = (
        results
        .groupby("condition")[metric]
        .mean()
        .reset_index()
        .sort_values("condition")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(summary["condition"], summary[metric])
    plt.xlabel("Experimental condition")
    plt.ylabel(metric)
    plt.title(f"{metric} by prompt condition")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def make_all_plots(results_path: str, figures_dir: str) -> None:
    """
    Create all project plots.
    """
    results = pd.read_csv(results_path)

    metrics = [
        "fuzzy_match",
        "contains_reference",
        "contradiction_marker",
        "reasoning_length",
    ]

    for metric in metrics:
        output_path = Path(figures_dir) / f"{metric}_by_condition.png"
        plot_metric_by_condition(results, metric, str(output_path))