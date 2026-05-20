from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_condition( 
# 10.3 aggregation of outputs into comparable summaries
    results: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    summary = (
        results
        .groupby("condition")[metric] # (basline, adversarial, self-check)[selects the metric column i want to plot]
        .mean()
        .reset_index()
        .sort_values("condition") # alphabetical order. useless but i already saved results
    ) # average selected metric for each prompt condition

    plt.figure(figsize=(8, 5))
    plt.bar(summary["condition"], summary[metric])
    plt.xlabel("Experimental condition")
    plt.ylabel(metric)
    plt.title(f"{metric} by prompt condition")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close() 
# i don't really use this in my results


def plot_metric_by_dataset_and_condition(
    results: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    summary = (
        results
        .groupby(["source_dataset", "condition"])[metric]
        .mean()
        .reset_index()
    ) # average selected metric for each dataset-condition pair

    pivot = summary.pivot( # summary table into a pivot table. makes each condition a row and each dataset a column
        index="condition",
        columns="source_dataset",
        values=metric,
    )

    pivot.plot(kind="bar", figsize=(8, 5)) # grouped bar chart from the pivot table

    plt.xlabel("Experimental condition")
    plt.ylabel(metric)
    plt.title(f"{metric} by dataset and prompt condition")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def make_all_plots(results_path: str, figures_dir: str) -> None:
    results = pd.read_csv(results_path) # load experiment results

    metrics = [
        "fuzzy_match",
        "contradiction_marker",
        "reasoning_length",
        "truth_token_probability",
    ]

    for metric in metrics:
        output_path = Path(figures_dir) / f"{metric}_by_condition.png"
        plot_metric_by_condition(results, metric, str(output_path))

        dataset_output_path = Path(figures_dir) / f"{metric}_by_dataset_and_condition.png"
        plot_metric_by_dataset_and_condition(
            results,
            metric,
            str(dataset_output_path),
        )

