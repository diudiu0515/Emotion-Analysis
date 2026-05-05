from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures_dropout"

RUNS = [
    "2026-05-02_21-08",  # dropout=0.6
    "2026-05-02_21-34",  # dropout=0.5
    "2026-05-02_21-40",  # dropout=0.3
]


def load_run(run_name):
    run_dir = RESULTS_DIR / run_name
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"

    df = pd.read_csv(metrics_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df["run"] = run_name
    df["dropout"] = config["dropout"]

    return df


def load_all():
    dfs = [load_run(run) for run in RUNS]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("dropout")
    return df


def plot_metric(df, metric):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("dropout")
        plt.plot(
            sub["dropout"],
            sub[metric],
            marker="o",
            label=model
        )

    plt.xlabel("Dropout")
    plt.ylabel(metric.upper())
    plt.title(f"Effect of Dropout on {metric.upper()}")
    plt.legend()
    plt.tight_layout()

    save_path = FIG_DIR / f"dropout_{metric}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    df = load_all()

    print(df[["run", "model", "dropout", "accuracy", "f1"]])

    plot_metric(df, "f1")
    plot_metric(df, "accuracy")


if __name__ == "__main__":
    main()