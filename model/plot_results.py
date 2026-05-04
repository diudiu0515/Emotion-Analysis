import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"


def load_all_runs(results_dir=RESULTS_DIR):
    records = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "figures":
            continue

        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        metrics_df = pd.read_csv(metrics_path)

        for _, row in metrics_df.iterrows():
            record = {
                "run": run_dir.name,
                "model": row["model"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "loss": row["loss"],
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "lr": config.get("lr"),
                "dropout": config.get("dropout"),
                "weight_decay": config.get("weight_decay"),
                "max_len": config.get("max_len"),
            }
            records.append(record)

    return pd.DataFrame(records)


def plot_f1_by_run(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["run"], sub["f1"], marker="o", label=model)

    plt.xlabel("Run")
    plt.ylabel("F1 Score")
    plt.title("F1 Score across Different Runs")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "f1_by_run.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


def plot_accuracy_by_run(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["run"], sub["accuracy"], marker="o", label=model)

    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across Different Runs")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "accuracy_by_run.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


def plot_best_model_bar(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    best_df = df.sort_values("f1", ascending=False).groupby("model").head(1)

    plt.figure(figsize=(8, 5))
    plt.bar(best_df["model"], best_df["f1"])

    plt.xlabel("Model")
    plt.ylabel("Best F1 Score")
    plt.title("Best F1 Score of Each Model")
    plt.tight_layout()

    save_path = FIGURE_DIR / "best_f1_by_model.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


def plot_dropout_effect(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if df["dropout"].isna().all():
        print("No dropout information found.")
        return

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("dropout")
        plt.plot(sub["dropout"], sub["f1"], marker="o", label=model)

    plt.xlabel("Dropout")
    plt.ylabel("F1 Score")
    plt.title("Effect of Dropout on F1 Score")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "dropout_f1.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


def plot_lr_effect(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if df["lr"].isna().all():
        print("No learning rate information found.")
        return

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("lr")
        plt.plot(sub["lr"], sub["f1"], marker="o", label=model)

    plt.xlabel("Learning Rate")
    plt.ylabel("F1 Score")
    plt.title("Effect of Learning Rate on F1 Score")
    plt.legend()
    plt.tight_layout()

    save_path = FIGURE_DIR / "lr_f1.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


def save_summary_table(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_path = RESULTS_DIR / "all_runs_summary.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {save_path}")

    best_path = RESULTS_DIR / "best_results.csv"
    best_df = df.sort_values("f1", ascending=False)
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {best_path}")


def main():
    df = load_all_runs()

    if df.empty:
        print("No valid result logs found.")
        return

    print(df)

    save_summary_table(df)

    plot_f1_by_run(df)
    plot_accuracy_by_run(df)
    plot_best_model_bar(df)
    plot_dropout_effect(df)
    plot_lr_effect(df)


if __name__ == "__main__":
    main()