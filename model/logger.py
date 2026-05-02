import os
import json
import csv
from datetime import datetime


def create_run_dir():
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(args, run_dir):
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


def save_metrics(run_dir, model_name, test_metrics):
    metrics_path = os.path.join(run_dir, "metrics.csv")
    file_exists = os.path.exists(metrics_path)

    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "loss"
            ])

        writer.writerow([
            model_name,
            test_metrics.acc,
            test_metrics.precision,
            test_metrics.recall,
            test_metrics.f1,
            test_metrics.loss
        ])