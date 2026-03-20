#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_results(csv_path: Path) -> dict[str, list[float]]:
    columns = {
        "volumeFraction": [],
        "residual2": [],
        "residual3": [],
        "residual4": [],
        "residual5": [],
    }

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            columns["volumeFraction"].append(float(row["volumeFraction"]))
            columns["residual2"].append(float(row["residual2"]))
            columns["residual3"].append(float(row["residual3"]))
            columns["residual4"].append(float(row["residual4"]))
            columns["residual5"].append(float(row["residual5"]))

    return columns


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "results.csv"
    pdf_path = project_root / "results_plot.pdf"

    data = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(data["volumeFraction"], data["residual2"], s=12, label="|| M^-1 - (I9 - K)||")
    ax.scatter(data["volumeFraction"], data["residual3"], s=12, label="|| M^-1 - (I9 - K + K^2)||")
    ax.scatter(data["volumeFraction"], data["residual4"], s=12, label="|| M^-1 - (I9 - K + K^2 - K^3)||")
    ax.scatter(data["volumeFraction"], data["residual5"], s=12, label="|| M^-1 - (I9 - K + K^2 - K^3 + K^4)||")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Volume Fraction")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Volume Fraction")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"Wrote plot to {pdf_path}")


if __name__ == "__main__":
    main()
