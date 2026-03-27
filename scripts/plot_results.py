#!/usr/bin/env python3

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "results"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_results(csv_path: Path) -> dict[str, list[float]]:
    columns = {
        "volumeFraction": [],
        "residualPair": [],
        "residual2": [],
        "residual3": [],
        "residual4": [],
        "residual5": [],
    }

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            columns["volumeFraction"].append(float(row["volumeFraction"]))
            columns["residualPair"].append(float(row["residualPair"]))
            columns["residual2"].append(float(row["residual2"]))
            columns["residual3"].append(float(row["residual3"]))
            columns["residual4"].append(float(row["residual4"]))
            columns["residual5"].append(float(row["residual5"]))

    return columns


def main() -> None:
    csv_path = RESULT_DIR / "results.csv"
    pdf_path = RESULT_DIR / "results_plot.pdf"

    data = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(data["volumeFraction"], data["residualPair"], s=12, label="|| M^-1 - PRM ||")
    ax.scatter(data["volumeFraction"], data["residual2"], s=12, label="|| M^-1 - (I - K)||")
    ax.scatter(data["volumeFraction"], data["residual3"], s=12, label="|| M^-1 - (I - K + K^2)||")
    ax.scatter(data["volumeFraction"], data["residual4"], s=12, label="|| M^-1 - (I - K + K^2 - K^3)||")
    ax.scatter(data["volumeFraction"], data["residual5"], s=12, label="|| M^-1 - (I - K + K^2 - K^3 + K^4)||")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Volume fraction")
    ax.set_ylabel("Residual norm")
   # ax.set_title("Residuals vs Volume Fraction")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"Wrote plot to {pdf_path}")


if __name__ == "__main__":
    main()
