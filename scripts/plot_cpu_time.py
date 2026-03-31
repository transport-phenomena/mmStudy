from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR.parent / "results"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
        }
    )


def read_cpu_time(csv_path: Path) -> tuple[list[float], list[str], list[list[float]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = [item.strip() for item in next(reader)]

        if len(header) < 3:
            raise ValueError("Expected at least three columns in cpuTime.csv")

        x_values: list[float] = []
        series = [[] for _ in header[1:-1]]

        for row in reader:
            if not row or all(not item.strip() for item in row):
                continue

            values = [float(item.strip()) for item in row]
            denominator = values[-1]
            if denominator == 0.0:
                raise ValueError(f"Last-column value is zero for row: {row}")

            x_values.append(values[0])
            for i, value in enumerate(values[1:-1]):
                series[i].append(value / denominator)

    return x_values, header[1:-1], series


def plot_ratios(csv_path: Path, output_path: Path, show: bool) -> None:
    configure_plot_style()
    x_values, labels, series = read_cpu_time(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, y_values in zip(labels, series):
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel(r"Number of particles")
    ax.set_ylabel(r"CPU time ratio to exact inverse ($M^{-1}$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cpuTime.csv as ratios to the last column, including the Pairwise series."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULT_DIR / "cpuTime.csv",
        help="Path to cpuTime.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "cpuTime_ratios.pdf",
        help="Output figure path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving the figure",
    )
    args = parser.parse_args()

    plot_ratios(args.csv, args.output, args.show)


if __name__ == "__main__":
    main()
