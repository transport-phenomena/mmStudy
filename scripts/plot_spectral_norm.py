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


def load_results(csv_path: Path) -> tuple[list[float], list[float]]:
    volume_fraction: list[float] = []
    spectral_norm: list[float] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            volume_fraction_value = row.get("volumeFraction")
            spectral_norm_value = row.get("spectralNorm")

            if not volume_fraction_value or not spectral_norm_value:
                continue

            volume_fraction.append(float(volume_fraction_value))
            spectral_norm.append(float(spectral_norm_value))

    return volume_fraction, spectral_norm


def main() -> None:
    csv_path = RESULT_DIR / "spectralNorm_2.csv"
    output_path = RESULT_DIR / "spectralRadius_plot.pdf"

    configure_plot_style()
    volume_fraction, spectral_norm = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(volume_fraction, spectral_norm, s=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Volume fraction, $\phi$")
    ax.set_ylabel(r"Spectral radius, $\rho(K)$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
