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


def load_results(csv_path: Path) -> dict[str, list[float]]:
    columns = {
        "nParticles": [],
        "volumeFraction": [],
        "residualPair": [],
        "residual2": [],
        "residual3": [],
        "residual4": [],
        "residual5": [],
        "residualANN": [],
    }

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            columns["nParticles"].append(float(row["nParticles"]))
            columns["volumeFraction"].append(float(row["volumeFraction"]))
            columns["residualPair"].append(float(row["residualPair"]))
            columns["residual2"].append(float(row["residual2"]))
            columns["residual3"].append(float(row["residual3"]))
            columns["residual4"].append(float(row["residual4"]))
            columns["residual5"].append(float(row["residual5"]))
            columns["residualANN"].append(float(row["residualANN"]))

    return columns


def build_threshold_mask(
    n_particles: list[float], volume_fraction: list[float]
) -> list[bool]:
    mask: list[bool] = []
    for particle_count, phi in zip(n_particles, volume_fraction):
        threshold_volume_fraction = 0.67 * (particle_count ** (-2.08))
        mask.append(phi < threshold_volume_fraction)

    return mask


def apply_mask(values: list[float], mask: list[bool]) -> list[float]:
    return [value for value, keep in zip(values, mask) if keep]


def main() -> None:
    csv_path = RESULT_DIR / "results.csv"
    pdf_path = RESULT_DIR / "results_plot.pdf"
    filtered_pdf_path = RESULT_DIR / "results_plot_below_threshold.pdf"

    configure_plot_style()
    data = load_results(csv_path)
    threshold_mask = build_threshold_mask(
        data["nParticles"],
        data["volumeFraction"],
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
#    ax.scatter(data["volumeFraction"], data["residualPair"], s=12, label=r"$\|M^{-1} - \mathrm{PRM}\|$")
    ax.scatter(data["volumeFraction"], data["residual2"], s=12, label=r"$\|M^{-1} - (I + K)\|$")
    ax.scatter(data["volumeFraction"], data["residual3"], s=12, label=r"$\|M^{-1} - (I + K + K^2)\|$")
    ax.scatter(data["volumeFraction"], data["residual4"], s=12, label=r"$\|M^{-1} - (I + K + K^2 + K^3)\|$")
    ax.scatter(data["volumeFraction"], data["residual5"], s=12, label=r"$\|M^{-1} - (I + K + K^2 + K^3 + K^4)\|$")
    ax.scatter(data["volumeFraction"], data["residualANN"], s=12, label=r"$\|M^{-1} - \mathrm{ANN}\|$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Volume fraction")
    ax.set_ylabel(r"Residual norm")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
#    ax.scatter(
#        apply_mask(data["volumeFraction"], threshold_mask),
#        apply_mask(data["residualPair"], threshold_mask),
#        s=12,
#        label=r"$\|M^{-1} - \mathrm{PRM}\|$",
#    )
    ax.scatter(
        apply_mask(data["volumeFraction"], threshold_mask),
        apply_mask(data["residual2"], threshold_mask),
        s=12,
        label=r"$\|\mathbf{M}^{-1} - (\mathbf{I} + \mathbf{K})\|_F$",
    )
    ax.scatter(
        apply_mask(data["volumeFraction"], threshold_mask),
        apply_mask(data["residual3"], threshold_mask),
        s=12,
        label=r"$\|\mathbf{M}^{-1} - (\mathbf{I} + \mathbf{K} + \mathbf{K}^2)\|_F$",
    )
    ax.scatter(
        apply_mask(data["volumeFraction"], threshold_mask),
        apply_mask(data["residual4"], threshold_mask),
        s=12,
        label=r"$\|\mathbf{M}^{-1} - (\mathbf{I} + \mathbf{K} + \mathbf{K}^2 + \mathbf{K}^3)\|_F$",
    )
    ax.scatter(
        apply_mask(data["volumeFraction"], threshold_mask),
        apply_mask(data["residual5"], threshold_mask),
        s=12,
        label=r"$\|\mathbf{M}^{-1} - (\mathbf{I} + \mathbf{K} + \mathbf{K}^2 + \mathbf{K}^3 + \mathbf{K}^4)\|_F$",
    )
    ax.scatter(
        apply_mask(data["volumeFraction"], threshold_mask),
        apply_mask(data["residualANN"], threshold_mask),
        s=12,
        label=r"$\|\mathbf{M}^{-1} - \mathrm{ANN}\|_F$",
    )


    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_xlabel(r"Volume fraction, $\varphi$")
    #ax.set_ylabel(r"Frobenius norm of the residual")
    #ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    #ax.legend()
#
    #fig.tight_layout()
    #fig.savefig(filtered_pdf_path)
    #plt.close(fig)

    print(f"Wrote plot to {pdf_path}")
    #print(f"Wrote thresholded plot to {filtered_pdf_path}")


if __name__ == "__main__":
    main()
