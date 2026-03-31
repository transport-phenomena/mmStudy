#!/usr/bin/env python3

import csv
import math
import os
from pathlib import Path

mpl_config_dir = Path(__file__).resolve().parent / ".mplconfig"
mpl_config_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri


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


def load_results(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    volume_fraction: list[float] = []
    n_particles: list[float] = []
    residual3: list[float] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            volume_fraction.append(float(row["volumeFraction"]))
            n_particles.append(float(row["nParticles"]))
            residual3.append(float(row["residual3"]))

    return volume_fraction, n_particles, residual3


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    result_dir = script_dir.parent / "results"

    csv_path = result_dir / "results.csv"
    output_path = result_dir / "residual3_contour.pdf"

    configure_plot_style()
    volume_fraction, n_particles, residual3 = load_results(csv_path)
    log_volume_fraction = np.log10(np.asarray(volume_fraction))
    n_particles_array = np.asarray(n_particles)
    residual3_array = np.asarray(residual3)

    triangulation = tri.Triangulation(log_volume_fraction, n_particles_array)
    contour_levels = np.logspace(
        math.log10(float(residual3_array.min())),
        math.log10(float(residual3_array.max())),
        12,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    filled_contour = ax.tricontourf(
        triangulation,
        residual3_array,
        levels=contour_levels,
        cmap="viridis",
        norm=LogNorm(vmin=float(residual3_array.min()), vmax=float(residual3_array.max())),
    )
    contour = ax.tricontour(
        triangulation,
        residual3_array,
        levels=contour_levels,
        colors="black",
        linewidths=0.6,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.1e")
    ax.scatter(
        log_volume_fraction,
        n_particles_array,
        s=5,
        c="black",
        alpha=0.18,
        linewidths=0,
    )

    tick_min = math.floor(float(log_volume_fraction.min()))
    tick_max = math.ceil(float(log_volume_fraction.max()))
    tick_positions = list(range(tick_min, tick_max + 1))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([rf"$10^{{{tick}}}$" for tick in tick_positions])
    ax.set_xlabel(r"Volume fraction")
    ax.set_ylabel(r"$n_{\mathrm{Particles}}$")
    ax.set_title(r"$\|M^{-1} - (I - K + K^2)\|$ isolines")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    colorbar = fig.colorbar(filled_contour, ax=ax)
    colorbar.set_label(r"Residual", fontsize=18)
    colorbar.ax.tick_params(labelsize=15)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Wrote contour plot to {output_path}")


if __name__ == "__main__":
    main()
