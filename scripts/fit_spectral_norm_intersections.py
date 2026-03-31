#!/usr/bin/env python3

import csv
import math
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


def load_xy(csv_path: Path) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            x_value = row.get("volumeFraction")
            y_value = row.get("spectralNorm")

            if not x_value or not y_value:
                continue

            x_values.append(float(x_value))
            y_values.append(float(y_value))

    if len(x_values) < 2:
        raise ValueError(f"Need at least two valid rows in {csv_path}")

    return x_values, y_values


def fit_power_law(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    if any(value <= 0.0 for value in x_values):
        raise ValueError("Power-law fitting requires positive volumeFraction values")
    if any(value <= 0.0 for value in y_values):
        raise ValueError("Power-law fitting requires positive spectralNorm values")

    log_x_values = [math.log10(value) for value in x_values]
    log_y_values = [math.log10(value) for value in y_values]

    n_points = len(log_x_values)
    x_mean = sum(log_x_values) / n_points
    y_mean = sum(log_y_values) / n_points

    numerator = 0.0
    denominator = 0.0
    for log_x_value, log_y_value in zip(log_x_values, log_y_values):
        x_delta = log_x_value - x_mean
        numerator += x_delta * (log_y_value - y_mean)
        denominator += x_delta * x_delta

    if denominator == 0.0:
        raise ValueError(
            "Cannot fit a power law when all volumeFraction values are identical"
        )

    exponent = numerator / denominator
    log_coefficient = y_mean - exponent * x_mean
    coefficient = 10.0 ** log_coefficient
    return exponent, coefficient


def intersection_at_spectral_norm_1(
    exponent: float, coefficient: float
) -> float | None:
    if exponent == 0.0 or coefficient <= 0.0:
        return None

    return (1.0 / coefficient) ** (1.0 / exponent)


def build_fit_curve(
    x_values: list[float], exponent: float, coefficient: float, n_points: int = 200
) -> tuple[list[float], list[float]]:
    x_min = min(x_values)
    x_max = max(x_values)

    if x_min <= 0.0 or x_max <= 0.0:
        raise ValueError("Fit curve requires positive x values")

    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    x_curve: list[float] = []
    y_curve: list[float] = []
    for index in range(n_points):
        fraction = index / (n_points - 1)
        x_value = 10.0 ** (log_x_min + fraction * (log_x_max - log_x_min))
        x_curve.append(x_value)
        y_curve.append(coefficient * (x_value ** exponent))

    return x_curve, y_curve


def fit_affine(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    n_points = len(x_values)
    x_mean = sum(x_values) / n_points
    y_mean = sum(y_values) / n_points

    numerator = 0.0
    denominator = 0.0
    for x_value, y_value in zip(x_values, y_values):
        x_delta = x_value - x_mean
        numerator += x_delta * (y_value - y_mean)
        denominator += x_delta * x_delta

    if denominator == 0.0:
        raise ValueError("Cannot fit an affine model when all x values are identical")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def fit_exponential(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    if any(value <= 0.0 for value in y_values):
        raise ValueError("Exponential fitting requires positive y values")

    log_y_values = [math.log(value) for value in y_values]
    exponent, log_coefficient = fit_affine(x_values, log_y_values)
    coefficient = math.exp(log_coefficient)
    return exponent, coefficient


def fit_log_quadratic(
    x_values: list[float], y_values: list[float]
) -> tuple[float, float, float]:
    if any(value <= 0.0 for value in y_values):
        raise ValueError("Quadratic log fitting requires positive y values")

    n_points = len(x_values)
    log_y_values = [math.log(value) for value in y_values]

    s_x = sum(x_values)
    s_x2 = sum(x_value * x_value for x_value in x_values)
    s_x3 = sum(x_value * x_value * x_value for x_value in x_values)
    s_x4 = sum(x_value * x_value * x_value * x_value for x_value in x_values)

    s_y = sum(log_y_values)
    s_xy = sum(x_value * log_y for x_value, log_y in zip(x_values, log_y_values))
    s_x2y = sum(
        x_value * x_value * log_y for x_value, log_y in zip(x_values, log_y_values)
    )

    matrix = [
        [s_x4, s_x3, s_x2, s_x2y],
        [s_x3, s_x2, s_x, s_xy],
        [s_x2, s_x, n_points, s_y],
    ]

    for pivot_index in range(3):
        pivot = matrix[pivot_index][pivot_index]
        if abs(pivot) < 1e-15:
            raise ValueError("Quadratic log fit is singular")

        for col_index in range(pivot_index, 4):
            matrix[pivot_index][col_index] /= pivot

        for row_index in range(3):
            if row_index == pivot_index:
                continue

            factor = matrix[row_index][pivot_index]
            for col_index in range(pivot_index, 4):
                matrix[row_index][col_index] -= factor * matrix[pivot_index][col_index]

    a = matrix[0][3]
    b = matrix[1][3]
    c = matrix[2][3]
    return a, b, c


def evaluate_power_law(
    x_values: list[float], exponent: float, coefficient: float
) -> list[float]:
    return [coefficient * (x_value ** exponent) for x_value in x_values]


def evaluate_exponential(
    x_values: list[float], exponent: float, coefficient: float
) -> list[float]:
    return [coefficient * math.exp(exponent * x_value) for x_value in x_values]


def evaluate_log_quadratic(
    x_values: list[float], a: float, b: float, c: float
) -> list[float]:
    return [math.exp(a * x_value * x_value + b * x_value + c) for x_value in x_values]


def build_curve_from_model(
    x_values: list[float],
    evaluator,
    parameters: tuple[float, ...],
    n_points: int = 200,
) -> tuple[list[float], list[float]]:
    x_min = min(x_values)
    x_max = max(x_values)
    x_curve: list[float] = []
    for index in range(n_points):
        fraction = index / (n_points - 1)
        x_curve.append(x_min + fraction * (x_max - x_min))

    return x_curve, evaluator(x_curve, *parameters)


def log_rmse(y_true: list[float], y_pred: list[float]) -> float:
    if any(value <= 0.0 for value in y_true) or any(value <= 0.0 for value in y_pred):
        raise ValueError("Log RMSE requires positive values")

    squared_error_sum = 0.0
    for true_value, predicted_value in zip(y_true, y_pred):
        error = math.log10(predicted_value) - math.log10(true_value)
        squared_error_sum += error * error

    return math.sqrt(squared_error_sum / len(y_true))


def extract_index(csv_path: Path) -> int:
    return int(csv_path.stem.split("_")[-1])


def format_equation(exponent: float, coefficient: float) -> str:
    return f"y = {coefficient:.12g} * x^{exponent:.12g}"


def plot_dataset_with_fit(csv_path: Path, output_path: Path) -> None:
    configure_plot_style()
    x_values, y_values = load_xy(csv_path)
    exponent, coefficient = fit_power_law(x_values, y_values)
    x_line, y_line = build_fit_curve(x_values, exponent, coefficient)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x_values, y_values, s=14, alpha=0.75)
    ax.plot(x_line, y_line, color="crimson", linewidth=2.0)
    ax.set_xlabel(r"Volume fraction")
    ax.set_ylabel(r"Spectral norm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    input_paths = sorted(
        (
            path
            for path in RESULT_DIR.glob("spectralNorm_*.csv")
            if path.stem.split("_")[-1].isdigit()
        ),
        key=extract_index,
    )
    if not input_paths:
        raise FileNotFoundError("No spectralNorm_*.csv files found in results/")

    output_path = RESULT_DIR / "spectralNorm_line_fits.csv"
    intersection_fit_output_path = RESULT_DIR / "spectralNorm_intersection_fit_comparison.csv"
    plot_path = RESULT_DIR / "spectralNorm_intersection_plot.pdf"
    fitted_lines_plot_path = RESULT_DIR / "spectralNorm_fitted_lines.pdf"
    dataset_15_plot_path = RESULT_DIR / "spectralNorm_15_fit.pdf"
    particle_counts: list[int] = []
    intersections: list[float] = []
    fitted_lines: list[tuple[list[float], float, float]] = []

    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "nParticles",
                "exponent",
                "coefficient",
                "equation",
                "volumeFraction_at_spectralNorm_1",
            ]
        )

        for input_path in input_paths:
            x_values, y_values = load_xy(input_path)
            exponent, coefficient = fit_power_law(x_values, y_values)
            particle_count = extract_index(input_path)
            fitted_lines.append((x_values, exponent, coefficient))

            intersection = intersection_at_spectral_norm_1(exponent, coefficient)
            if intersection is not None:
                particle_counts.append(particle_count)
                intersections.append(intersection)

            writer.writerow(
                [
                    particle_count,
                    f"{exponent:.16g}",
                    f"{coefficient:.16g}",
                    format_equation(exponent, coefficient),
                    "" if intersection is None else f"{intersection:.16g}",
                ]
            )

    intersection_x_values = [float(value) for value in particle_counts]
    fit_candidates: list[tuple[str, tuple[float, ...], float, str]] = []

    power_exponent, power_coefficient = fit_power_law(
        intersection_x_values,
        intersections,
    )
    fit_candidates.append(
        (
            "power_law",
            (power_exponent, power_coefficient),
            log_rmse(
                intersections,
                evaluate_power_law(
                    intersection_x_values, power_exponent, power_coefficient
                ),
            ),
            f"y = {power_coefficient:.12g} * x^{power_exponent:.12g}",
        )
    )

    exp_exponent, exp_coefficient = fit_exponential(
        intersection_x_values,
        intersections,
    )
    fit_candidates.append(
        (
            "exponential",
            (exp_exponent, exp_coefficient),
            log_rmse(
                intersections,
                evaluate_exponential(
                    intersection_x_values, exp_exponent, exp_coefficient
                ),
            ),
            f"y = {exp_coefficient:.12g} * exp({exp_exponent:.12g} * x)",
        )
    )

    quad_a, quad_b, quad_c = fit_log_quadratic(
        intersection_x_values,
        intersections,
    )
    fit_candidates.append(
        (
            "log_quadratic",
            (quad_a, quad_b, quad_c),
            log_rmse(
                intersections,
                evaluate_log_quadratic(intersection_x_values, quad_a, quad_b, quad_c),
            ),
            f"y = exp({quad_a:.12g} * x^2 + {quad_b:.12g} * x + {quad_c:.12g})",
        )
    )

    best_model_name, best_parameters, best_rmse, best_equation = min(
        fit_candidates,
        key=lambda item: item[2],
    )

    with intersection_fit_output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["model", "log10_rmse", "equation", "selected"])
        for model_name, parameters, rmse, equation in fit_candidates:
            writer.writerow(
                [
                    model_name,
                    f"{rmse:.16g}",
                    equation,
                    "yes" if model_name == best_model_name else "no",
                ]
            )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(particle_counts, intersections, s=24, alpha=0.8)
    if best_model_name == "power_law":
        fit_particle_counts, fit_intersections = build_fit_curve(
            intersection_x_values,
            best_parameters[0],
            best_parameters[1],
        )
    elif best_model_name == "exponential":
        fit_particle_counts, fit_intersections = build_curve_from_model(
            intersection_x_values,
            evaluate_exponential,
            best_parameters,
        )
    else:
        fit_particle_counts, fit_intersections = build_curve_from_model(
            intersection_x_values,
            evaluate_log_quadratic,
            best_parameters,
        )

    ax.plot(fit_particle_counts, fit_intersections, color="crimson", linewidth=2.0)
    ax.set_xlabel(r"Number of particles")
    ax.set_ylabel(r"Volume fraction at spectral norm $= 1$")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = list(plt.cm.tab20.colors)
    for index, (x_values, exponent, coefficient) in enumerate(fitted_lines):
        x_line, y_line = build_fit_curve(x_values, exponent, coefficient)
        ax.plot(
            x_line,
            y_line,
            color=colors[index % len(colors)],
            linewidth=1.2,
            alpha=0.9,
        )

    ax.set_xlabel(r"Volume fraction")
    ax.set_ylabel(r"Spectral norm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fitted_lines_plot_path)
    plt.close(fig)

    plot_dataset_with_fit(RESULT_DIR / "spectralNorm_15.csv", dataset_15_plot_path)

    print(f"Wrote line-fit summary to {output_path}")
    print(f"Wrote intersection-fit comparison to {intersection_fit_output_path}")
    print(f"Selected intersection-fit model: {best_model_name} ({best_equation})")
    print(f"Wrote intersection plot to {plot_path}")
    print(f"Wrote fitted-lines plot to {fitted_lines_plot_path}")
    print(f"Wrote spectralNorm_15 fit plot to {dataset_15_plot_path}")


if __name__ == "__main__":
    main()
