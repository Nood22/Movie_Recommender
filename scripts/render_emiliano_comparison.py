"""Turn the matched Emiliano-split evaluation into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tears_training.artifacts import atomic_write_json, atomic_write_text, sha256_file


DATA_ROOT = PROJECT_ROOT / "TEARS_Project" / "Code4Neda" / "data_preprocessed" / "ml-1m"
MOVIES_PATH = PROJECT_ROOT / "TEARS_Project" / "Code4Neda" / "data" / "ml-1m" / "movies.dat"
MODEL_LABELS = {"current": "Current (ML-32M, zero-shot)", "emiliano": "Emiliano (ML-1M)"}
MODEL_COLORS = {"current": "#0072B2", "emiliano": "#D55E00"}
METRICS = [
    f"{metric}@{k}"
    for k in (20, 50)
    for metric in ("recall", "ndcg", "precision", "map", "mrr", "hit_rate")
]


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    clean = [[str(cell).replace("|", "\\|") for cell in row] for row in rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in clean)
    return "\n".join(lines)


def latex_table(headers: list[str], rows: Iterable[Iterable[object]], caption: str, label: str) -> str:
    rows = list(rows)
    alignment = "l" + "r" * (len(headers) - 1)
    body = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}"]
    if len(headers) > 5:
        body.append(r"\resizebox{\linewidth}{!}{%")
    body.extend([rf"\begin{{tabular}}{{{alignment}}}", r"\toprule"])
    body.append(" & ".join(latex_escape(value) for value in headers) + r" \\")
    body.append(r"\midrule")
    body.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if len(headers) > 5:
        body.append("}")
    body.extend([r"\end{table}", ""])
    return "\n".join(body)


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 2024, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = a - b
    sampled = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        sampled[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


class Plot:
    """Small dependency-free dual SVG/PNG drawing surface."""

    def __init__(self, width: int = 900, height: int = 560):
        self.width, self.height, self.scale = width, height, 2
        self.image = Image.new("RGB", (width * self.scale, height * self.scale), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default(size=15 * self.scale)
        self.small = ImageFont.load_default(size=12 * self.scale)
        self.svg: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']

    def line(self, points: list[tuple[float, float]], color: str, width: float = 2) -> None:
        scaled = [(round(x * self.scale), round(y * self.scale)) for x, y in points]
        self.draw.line(scaled, fill=color, width=max(1, round(width * self.scale)), joint="curve")
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, box: tuple[float, float, float, float], fill: str, outline: str | None = None) -> None:
        x0, y0, x1, y1 = box
        self.draw.rectangle(tuple(round(v * self.scale) for v in box), fill=fill, outline=outline)
        stroke = f' stroke="{outline}"' if outline else ""
        self.svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" fill="{fill}"{stroke}/>')

    def text(self, xy: tuple[float, float], value: str, size: int = 15, anchor: str = "la", color: str = "#222222") -> None:
        x, y = xy
        pil_anchor = {"la": "la", "ma": "ma", "ra": "ra", "mm": "mm"}.get(anchor, "la")
        font = self.font if size >= 14 else self.small
        self.draw.text((x * self.scale, y * self.scale), value, fill=color, font=font, anchor=pil_anchor)
        svg_anchor = {"la": "start", "ma": "middle", "ra": "end", "mm": "middle"}.get(anchor, "start")
        baseline = "middle" if anchor == "mm" else "auto"
        self.svg.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="{size}" text-anchor="{svg_anchor}" dominant-baseline="{baseline}" fill="{color}">{escape(value)}</text>')

    def circle(self, xy: tuple[float, float], radius: float, fill: str) -> None:
        x, y = xy
        self.draw.ellipse(((x-radius)*self.scale, (y-radius)*self.scale, (x+radius)*self.scale, (y+radius)*self.scale), fill=fill)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>')

    def save(self, stem: Path) -> None:
        self.svg.append("</svg>")
        atomic_write_text(stem.with_suffix(".svg"), "\n".join(self.svg) + "\n")
        self.image.save(stem.with_suffix(".png"), dpi=(300, 300))


def axes(plot: Plot, title: str, x_label: str, y_label: str, x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]]) -> tuple[int, int, int, int]:
    left, top, right, bottom = 90, 55, plot.width - 35, plot.height - 75
    plot.text((plot.width / 2, 24), title, 15, "ma")
    plot.line([(left, top), (left, bottom), (right, bottom)], "#333333", 1.2)
    for y, label in y_ticks:
        plot.line([(left, y), (right, y)], "#DDDDDD", 0.8)
        plot.text((left - 10, y), label, 12, "ra")
    for x, label in x_ticks:
        plot.line([(x, bottom), (x, bottom + 5)], "#333333", 1)
        plot.text((x, bottom + 24), label, 12, "ma")
    plot.text(((left + right) / 2, plot.height - 20), x_label, 14, "ma")
    # Horizontal caption above the y axis stays legible in both SVG and PNG
    # without relying on font rotation support.
    plot.text((left, top - 10), y_label, 14, "la")
    return left, top, right, bottom


def alpha_plot(report: dict, output_dir: Path) -> None:
    series = {}
    for model in MODEL_LABELS:
        sweep = report["validation"][model]["metrics_by_alpha"]
        series[model] = [(float(alpha), values["ndcg@50"]) for alpha, values in sweep.items()]
    all_y = [y for values in series.values() for _, y in values]
    y_min, y_max = min(all_y), max(all_y)
    pad = max((y_max - y_min) * 0.12, 0.002)
    y_min, y_max = y_min - pad, y_max + pad
    plot = Plot()
    left, top, right, bottom = 90, 55, 865, 485
    xmap = lambda x: left + x * (right - left)
    ymap = lambda y: bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
    bounds = axes(plot, "Validation alpha sweep (selection metric)", "alpha (0 = RecVAE, 1 = text)", "NDCG@50", [(xmap(x), f"{x:.2f}") for x in np.linspace(0, 1, 6)], [(ymap(y), f"{y:.3f}") for y in np.linspace(y_min, y_max, 6)])
    del bounds
    for model, values in series.items():
        points = [(xmap(x), ymap(y)) for x, y in values]
        plot.line(points, MODEL_COLORS[model], 2.5)
        selected = float(report["test"]["models"][model]["selected_alpha"])
        chosen = min(values, key=lambda item: abs(item[0] - selected))
        plot.circle((xmap(chosen[0]), ymap(chosen[1])), 5, MODEL_COLORS[model])
    for index, model in enumerate(MODEL_LABELS):
        y = 74 + index * 24
        plot.line([(620, y), (650, y)], MODEL_COLORS[model], 3)
        plot.text((660, y + 5), MODEL_LABELS[model], 12)
    plot.save(output_dir / "figure_alpha_sweep")


def ranking_plot(report: dict, output_dir: Path) -> None:
    shown = ["recall@20", "ndcg@20", "map@20", "mrr@20", "recall@50", "ndcg@50", "map@50", "mrr@50"]
    maximum = max(report["test"]["models"][model]["metrics"][metric] for model in MODEL_LABELS for metric in shown) * 1.15
    plot = Plot(1050, 600)
    left, top, right, bottom = 90, 55, 1015, 525
    ymap = lambda y: bottom - y / maximum * (bottom - top)
    centers = np.linspace(left + 55, right - 55, len(shown))
    axes(plot, "Matched held-out test ranking quality", "Metric", "Score", [(float(x), metric.replace("hit_rate", "hit")) for x, metric in zip(centers, shown)], [(ymap(y), f"{y:.2f}") for y in np.linspace(0, maximum, 6)])
    width = 34
    for offset, model in zip((-width / 2, width / 2), MODEL_LABELS):
        for x, metric in zip(centers, shown):
            value = report["test"]["models"][model]["metrics"][metric]
            plot.rect((x + offset - width / 2, ymap(value), x + offset + width / 2, bottom), MODEL_COLORS[model])
    for index, model in enumerate(MODEL_LABELS):
        plot.rect((720, 73 + index * 24, 738, 87 + index * 24), MODEL_COLORS[model])
        plot.text((746, 86 + index * 24), MODEL_LABELS[model], 12)
    plot.save(output_dir / "figure_test_ranking")


def paired_histogram(current: np.ndarray, emiliano: np.ndarray, output_dir: Path) -> None:
    difference = current - emiliano
    bins = np.linspace(float(difference.min()), float(difference.max()), 20)
    if np.allclose(bins[0], bins[-1]):
        bins = np.linspace(bins[0] - 0.01, bins[0] + 0.01, 20)
    counts, edges = np.histogram(difference, bins=bins)
    plot = Plot()
    left, top, right, bottom = 90, 55, 865, 485
    xmap = lambda x: left + (x - edges[0]) / (edges[-1] - edges[0]) * (right - left)
    ymap = lambda y: bottom - y / max(counts.max(), 1) * (bottom - top)
    axes(plot, "Per-user paired NDCG@50 difference", "Current minus Emiliano", "Users", [(xmap(x), f"{x:+.2f}") for x in np.linspace(edges[0], edges[-1], 6)], [(ymap(y), str(int(y))) for y in np.linspace(0, counts.max(), 6)])
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        plot.rect((xmap(x0) + 1, ymap(count), xmap(x1) - 1, bottom), "#56B4E9")
    if edges[0] <= 0 <= edges[-1]:
        plot.line([(xmap(0), top), (xmap(0), bottom)], "#222222", 1.5)
    plot.save(output_dir / "figure_paired_ndcg50")


def efficiency_plot(report: dict, output_dir: Path) -> None:
    plot = Plot()
    values = {model: (report["test"]["models"][model]["resources"]["users_per_second"], report["test"]["models"][model]["metrics"]["ndcg@50"]) for model in MODEL_LABELS}
    xs, ys = zip(*values.values())
    xmin, xmax = min(xs) * 0.85, max(xs) * 1.15
    ymin, ymax = min(ys) - 0.01, max(ys) + 0.01
    left, top, right, bottom = 90, 55, 865, 485
    xmap = lambda x: left + (x - xmin) / max(xmax - xmin, 1e-9) * (right - left)
    ymap = lambda y: bottom - (y - ymin) / max(ymax - ymin, 1e-9) * (bottom - top)
    axes(plot, "Test quality–throughput trade-off", "Users per second (single GPU)", "NDCG@50", [(xmap(x), f"{x:.1f}") for x in np.linspace(xmin, xmax, 6)], [(ymap(y), f"{y:.3f}") for y in np.linspace(ymin, ymax, 6)])
    for model, (x, y) in values.items():
        plot.circle((xmap(x), ymap(y)), 8, MODEL_COLORS[model])
        plot.text((xmap(x) + 12, ymap(y) - 9), MODEL_LABELS[model], 12)
    plot.save(output_dir / "figure_quality_throughput")


def load_item_metadata(shared_movies: list[int]) -> tuple[list[set[str]], np.ndarray]:
    with (DATA_ROOT / "show2id.pkl").open("rb") as handle:
        movie_to_item = {int(movie): int(item) for movie, item in pickle.load(handle).items()}
    genre_by_movie: dict[int, set[str]] = {}
    with MOVIES_PATH.open("r", encoding="latin-1") as handle:
        for line in handle:
            movie, _, genres = line.rstrip("\n").split("::")
            genre_by_movie[int(movie)] = set(genres.split("|"))
    train = np.genfromtxt(DATA_ROOT / "train.csv", delimiter=",", names=True)
    counts = np.zeros(max(movie_to_item.values()) + 1, dtype=np.float64)
    for item in train["sid"].astype(int):
        counts[item] += 1
    probabilities = (counts + 1.0) / (counts.sum() + len(counts))
    return [genre_by_movie.get(movie, {"Unknown"}) for movie in shared_movies], np.asarray([probabilities[movie_to_item[movie]] for movie in shared_movies])


def beyond_accuracy(report: dict) -> dict[str, dict[str, float]]:
    shared_movies = [int(movie) for movie in report["candidate_catalog"]["shared_movie_ids"]]
    genres, popularity = load_item_metadata(shared_movies)
    results: dict[str, dict[str, float]] = {}
    for model in MODEL_LABELS:
        recommendations = report["test"]["models"][model]["per_user"]
        model_result: dict[str, float] = {}
        for k in (20, 50):
            recs = [row["recommendations"][:k] for row in recommendations]
            unique = {item for row in recs for item in row}
            diversities = []
            novelty = []
            mean_popularity = []
            for row in recs:
                distances = []
                for first in range(len(row)):
                    for second in range(first + 1, len(row)):
                        union = genres[row[first]] | genres[row[second]]
                        distances.append(1.0 - len(genres[row[first]] & genres[row[second]]) / max(len(union), 1))
                diversities.append(float(np.mean(distances)))
                novelty.append(float(np.mean(-np.log2(popularity[row]))))
                mean_popularity.append(float(np.mean(popularity[row])))
            model_result[f"catalog_coverage@{k}"] = len(unique) / len(shared_movies)
            model_result[f"genre_diversity@{k}"] = float(np.mean(diversities))
            model_result[f"novelty_bits@{k}"] = float(np.mean(novelty))
            model_result[f"mean_train_probability@{k}"] = float(np.mean(mean_popularity))
        results[model] = model_result
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input.read_text(encoding="utf-8"))

    test_models = report["test"]["models"]
    ranking_rows = []
    for metric in METRICS:
        current = float(test_models["current"]["metrics"][metric])
        old = float(test_models["emiliano"]["metrics"][metric])
        ranking_rows.append([metric, fmt(current), fmt(old), f"{current-old:+.4f}", f"{(current-old)/old*100:+.2f}%" if old else "n/a"])
    write_csv(args.output_dir / "table_ranking_metrics.csv", ["Metric", "Current", "Emiliano", "Absolute delta", "Relative delta"], ranking_rows)

    current_rows = {int(row["user_id"]): row for row in test_models["current"]["per_user"] if row["eligible"]}
    old_rows = {int(row["user_id"]): row for row in test_models["emiliano"]["per_user"] if row["eligible"]}
    paired_users = sorted(set(current_rows) & set(old_rows))
    significance = []
    raw_p = []
    paired_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in METRICS:
        current = np.asarray([current_rows[user][metric] for user in paired_users])
        old = np.asarray([old_rows[user][metric] for user in paired_users])
        paired_arrays[metric] = current, old
        low, high = bootstrap_difference(current, old)
        t_result = stats.ttest_rel(current, old)
        try:
            wilcoxon_p = float(stats.wilcoxon(current, old, zero_method="zsplit").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        raw_p.append(wilcoxon_p)
        significance.append([metric, len(current), float((current-old).mean()), low, high, float(t_result.pvalue), wilcoxon_p])
    adjusted = holm_adjust(raw_p)
    significance_rows = [[row[0], row[1], fmt(row[2]), f"[{row[3]:+.4f}, {row[4]:+.4f}]", f"{row[5]:.3g}", f"{row[6]:.3g}", f"{adjusted[index]:.3g}", "yes" if adjusted[index] < 0.05 else "no"] for index, row in enumerate(significance)]
    write_csv(args.output_dir / "table_paired_significance.csv", ["Metric", "Paired users", "Mean delta", "Bootstrap 95% CI", "Paired t p", "Wilcoxon p", "Holm-adjusted p", "Significant at .05"], significance_rows)

    selection = json.loads(Path(report["frozen_selection"]).read_text(encoding="utf-8"))
    resource_rows = []
    for model in MODEL_LABELS:
        resource = test_models[model]["resources"]
        checkpoint = selection["checkpoints"][model]
        resource_rows.append([MODEL_LABELS[model], f"{resource['parameter_count']:,}", f"{resource['trainable_parameter_count']:,}", f"{checkpoint['bytes']/2**20:.1f}", fmt(resource["inference_seconds"], 2), fmt(resource["load_plus_inference_seconds"], 2), fmt(resource["users_per_second"], 2), fmt(1000/resource["users_per_second"], 2), f"{resource['peak_gpu_memory_bytes']/2**30:.2f}"])
    resource_headers = ["Model", "Parameters", "Requires-grad parameters", "Checkpoint MiB", "Test inference seconds", "Cold load + inference seconds", "Users/s", "ms/user", "Peak allocated GiB"]
    write_csv(args.output_dir / "table_resources.csv", resource_headers, resource_rows)

    token_rows = []
    for model in MODEL_LABELS:
        resource = test_models[model]["resources"]
        utilization = resource["unpadded_t5_tokens"] / max(resource["padded_t5_token_positions"], 1)
        token_rows.append([MODEL_LABELS[model], resource["max_text_tokens"], f"{resource['unpadded_t5_tokens']:,}", fmt(resource["unpadded_tokens_per_user"], 2), f"{resource['padded_t5_token_positions']:,}", f"{utilization*100:.2f}%"])
    token_headers = ["Model", "Max sequence", "Total unpadded tokens", "Unpadded tokens/user", "Padded token positions", "Token utilization"]
    write_csv(args.output_dir / "table_token_usage.csv", token_headers, token_rows)

    beyond = beyond_accuracy(report)
    beyond_rows = [[metric, *[fmt(beyond[model][metric]) for model in MODEL_LABELS], f"{beyond['current'][metric]-beyond['emiliano'][metric]:+.4f}"] for metric in beyond["current"]]
    write_csv(args.output_dir / "table_beyond_accuracy.csv", ["Metric", "Current", "Emiliano", "Absolute delta"], beyond_rows)

    protocol_rows = [
        ["Evaluation dataset", "Emiliano ML-1M validation/test split"],
        ["Candidate items", report["candidate_catalog"]["shared_items"]],
        ["Catalog retention", f"{report['candidate_catalog']['shared_items']/report['candidate_catalog']['emiliano_items']*100:.2f}%"],
        ["Validation users", report["validation"]["current"]["metrics_by_alpha"]["0.000"]["users"]],
        ["Test users", report["test"]["users"]],
        ["Test observed interactions", report["test"]["shared_observed_interactions"]],
        ["Test positive targets", report["test"]["shared_positive_targets"]],
        ["Dropped test observed rows", report["test"]["dropped_observed_rows"]],
        ["Dropped test target rows", report["test"]["dropped_target_rows"]],
        ["Selection", "Per-model alpha maximizing validation NDCG@50"],
        ["Alpha grid", "0.000 to 1.000 inclusive, step 0.025"],
        ["Current alpha", test_models["current"]["selected_alpha"]],
        ["Emiliano alpha", test_models["emiliano"]["selected_alpha"]],
        ["Positive relevance", "rating >= 4"],
        ["Seen-item masking", "Yes"],
        ["Inference", "Posterior means; deterministic"],
    ]
    write_csv(args.output_dir / "table_protocol.csv", ["Field", "Value"], protocol_rows)
    selection_rows = []
    for model in MODEL_LABELS:
        alpha = float(test_models[model]["selected_alpha"])
        validation_metric = report["validation"][model]["metrics_by_alpha"][f"{alpha:.3f}"]["ndcg@50"]
        selection_rows.append([MODEL_LABELS[model], fmt(alpha, 3), fmt(validation_metric), fmt(test_models[model]["metrics"]["ndcg@50"])])
    selection_headers = ["Model", "Selected alpha", "Validation NDCG@50", "Test NDCG@50"]
    write_csv(args.output_dir / "table_model_selection.csv", selection_headers, selection_rows)

    alpha_plot(report, args.output_dir)
    ranking_plot(report, args.output_dir)
    paired_histogram(*paired_arrays["ndcg@50"], args.output_dir)
    efficiency_plot(report, args.output_dir)

    tables_tex = "% Requires \\usepackage{booktabs,graphicx}.\n" + "\n".join([
        latex_table(["Metric", "Current", "Emiliano", "Difference", "Difference (%)"], ranking_rows, "Matched test ranking metrics. Current is evaluated zero-shot on Emiliano's ML-1M split.", "tab:matched-ranking"),
        latex_table(["Metric", "n", "Mean difference", "Bootstrap 95% CI", "Paired t p", "Wilcoxon p", "Holm p", "Sig."], significance_rows, "Paired per-user tests; confidence intervals use 10,000 paired bootstrap resamples and Wilcoxon p-values use Holm correction.", "tab:paired-tests"),
        latex_table(selection_headers, selection_rows, "Validation-only interpolation selection and corresponding held-out test result.", "tab:model-selection"),
        latex_table(["Model", "Params", "Trainable", "Ckpt MiB", "Inference s", "Cold total s", "Users/s", "ms/user", "Peak GiB"], resource_rows, "Single-GPU inference resource comparison on the matched test split.", "tab:resources"),
        latex_table(token_headers, token_rows, "Tokenizer workload on the matched test split.", "tab:tokens"),
        latex_table(["Metric", "Current", "Emiliano", "Difference"], beyond_rows, "Beyond-accuracy metrics on matched recommendations.", "tab:beyond-accuracy"),
        latex_table(["Field", "Value"], protocol_rows, "Matched comparison protocol.", "tab:protocol"),
    ])
    atomic_write_text(args.output_dir / "paper_tables.tex", tables_tex)

    ndcg_current = test_models["current"]["metrics"]["ndcg@50"]
    ndcg_old = test_models["emiliano"]["metrics"]["ndcg@50"]
    ndcg_sig = significance_rows[METRICS.index("ndcg@50")]
    winner = "current model" if ndcg_current > ndcg_old else "Emiliano model"
    report_md = f"""# Matched TEARS model comparison on Emiliano's ML-1M split

Generated {datetime.now(timezone.utc).isoformat()} from `{args.input.name}`.

## Executive result

The {winner} has the higher held-out NDCG@50 ({ndcg_current:.4f} current versus {ndcg_old:.4f} Emiliano; paired mean difference {ndcg_sig[2]}, bootstrap 95% CI {ndcg_sig[3]}, Holm-adjusted Wilcoxon p={ndcg_sig[6]}). This is a matched checkpoint comparison, not a claim about architecture alone: the current checkpoint was trained on ML-32M and is evaluated zero-shot, while Emiliano's supplied checkpoint was trained on ML-1M.

## Protocol

{markdown_table(["Field", "Value"], protocol_rows)}

Both checkpoints receive the identical shared observed histories and GPT-4 summaries. Scoring is restricted to the {report['candidate_catalog']['shared_items']:,}-movie intersection ({report['candidate_catalog']['shared_items']/report['candidate_catalog']['emiliano_items']*100:.2f}% of Emiliano's catalog), observed items are masked, and relevance is rating >= 4. Alpha is selected independently for each checkpoint on validation NDCG@50 and persisted before the test files are loaded. Alpha follows the paper convention: 0 is RecVAE-only and 1 is text-only.

{markdown_table(selection_headers, selection_rows)}

## Held-out ranking metrics

{markdown_table(["Metric", "Current", "Emiliano", "Absolute delta", "Relative delta"], ranking_rows)}

## Paired uncertainty and significance

{markdown_table(["Metric", "Paired users", "Mean delta", "Bootstrap 95% CI", "Paired t p", "Wilcoxon p", "Holm-adjusted p", "Significant at .05"], significance_rows)}

Differences are current minus Emiliano. Confidence intervals use 10,000 seeded paired bootstrap resamples. Wilcoxon signed-rank p-values are corrected jointly across the 12 reported metrics using Holm's family-wise procedure; paired t-tests are included as a parametric sensitivity check.

## Beyond-accuracy behavior

{markdown_table(["Metric", "Current", "Emiliano", "Absolute delta"], beyond_rows)}

Catalog coverage is the fraction of shared candidates recommended at least once. Genre diversity is mean pairwise Jaccard distance within each list. Novelty is self-information, -log2 of Laplace-smoothed ML-1M training popularity; larger values mean less-popular recommendations.

## Token, memory, model-size, and throughput comparison

{markdown_table(resource_headers, resource_rows)}

{markdown_table(token_headers, token_rows)}

Runtime and GPU memory are measured in the same scheduled process and hardware allocation. Token counts are tokenizer-dependent and therefore reflect actual model input cost, while inference time includes tokenization and model scoring but excludes checkpoint loading. Peak memory is PyTorch allocated memory, not whole-device residency.

## Figures

1. `figure_alpha_sweep.svg/png`: validation selection curves.
2. `figure_test_ranking.svg/png`: matched ranking metrics.
3. `figure_paired_ndcg50.svg/png`: distribution of per-user paired effects.
4. `figure_quality_throughput.svg/png`: quality–throughput trade-off.

## Interpretation limits

- The current model is not retrained on ML-1M, so domain/scale transfer is intentionally part of its result.
- Sixteen Emiliano-catalog movies are unavailable to the current support-20 catalog; all scores use the 2,729-item intersection.
- This run compares one supplied checkpoint per system. It does not replace a multi-seed training comparison.
- Parameter count, checkpoint size, and speed describe these concrete implementations and settings, not only their conceptual architectures.

## Reproducibility

- Input SHA-256: `{sha256_file(args.input)}`
- Frozen selection SHA-256: `{report['frozen_selection_sha256']}`
- Device: `{report['reproducibility']['gpu']}`
- Seed: `{report['reproducibility']['seed']}`
- Python/Torch/Transformers: `{report['reproducibility']['python']}` / `{report['reproducibility']['torch']}` / `{report['reproducibility']['transformers']}`
- Slurm job: `{report['reproducibility']['slurm_job_id']}`
"""
    atomic_write_text(args.output_dir / "paper_ready_report.md", report_md)

    artifacts = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "comparison_manifest.json")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "artifacts": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts],
    }
    atomic_write_json(args.output_dir / "comparison_manifest.json", manifest)
    print(f"Rendered {len(artifacts)} paper-ready artifacts in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
