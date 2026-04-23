from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

PLOT_BG = "#0b1424"
PANEL_BG = "#101a2c"
GRID_COLOR = "#28405f"
ACCENT = "#19e3ff"
ACCENT_ALT = "#f5c65b"
TEXT = "#eef4ff"
MUTED = "#92a7c4"
RISK_RED = "#ff6b78"
SAFE_GREEN = "#4cd7a7"
BAR_BLUE = "#5aa8ff"
BAR_PURPLE = "#8ba8ff"


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 5.0, height: float = 3.0) -> None:
        self.figure = Figure(figsize=(width, height), dpi=100, facecolor=PANEL_BG)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setMinimumHeight(240)
        self.figure.tight_layout()


def style_axes(ax, title: str = "") -> None:
    ax.set_facecolor(PLOT_BG)
    ax.grid(True, color=GRID_COLOR, alpha=0.28, linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)


def draw_empty(ax, title: str, message: str) -> None:
    ax.clear()
    style_axes(ax, title)
    ax.text(0.5, 0.5, message, color=MUTED, ha="center", va="center", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_loss_curve(ax, train_loss: list[float], val_loss: list[float]) -> None:
    ax.clear()
    style_axes(ax, "\u8bad\u7ec3\u635f\u5931\u66f2\u7ebf")
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=ACCENT, linewidth=2.2, label="训练集损失")
    ax.plot(epochs, val_loss, color=ACCENT_ALT, linewidth=2.2, label="验证集损失")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("损失值")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT)


def draw_confusion_matrix(ax, confusion_matrix_values: list[list[int]]) -> None:
    ax.clear()
    style_axes(ax, "\u6df7\u6dc6\u77e9\u9635")
    matrix = np.array(confusion_matrix_values)
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    labels = ["\u6b63\u5e38", "\u6b3a\u8bc8"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("\u9884\u6d4b\u6807\u7b7e")
    ax.set_ylabel("\u771f\u5b9e\u6807\u7b7e")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=TEXT, fontsize=11)


def draw_roc_curve(ax, fpr: list[float], tpr: list[float], auc_score: float | None) -> None:
    ax.clear()
    style_axes(ax, "ROC \u66f2\u7ebf")
    if not fpr or not tpr:
        draw_empty(ax, "ROC \u66f2\u7ebf", "\u5f53\u524d\u6570\u636e\u65e0\u6cd5\u8ba1\u7b97 ROC-AUC")
        return
    label = f"ROC-AUC = {auc_score:.4f}" if auc_score is not None else "ROC"
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.2, label=label)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT, loc="lower right")


def draw_pr_curve(ax, recall: list[float], precision: list[float], auc_score: float | None) -> None:
    ax.clear()
    style_axes(ax, "PR \u66f2\u7ebf")
    if not recall or not precision:
        draw_empty(ax, "PR \u66f2\u7ebf", "\u5f53\u524d\u6570\u636e\u65e0\u6cd5\u8ba1\u7b97 PR-AUC")
        return
    label = f"PR-AUC = {auc_score:.4f}" if auc_score is not None else "PR"
    ax.plot(recall, precision, color=ACCENT_ALT, linewidth=2.2, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT, loc="lower left")


def draw_class_distribution(ax, actual_counts: dict[str, int], predicted_counts: dict[str, int] | None = None) -> None:
    ax.clear()
    style_axes(ax, "\u7c7b\u522b\u5206\u5e03")
    categories = list(actual_counts.keys())
    indices = np.arange(len(categories))
    width = 0.35
    actual_values = [actual_counts[key] for key in categories]
    ax.bar(indices - width / 2, actual_values, width=width, color=SAFE_GREEN, label="\u771f\u5b9e")
    if predicted_counts:
        predicted_values = [predicted_counts.get(key, 0) for key in categories]
        ax.bar(indices + width / 2, predicted_values, width=width, color=ACCENT_ALT, label="\u9884\u6d4b")
    ax.set_xticks(indices, labels=categories)
    ax.set_ylabel("\u6837\u672c\u6570")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT)


def draw_bar_chart(ax, labels: list[str], values: list[float], title: str, ylabel: str, color: str = ACCENT) -> None:
    ax.clear()
    style_axes(ax, title)
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=color, alpha=0.92)
    ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)


def draw_dual_bar_chart(ax, labels: list[str], left_values: list[float], right_values: list[float], title: str, left_label: str, right_label: str) -> None:
    ax.clear()
    style_axes(ax, title)
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, left_values, width=width, color=BAR_BLUE, label=left_label)
    ax.bar(positions + width / 2, right_values, width=width, color=ACCENT_ALT, label=right_label)
    ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT)


def draw_heatmap(ax, matrix: np.ndarray, labels: list[str], title: str) -> None:
    ax.clear()
    style_axes(ax, title)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.grid(False)


def draw_embedding(ax, x: list[float], y: list[float], labels: list[int], title: str, xlabel: str, ylabel: str) -> None:
    ax.clear()
    style_axes(ax, title)
    points_x = np.asarray(x)
    points_y = np.asarray(y)
    label_array = np.asarray(labels)
    normal_mask = label_array == 0
    fraud_mask = label_array == 1
    ax.scatter(points_x[normal_mask], points_y[normal_mask], s=12, alpha=0.55, c=SAFE_GREEN, label="\u6b63\u5e38")
    ax.scatter(points_x[fraud_mask], points_y[fraud_mask], s=14, alpha=0.72, c=RISK_RED, label="\u6b3a\u8bc8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT)


def draw_histogram_by_class(ax, normal_values: list[float], fraud_values: list[float], title: str, xlabel: str) -> None:
    ax.clear()
    style_axes(ax, title)
    ax.hist(normal_values, bins=32, alpha=0.58, color=SAFE_GREEN, label="\u6b63\u5e38")
    if fraud_values:
        ax.hist(fraud_values, bins=32, alpha=0.58, color=RISK_RED, label="\u6b3a\u8bc8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("\u6837\u672c\u6570")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT)


def draw_model_metric_bars(ax, model_names: list[str], metrics: list[float], title: str, ylabel: str) -> None:
    ax.clear()
    style_axes(ax, title)
    positions = np.arange(len(model_names))
    colors = [ACCENT, ACCENT_ALT, BAR_BLUE, BAR_PURPLE, SAFE_GREEN, RISK_RED][: len(model_names)]
    ax.bar(positions, metrics, color=colors)
    ax.set_xticks(positions, labels=model_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(ylabel)


def draw_curve_collection(ax, curves: dict[str, dict[str, list[float] | float | None]], curve_type: str) -> None:
    title = "ROC \u5bf9\u6bd4" if curve_type == "roc" else "PR \u5bf9\u6bd4"
    ax.clear()
    style_axes(ax, title)
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.0)
    palette = [ACCENT, ACCENT_ALT, BAR_BLUE, BAR_PURPLE, SAFE_GREEN, RISK_RED]
    for idx, (name, payload) in enumerate(curves.items()):
        color = palette[idx % len(palette)]
        if curve_type == "roc":
            x = payload.get("fpr", [])
            y = payload.get("tpr", [])
            score = payload.get("auc")
            if not x or not y:
                continue
            label = f"{name} ({score:.3f})" if score is not None else name
            ax.plot(x, y, linewidth=2.0, color=color, label=label)
            ax.set_xlabel("假阳性率")
            ax.set_ylabel("真阳性率")
        else:
            x = payload.get("recall", [])
            y = payload.get("precision", [])
            score = payload.get("auc")
            if not x or not y:
                continue
            label = f"{name} ({score:.3f})" if score is not None else name
            ax.plot(x, y, linewidth=2.0, color=color, label=label)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT, fontsize=9)

