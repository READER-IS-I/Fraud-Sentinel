from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QSpinBox, QVBoxLayout, QWidget

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, create_path_row, labeled_value, show_error, show_info
from src.utils.file_utils import DATA_DIR, DEMO_DIR, get_dialog_start_dir
from src.utils.plot_utils import PlotCanvas, draw_bar_chart, draw_class_distribution, draw_embedding, draw_empty, draw_heatmap, draw_histogram_by_class


def _default_csv() -> str:
    final_csv = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
    return str(final_csv) if final_csv.exists() else str(DEMO_DIR / "sample_demo.csv")


class AnalysisPage(QWidget):
    page_title = "\u6570\u636e\u5206\u6790"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: AnalysisWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        control_card = CardFrame("\u5206\u6790\u63a7\u5236\u53f0", "\u6309\u8bf4\u660e\u4e66\u8865\u5145\u63a2\u7d22\u6027\u5206\u6790\uff1a\u7c7b\u522b\u5206\u5e03\u3001\u4ea4\u6613\u7c7b\u578b\u3001\u964d\u7ef4\u53ef\u89c6\u5316\u4e0e\u76f8\u5173\u6027\u5206\u6790\u3002")
        self.csv_edit = QLineEdit(_default_csv())
        browse_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        browse_button.clicked.connect(self.browse_csv)
        control_card.layout.addWidget(create_path_row("\u6570\u636e\u6587\u4ef6", self.csv_edit, browse_button))
        row = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(2000, 50000)
        self.sample_spin.setSingleStep(2000)
        self.sample_spin.setValue(20000)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(42)
        analyze_button = HighlightButton("\u5f00\u59cb\u5206\u6790", variant="primary")
        analyze_button.clicked.connect(self.start_analysis)
        row.addWidget(QLabel("\u62bd\u6837\u89c4\u6a21"))
        row.addWidget(self.sample_spin)
        row.addSpacing(16)
        row.addWidget(QLabel("\u968f\u673a\u79cd\u5b50"))
        row.addWidget(self.seed_spin)
        row.addStretch(1)
        row.addWidget(analyze_button)
        control_card.layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        control_card.layout.addWidget(self.progress)
        root.addWidget(control_card)

        summary_card = CardFrame("\u5206\u6790\u6982\u89c8")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.schema_tile = labeled_value("\u6570\u636e\u7ed3\u6784")
        self.total_tile = labeled_value("\u603b\u6837\u672c\u91cf")
        self.sample_tile = labeled_value("\u5206\u6790\u6837\u672c")
        self.fraud_tile = labeled_value("\u6b3a\u8bc8\u6837\u672c")
        for index, tile in enumerate([self.schema_tile, self.total_tile, self.sample_tile, self.fraud_tile]):
            grid.addWidget(tile, 0, index)
        summary_card.layout.addLayout(grid)
        root.addWidget(summary_card)

        self.class_canvas = PlotCanvas(width=4.8, height=2.8)
        self.type_canvas = PlotCanvas(width=4.8, height=2.8)
        self.rate_canvas = PlotCanvas(width=4.8, height=2.8)
        self.amount_canvas = PlotCanvas(width=4.8, height=2.8)
        self.pca_canvas = PlotCanvas(width=4.8, height=2.8)
        self.tsne_canvas = PlotCanvas(width=4.8, height=2.8)
        self.corr_canvas = PlotCanvas(width=4.8, height=2.8)
        self.var_canvas = PlotCanvas(width=4.8, height=2.8)
        for canvas, title in [(self.class_canvas, "\u7c7b\u522b\u5206\u5e03"), (self.type_canvas, "\u4ea4\u6613\u7c7b\u578b\u5206\u5e03"), (self.rate_canvas, "\u5404\u7c7b\u578b\u6b3a\u8bc8\u7387"), (self.amount_canvas, "\u91d1\u989d\u5206\u5e03"), (self.pca_canvas, "PCA \u964d\u7ef4"), (self.tsne_canvas, "t-SNE \u964d\u7ef4"), (self.corr_canvas, "\u76f8\u5173\u6027\u70ed\u529b\u56fe"), (self.var_canvas, "PCA \u89e3\u91ca\u65b9\u5dee")]:
            draw_empty(canvas.axes, title, "\u7b49\u5f85\u5206\u6790\u7ed3\u679c")
            canvas.draw()

        plot_grid = QGridLayout()
        plot_grid.setSpacing(14)
        cards = [("\u7c7b\u522b\u5206\u5e03", self.class_canvas), ("\u4ea4\u6613\u7c7b\u578b\u5206\u5e03", self.type_canvas), ("\u5404\u7c7b\u578b\u6b3a\u8bc8\u7387", self.rate_canvas), ("\u91d1\u989d\u5206\u5e03 (Log1p)", self.amount_canvas), ("PCA \u964d\u7ef4\u6563\u70b9", self.pca_canvas), ("t-SNE \u964d\u7ef4\u6563\u70b9", self.tsne_canvas), ("\u76f8\u5173\u6027\u70ed\u529b\u56fe", self.corr_canvas), ("PCA \u89e3\u91ca\u65b9\u5dee", self.var_canvas)]
        for index, (title, canvas) in enumerate(cards):
            card = CardFrame(title)
            card.layout.addWidget(canvas)
            plot_grid.addWidget(card, index // 2, index % 2)
        root.addLayout(plot_grid)

        log_card = CardFrame("\u5206\u6790\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9\u5206\u6790 CSV", get_dialog_start_dir(DATA_DIR), "CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def start_analysis(self) -> None:
        try:
            from src.workers.analysis_worker import AnalysisWorker

            self.worker = AnalysisWorker(self.csv_edit.text().strip(), int(self.sample_spin.value()), int(self.seed_spin.value()))
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self.log_panel.append_line)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.progress.setValue(0)
            self.worker.start()
        except Exception as exc:
            show_error(self, "\u5206\u6790\u542f\u52a8\u5931\u8d25", str(exc))

    def on_finished(self, result: dict) -> None:
        sample_profile = result["sample_profile"]
        self.schema_tile.value_label.setText(sample_profile["schema_name"])
        self.total_tile.value_label.setText(str(result["profile"]["rows"]))
        self.sample_tile.value_label.setText(str(sample_profile["rows"]))
        self.fraud_tile.value_label.setText(str(result["class_distribution"].get("\u6b3a\u8bc8", 0)))
        draw_class_distribution(self.class_canvas.axes, result["class_distribution"])
        type_distribution = result["type_distribution"]
        if type_distribution:
            draw_bar_chart(self.type_canvas.axes, list(type_distribution.keys()), list(type_distribution.values()), "\u4ea4\u6613\u7c7b\u578b\u5206\u5e03", "\u6837\u672c\u6570", color="#5aa8ff")
            draw_bar_chart(self.rate_canvas.axes, list(result["fraud_rate_by_type"].keys()), list(result["fraud_rate_by_type"].values()), "\u5404\u7c7b\u578b\u6b3a\u8bc8\u7387", "\u6b3a\u8bc8\u7387", color="#f5c65b")
        else:
            draw_empty(self.type_canvas.axes, "\u4ea4\u6613\u7c7b\u578b\u5206\u5e03", "\u5f53\u524d\u6570\u636e\u65e0 type \u5b57\u6bb5")
            draw_empty(self.rate_canvas.axes, "\u5404\u7c7b\u578b\u6b3a\u8bc8\u7387", "\u5f53\u524d\u6570\u636e\u65e0 type \u5b57\u6bb5")
        draw_histogram_by_class(self.amount_canvas.axes, result["amount_hist"]["normal"], result["amount_hist"]["fraud"], "\u91d1\u989d\u5206\u5e03 (Log1p)", "log1p(amount)")
        draw_embedding(self.pca_canvas.axes, result["pca"]["x"], result["pca"]["y"], result["pca"]["labels"], "PCA \u964d\u7ef4", "PC1", "PC2")
        draw_embedding(self.tsne_canvas.axes, result["tsne"]["x"], result["tsne"]["y"], result["tsne"]["labels"], "t-SNE \u964d\u7ef4", "t-SNE-1", "t-SNE-2")
        corr = result["correlation"]
        if corr["labels"]:
            import numpy as np
            draw_heatmap(self.corr_canvas.axes, np.array(corr["matrix"]), corr["labels"], "\u76f8\u5173\u6027\u70ed\u529b\u56fe")
        else:
            draw_empty(self.corr_canvas.axes, "\u76f8\u5173\u6027\u70ed\u529b\u56fe", "\u5f53\u524d\u6837\u672c\u65e0\u6cd5\u8ba1\u7b97\u76f8\u5173\u77e9\u9635")
        variance = result["pca"]["variance_ratio"]
        draw_bar_chart(self.var_canvas.axes, ["PC1", "PC2"], variance, "PCA \u89e3\u91ca\u65b9\u5dee", "\u5360\u6bd4", color="#8ba8ff")
        for canvas in [self.class_canvas, self.type_canvas, self.rate_canvas, self.amount_canvas, self.pca_canvas, self.tsne_canvas, self.corr_canvas, self.var_canvas]:
            canvas.draw()
        show_info(self, "\u5206\u6790\u5b8c\u6210", "\u6570\u636e\u5206\u6790\u56fe\u8868\u5df2\u66f4\u65b0\u3002")

    def on_failed(self, message: str) -> None:
        self.log_panel.append_line(f"\u5206\u6790\u5931\u8d25: {message}")
        show_error(self, "\u5206\u6790\u5931\u8d25", message)
