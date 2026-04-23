from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, MetricCard, create_path_row, show_error, show_info
from src.utils.file_utils import DATA_DIR, DEMO_DIR, get_dialog_start_dir
from src.utils.plot_utils import PlotCanvas, draw_curve_collection, draw_empty, draw_model_metric_bars


def _default_csv() -> str:
    final_csv = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
    return str(final_csv) if final_csv.exists() else str(DEMO_DIR / "sample_demo.csv")


class ComparePage(QWidget):
    page_title = "\u6a21\u578b\u5bf9\u6bd4"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: CompareWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        control_card = CardFrame("\u5bf9\u6bd4\u63a7\u5236\u53f0", "\u6309\u8bf4\u660e\u4e66\u8865\u5145\u903b\u8f91\u56de\u5f52\u3001KNN\u3001SVM\u3001\u51b3\u7b56\u6811\u4e0e MLP \u7684\u6027\u80fd\u5bf9\u6bd4\u3002")
        self.csv_edit = QLineEdit(_default_csv())
        browse_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        browse_button.clicked.connect(self.browse_csv)
        control_card.layout.addWidget(create_path_row("\u6570\u636e\u6587\u4ef6", self.csv_edit, browse_button))
        row = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(2000, 2147483647)
        self.sample_spin.setSingleStep(1000)
        self.sample_spin.setValue(12000)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(42)
        self.smote_spin = QDoubleSpinBox()
        self.smote_spin.setDecimals(2)
        self.smote_spin.setRange(0.01, 1.0)
        self.smote_spin.setSingleStep(0.05)
        self.smote_spin.setValue(0.20)
        run_button = HighlightButton("\u5f00\u59cb\u5bf9\u6bd4", variant="primary")
        run_button.clicked.connect(self.start_compare)
        for label, widget in [("\u62bd\u6837\u89c4\u6a21", self.sample_spin), ("\u968f\u673a\u79cd\u5b50", self.seed_spin), ("SMOTE \u6bd4\u4f8b", self.smote_spin)]:
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            row.addSpacing(12)
        row.addStretch(1)
        row.addWidget(run_button)
        control_card.layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        control_card.layout.addWidget(self.progress)
        root.addWidget(control_card)

        summary_card = CardFrame("\u5bf9\u6bd4\u6982\u89c8")
        summary_grid = QGridLayout()
        summary_grid.setSpacing(12)
        self.best_card = MetricCard("\u6700\u4f73\u6a21\u578b", "--", "\u6309 F1-score \u6392\u5e8f")
        self.rows_card = MetricCard("\u5bf9\u6bd4\u6837\u672c\u6570", "--", "\u62bd\u6837\u540e\u7684\u6d4b\u8bd5\u8303\u56f4")
        self.model_card = MetricCard("\u6a21\u578b\u6570\u91cf", "--", "\u5f53\u524d\u53c2\u4e0e\u5bf9\u6bd4\u7684\u6a21\u578b")
        for index, card in enumerate([self.best_card, self.rows_card, self.model_card]):
            summary_grid.addWidget(card, 0, index)
        summary_card.layout.addLayout(summary_grid)
        root.addWidget(summary_card)

        table_card = CardFrame("\u6307\u6807\u8868")
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        table_card.layout.addWidget(self.table)
        root.addWidget(table_card)

        self.f1_canvas = PlotCanvas(width=4.8, height=2.8)
        self.auc_canvas = PlotCanvas(width=4.8, height=2.8)
        self.roc_canvas = PlotCanvas(width=4.8, height=2.8)
        self.pr_canvas = PlotCanvas(width=4.8, height=2.8)
        for canvas, title in [(self.f1_canvas, "F1 \u5bf9\u6bd4"), (self.auc_canvas, "ROC-AUC \u5bf9\u6bd4"), (self.roc_canvas, "ROC \u5bf9\u6bd4"), (self.pr_canvas, "PR \u5bf9\u6bd4")]:
            draw_empty(canvas.axes, title, "\u7b49\u5f85\u6a21\u578b\u5bf9\u6bd4\u7ed3\u679c")
            canvas.draw()
        plot_grid = QGridLayout()
        plot_grid.setSpacing(14)
        for index, (title, canvas) in enumerate([("F1-score \u5bf9\u6bd4", self.f1_canvas), ("ROC-AUC \u5bf9\u6bd4", self.auc_canvas), ("ROC Curve \u5bf9\u6bd4", self.roc_canvas), ("PR Curve \u5bf9\u6bd4", self.pr_canvas)]):
            card = CardFrame(title)
            card.layout.addWidget(canvas)
            plot_grid.addWidget(card, index // 2, index % 2)
        root.addLayout(plot_grid)

        log_card = CardFrame("\u5bf9\u6bd4\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9\u5bf9\u6bd4 CSV", get_dialog_start_dir(DATA_DIR), "CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def start_compare(self) -> None:
        try:
            from src.workers.compare_worker import CompareWorker

            self.worker = CompareWorker(self.csv_edit.text().strip(), int(self.sample_spin.value()), int(self.seed_spin.value()), float(self.smote_spin.value()))
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self.log_panel.append_line)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.progress.setValue(0)
            self.worker.start()
        except Exception as exc:
            show_error(self, "\u5bf9\u6bd4\u542f\u52a8\u5931\u8d25", str(exc))

    def on_finished(self, result: dict) -> None:
        rows = result["rows"]
        self.best_card.set_value(result["best_model"])
        self.rows_card.set_value(str(result["profile"]["rows"]))
        self.model_card.set_value(str(len(rows)))
        self.populate_table(rows)
        names = [row["model"] for row in rows]
        f1_values = [row["f1_score"] for row in rows]
        roc_auc_values = [0.0 if row["roc_auc"] is None else row["roc_auc"] for row in rows]
        draw_model_metric_bars(self.f1_canvas.axes, names, f1_values, "F1-score \u5bf9\u6bd4", "F1-score")
        draw_model_metric_bars(self.auc_canvas.axes, names, roc_auc_values, "ROC-AUC \u5bf9\u6bd4", "ROC-AUC")
        draw_curve_collection(self.roc_canvas.axes, result["roc_curves"], "roc")
        draw_curve_collection(self.pr_canvas.axes, result["pr_curves"], "pr")
        for canvas in [self.f1_canvas, self.auc_canvas, self.roc_canvas, self.pr_canvas]:
            canvas.draw()
        show_info(self, "\u5bf9\u6bd4\u5b8c\u6210", "\u6a21\u578b\u5bf9\u6bd4\u56fe\u8868\u5df2\u66f4\u65b0\u3002")

    def on_failed(self, message: str) -> None:
        self.log_panel.append_line(f"\u5bf9\u6bd4\u5931\u8d25: {message}")
        show_error(self, "\u5bf9\u6bd4\u5931\u8d25", message)

    def populate_table(self, rows: list[dict]) -> None:
        columns = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "seconds"]
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, column in enumerate(columns):
                value = row.get(column, "--")
                display = f"{value:.4f}" if isinstance(value, float) else str(value)
                self.table.setItem(row_index, col_index, QTableWidgetItem(display))
        self.table.resizeColumnsToContents()
