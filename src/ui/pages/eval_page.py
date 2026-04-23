from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QGridLayout, QLineEdit, QProgressBar, QVBoxLayout, QWidget

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, MetricCard, create_path_row, show_error, show_info
from src.utils.file_utils import DEMO_DIR, get_dialog_start_dir
from src.utils.plot_utils import PlotCanvas, draw_class_distribution, draw_confusion_matrix, draw_empty, draw_pr_curve, draw_roc_curve


def _default_eval_csv() -> str:
    return str(DEMO_DIR / "sample_demo.csv")


class EvalPage(QWidget):
    page_title = "\u6a21\u578b\u8bc4\u4f30"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: EvalWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        control_card = CardFrame("\u8bc4\u4f30\u63a7\u5236\u53f0", "\u52a0\u8f7d\u6a21\u578b\u4e0e\u9884\u5904\u7406\u5668\uff0c\u5728\u9a8c\u8bc1\u96c6\u6216\u72ec\u7acb\u6d4b\u8bd5\u96c6\u4e0a\u8fdb\u884c\u8bc4\u4f30\u3002")
        self.model_edit = QLineEdit()
        self.preprocessor_edit = QLineEdit()
        self.csv_edit = QLineEdit(_default_eval_csv())
        model_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        pre_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        csv_button = HighlightButton("\u6d4f\u89c8", variant="ghost")
        model_button.clicked.connect(lambda: self._browse_file(self.model_edit, "\u9009\u62e9\u6a21\u578b\u6587\u4ef6", "PyTorch Files (*.pt)"))
        pre_button.clicked.connect(lambda: self._browse_file(self.preprocessor_edit, "\u9009\u62e9\u9884\u5904\u7406\u5668", "Joblib Files (*.joblib)"))
        csv_button.clicked.connect(lambda: self._browse_file(self.csv_edit, "\u9009\u62e9\u8bc4\u4f30 CSV", "CSV Files (*.csv)"))
        control_card.layout.addWidget(create_path_row("\u6a21\u578b", self.model_edit, model_button))
        control_card.layout.addWidget(create_path_row("\u9884\u5904\u7406\u5668", self.preprocessor_edit, pre_button))
        control_card.layout.addWidget(create_path_row("\u6570\u636e\u6587\u4ef6", self.csv_edit, csv_button))
        self.evaluate_button = HighlightButton("\u5f00\u59cb\u8bc4\u4f30", variant="primary")
        self.evaluate_button.clicked.connect(self.start_evaluation)
        control_card.layout.addWidget(self.evaluate_button)
        root.addWidget(control_card)

        metrics_card = CardFrame("\u8bc4\u4f30\u6307\u6807")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        metrics_card.layout.addWidget(self.progress)
        metric_grid = QGridLayout()
        metric_grid.setSpacing(12)
        self.cards = {
            "accuracy": MetricCard("\u51c6\u786e\u7387", "--", "\u6574\u4f53\u51c6\u786e\u7387"),
            "precision": MetricCard("\u7cbe\u786e\u7387", "--", "\u9884\u6d4b\u4e3a\u6b63\u7684\u51c6\u786e\u6027"),
            "recall": MetricCard("Recall", "--", "\u6b3a\u8bc8\u6837\u672c\u53ec\u56de\u7387"),
            "f1": MetricCard("F1-score", "--", "\u5e73\u8861\u8868\u73b0\u6307\u6807"),
            "roc_auc": MetricCard("ROC-AUC", "--", "\u533a\u5206\u80fd\u529b"),
            "pr_auc": MetricCard("PR-AUC", "--", "\u4e0d\u5e73\u8861\u4efb\u52a1\u8868\u73b0"),
        }
        for idx, key in enumerate(self.cards):
            metric_grid.addWidget(self.cards[key], idx // 3, idx % 3)
        metrics_card.layout.addLayout(metric_grid)
        root.addWidget(metrics_card)

        plot_grid = QGridLayout()
        plot_grid.setSpacing(14)
        self.cm_canvas = PlotCanvas(width=4.6, height=2.8)
        self.roc_canvas = PlotCanvas(width=4.6, height=2.8)
        self.pr_canvas = PlotCanvas(width=4.6, height=2.8)
        self.dist_canvas = PlotCanvas(width=4.6, height=2.8)
        for canvas, title in [(self.cm_canvas, "\u6df7\u6dc6\u77e9\u9635"), (self.roc_canvas, "ROC Curve"), (self.pr_canvas, "PR Curve"), (self.dist_canvas, "\u7c7b\u522b\u5206\u5e03")]:
            draw_empty(canvas.axes, title, "\u7b49\u5f85\u8bc4\u4f30\u7ed3\u679c")
            canvas.draw()
        for index, (title, canvas) in enumerate([("\u6df7\u6dc6\u77e9\u9635", self.cm_canvas), ("ROC Curve", self.roc_canvas), ("PR Curve", self.pr_canvas), ("\u7c7b\u522b\u5206\u5e03", self.dist_canvas)]):
            card = CardFrame(title)
            card.layout.addWidget(canvas)
            plot_grid.addWidget(card, index // 2, index % 2)
        root.addLayout(plot_grid)

        log_card = CardFrame("\u8bc4\u4f30\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def _browse_file(self, target: QLineEdit, title: str, pattern: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, get_dialog_start_dir(DEMO_DIR), pattern)
        if path:
            target.setText(path)

    def set_artifact_paths(self, model_path: str, preprocessor_path: str) -> None:
        self.model_edit.setText(model_path)
        self.preprocessor_edit.setText(preprocessor_path)

    def start_evaluation(self) -> None:
        try:
            from src.workers.eval_worker import EvalWorker

            self.worker = EvalWorker(self.model_edit.text().strip(), self.preprocessor_edit.text().strip(), self.csv_edit.text().strip())
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self.log_panel.append_line)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.evaluate_button.setEnabled(False)
            self.progress.setValue(0)
            self.worker.start()
        except Exception as exc:
            show_error(self, "\u8bc4\u4f30\u542f\u52a8\u5931\u8d25", str(exc))

    def on_finished(self, result: dict) -> None:
        self.evaluate_button.setEnabled(True)
        self.cards["accuracy"].set_value(f"{result['accuracy']:.4f}")
        self.cards["precision"].set_value(f"{result['precision']:.4f}")
        self.cards["recall"].set_value(f"{result['recall']:.4f}")
        self.cards["f1"].set_value(f"{result['f1_score']:.4f}")
        self.cards["roc_auc"].set_value("--" if result["roc_auc"] is None else f"{result['roc_auc']:.4f}")
        self.cards["pr_auc"].set_value("--" if result["pr_auc"] is None else f"{result['pr_auc']:.4f}")
        draw_confusion_matrix(self.cm_canvas.axes, result["confusion_matrix"])
        draw_roc_curve(self.roc_canvas.axes, result["roc_curve"]["fpr"], result["roc_curve"]["tpr"], result["roc_auc"])
        draw_pr_curve(self.pr_canvas.axes, result["pr_curve"]["recall"], result["pr_curve"]["precision"], result["pr_auc"])
        draw_class_distribution(self.dist_canvas.axes, result["actual_distribution"], result["predicted_distribution"])
        for canvas in [self.cm_canvas, self.roc_canvas, self.pr_canvas, self.dist_canvas]:
            canvas.draw()
        self.log_panel.append_line(f"\u8bc4\u4f30\u6837\u672c\u6570: {result['sample_count']}")
        self.log_panel.append_line(f"\u6570\u636e\u7ed3\u6784: {result['schema_name']}")
        show_info(self, "\u8bc4\u4f30\u5b8c\u6210", "\u8bc4\u4f30\u6307\u6807\u4e0e\u56fe\u8868\u5df2\u66f4\u65b0\u3002")

    def on_failed(self, message: str) -> None:
        self.evaluate_button.setEnabled(True)
        self.log_panel.append_line(f"\u8bc4\u4f30\u5931\u8d25: {message}")
        show_error(self, "\u8bc4\u4f30\u5931\u8d25", message)
