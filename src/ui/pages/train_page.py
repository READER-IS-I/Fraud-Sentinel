from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, MetricCard, create_path_row, labeled_value, show_error, show_info
from src.utils.file_utils import DEMO_DIR, MODELS_DIR, create_timestamped_dir, get_dialog_start_dir
from src.utils.plot_utils import PlotCanvas, draw_empty, draw_loss_curve


def _default_train_csv() -> str:
    return str(DEMO_DIR / "sample_demo.csv")


class TrainPage(QWidget):
    artifacts_ready = Signal(dict)
    page_title = "\u6a21\u578b\u8bad\u7ec3"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: TrainWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        control_card = CardFrame(
            "\u8bad\u7ec3\u63a7\u5236\u53f0",
            "SMOTE \u4ec5\u5728\u8bad\u7ec3\u96c6\u4e0a\u6267\u884c\uff0c\u9ed8\u8ba4\u76ee\u6807\u6bd4\u4f8b\u4e3a 1:5\uff0c\u907f\u514d\u5168\u91cf 1:1 \u9020\u6210\u8fc7\u9ad8\u5185\u5b58\u538b\u529b\u3002",
        )
        self.csv_edit = QLineEdit(_default_train_csv())
        self.output_edit = QLineEdit(str(MODELS_DIR / "training_runs"))
        browse_csv = HighlightButton("\u6d4f\u89c8", variant="secondary")
        browse_csv.clicked.connect(self.browse_csv)
        browse_out = HighlightButton("\u9009\u62e9", variant="ghost")
        browse_out.clicked.connect(self.browse_output_dir)
        control_card.layout.addWidget(create_path_row("\u8bad\u7ec3\u6570\u636e", self.csv_edit, browse_csv))
        control_card.layout.addWidget(create_path_row("\u8f93\u51fa\u76ee\u5f55", self.output_edit, browse_out))

        form = QFormLayout()
        form.setSpacing(12)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 500)
        self.epoch_spin.setValue(20)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 4096)
        self.batch_spin.setValue(128)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(42)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setDecimals(2)
        self.val_spin.setRange(0.1, 0.4)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setDecimals(2)
        self.test_spin.setRange(0.1, 0.4)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.2)
        self.smote_spin = QDoubleSpinBox()
        self.smote_spin.setDecimals(2)
        self.smote_spin.setRange(0.01, 1.0)
        self.smote_spin.setSingleStep(0.05)
        self.smote_spin.setValue(0.20)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        form.addRow("\u8bad\u7ec3\u8f6e\u6b21", self.epoch_spin)
        form.addRow("\u6279\u5927\u5c0f", self.batch_spin)
        form.addRow("\u5b66\u4e60\u7387", self.lr_spin)
        form.addRow("\u968f\u673a\u79cd\u5b50", self.seed_spin)
        form.addRow("\u9a8c\u8bc1\u96c6\u6bd4\u4f8b", self.val_spin)
        form.addRow("\u6d4b\u8bd5\u96c6\u6bd4\u4f8b", self.test_spin)
        form.addRow("SMOTE \u6bd4\u4f8b", self.smote_spin)
        form.addRow("\u8bad\u7ec3\u8bbe\u5907", self.device_combo)
        control_card.layout.addLayout(form)

        button_row = QHBoxLayout()
        self.start_button = HighlightButton("\u5f00\u59cb\u8bad\u7ec3", variant="primary")
        self.start_button.clicked.connect(self.start_training)
        sample_button = HighlightButton("\u52a0\u8f7d\u793a\u4f8b\u6570\u636e", variant="secondary")
        sample_button.clicked.connect(lambda: self.csv_edit.setText(str(DEMO_DIR / "sample_demo.csv")))
        button_row.addWidget(self.start_button)
        button_row.addWidget(sample_button)
        button_row.addStretch(1)
        control_card.layout.addLayout(button_row)
        root.addWidget(control_card)

        status_card = CardFrame("\u8bad\u7ec3\u72b6\u6001", "\u9a8c\u8bc1\u96c6\u4e0e\u72ec\u7acb\u6d4b\u8bd5\u96c6\u6307\u6807\u5206\u5f00\u5c55\u793a\u3002")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        status_card.layout.addWidget(self.progress)
        metric_grid = QGridLayout()
        metric_grid.setSpacing(12)
        self.metric_accuracy = MetricCard("\u9a8c\u8bc1\u96c6\u51c6\u786e\u7387", "--", "\u9a8c\u8bc1\u96c6\u603b\u4f53\u51c6\u786e\u7387")
        self.metric_recall = MetricCard("\u9a8c\u8bc1\u96c6 Recall", "--", "\u9a8c\u8bc1\u96c6\u6b3a\u8bc8\u53ec\u56de\u7387")
        self.metric_f1 = MetricCard("\u9a8c\u8bc1\u96c6 F1-score", "--", "\u9a8c\u8bc1\u96c6\u5e73\u8861\u8868\u73b0")
        self.metric_auc = MetricCard("\u9a8c\u8bc1\u96c6 ROC-AUC", "--", "\u9a8c\u8bc1\u96c6\u533a\u5206\u80fd\u529b")
        for idx, card in enumerate([self.metric_accuracy, self.metric_recall, self.metric_f1, self.metric_auc]):
            metric_grid.addWidget(card, 0, idx)
        self.test_accuracy = MetricCard("\u6d4b\u8bd5\u96c6\u51c6\u786e\u7387", "--", "\u72ec\u7acb\u6d4b\u8bd5\u96c6\u603b\u4f53\u51c6\u786e\u7387")
        self.test_recall = MetricCard("\u6d4b\u8bd5\u96c6 Recall", "--", "\u72ec\u7acb\u6d4b\u8bd5\u96c6\u6b3a\u8bc8\u53ec\u56de\u7387")
        self.test_f1 = MetricCard("\u6d4b\u8bd5\u96c6 F1-score", "--", "\u72ec\u7acb\u6d4b\u8bd5\u96c6\u5e73\u8861\u8868\u73b0")
        self.test_auc = MetricCard("\u6d4b\u8bd5\u96c6 ROC-AUC", "--", "\u72ec\u7acb\u6d4b\u8bd5\u96c6\u533a\u5206\u80fd\u529b")
        for idx, card in enumerate([self.test_accuracy, self.test_recall, self.test_f1, self.test_auc]):
            metric_grid.addWidget(card, 1, idx)
        status_card.layout.addLayout(metric_grid)
        root.addWidget(status_card)

        generalization_card = CardFrame("\u6cdb\u5316\u914d\u7f6e", "\u8fd9\u4e00\u533a\u57df\u7528\u4e8e\u5c55\u793a\u6a21\u578b\u5982\u4f55\u907f\u514d\u8fc7\u4e8e\u4e50\u89c2\u7684\u9a8c\u8bc1\u7ed3\u679c\u3002")
        generalization_grid = QGridLayout()
        generalization_grid.setSpacing(12)
        self.tile_schema = labeled_value("\u6570\u636e\u7ed3\u6784", "--")
        self.tile_split = labeled_value("\u6570\u636e\u5212\u5206", "--")
        self.tile_smote = labeled_value("SMOTE \u8303\u56f4", "--")
        self.tile_regularization = labeled_value("\u6b63\u5219\u5316", "--")
        self.tile_guard = labeled_value("\u9632\u6cc4\u6f0f\u7b56\u7565", "--")
        tiles = [self.tile_schema, self.tile_split, self.tile_smote, self.tile_regularization, self.tile_guard]
        for idx, tile in enumerate(tiles):
            generalization_grid.addWidget(tile, 0, idx)
        generalization_card.layout.addLayout(generalization_grid)
        root.addWidget(generalization_card)

        plot_card = CardFrame(
            "\u8bad\u7ec3\u66f2\u7ebf",
            "\u8bad\u7ec3\u5b8c\u6210\u540e\u663e\u793a loss \u53d8\u5316\u66f2\u7ebf\uff0c\u56fe\u8868\u533a\u57df\u5df2\u538b\u7f29\u5e76\u652f\u6301\u6eda\u52a8\u67e5\u770b\u3002",
        )
        self.loss_canvas = PlotCanvas(width=5.3, height=2.15)
        self.loss_canvas.setMinimumHeight(220)
        draw_empty(self.loss_canvas.axes, "\u8bad\u7ec3\u635f\u5931\u66f2\u7ebf", "\u8fd0\u884c\u8bad\u7ec3\u540e\u663e\u793a\u66f2\u7ebf")
        self.loss_canvas.draw()
        self.loss_scroll = QScrollArea()
        self.loss_scroll.setWidgetResizable(True)
        self.loss_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.loss_scroll.setWidget(self.loss_canvas)
        self.loss_scroll.setMinimumHeight(240)
        self.loss_scroll.setMaximumHeight(280)
        plot_card.layout.addWidget(self.loss_scroll)
        root.addWidget(plot_card)

        log_card = CardFrame("\u8bad\u7ec3\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9\u8bad\u7ec3 CSV", get_dialog_start_dir(DEMO_DIR), "CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55", self.output_edit.text() or str(MODELS_DIR))
        if path:
            self.output_edit.setText(path)

    def start_training(self) -> None:
        try:
            from src.core.trainer import TrainingConfig
            from src.workers.train_worker import TrainWorker

            base_output = Path(self.output_edit.text().strip()) if self.output_edit.text().strip() else MODELS_DIR / "training_runs"
            run_dir = create_timestamped_dir(base_output, prefix="mlp_smote")
            config = TrainingConfig(
                csv_path=self.csv_edit.text().strip(),
                output_dir=str(run_dir),
                epochs=int(self.epoch_spin.value()),
                batch_size=int(self.batch_spin.value()),
                learning_rate=float(self.lr_spin.value()),
                random_seed=int(self.seed_spin.value()),
                val_size=float(self.val_spin.value()),
                test_size=float(self.test_spin.value()),
                smote_ratio=float(self.smote_spin.value()),
                device=self.device_combo.currentText(),
            )
            self.worker = TrainWorker(config)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self.log_panel.append_line)
            self.worker.finished_ok.connect(self.on_training_finished)
            self.worker.failed.connect(self.on_training_failed)
            self.start_button.setEnabled(False)
            self.progress.setValue(0)
            self.log_panel.append_line(f"\u8bad\u7ec3\u5df2\u542f\u52a8\uff0c\u8f93\u51fa\u76ee\u5f55: {run_dir}")
            self.worker.start()
        except Exception as exc:
            show_error(self, "\u8bad\u7ec3\u542f\u52a8\u5931\u8d25", str(exc))

    def on_training_finished(self, result: dict) -> None:
        self.start_button.setEnabled(True)
        metrics = result["metrics"]
        val_metrics = metrics["validation"]
        test_metrics = metrics["test"]
        generalization = metrics["generalization"]
        self.metric_accuracy.set_value(f"{val_metrics['accuracy']:.4f}")
        self.metric_recall.set_value(f"{val_metrics['recall']:.4f}")
        self.metric_f1.set_value(f"{val_metrics['f1_score']:.4f}")
        val_auc = val_metrics["roc_auc"]
        self.metric_auc.set_value("--" if val_auc is None else f"{val_auc:.4f}")
        self.test_accuracy.set_value(f"{test_metrics['accuracy']:.4f}")
        self.test_recall.set_value(f"{test_metrics['recall']:.4f}")
        self.test_f1.set_value(f"{test_metrics['f1_score']:.4f}")
        test_auc = test_metrics["roc_auc"]
        self.test_auc.set_value("--" if test_auc is None else f"{test_auc:.4f}")
        self.tile_schema.value_label.setText(generalization.get("schema", "--"))
        self.tile_split.value_label.setText(generalization["split_strategy"])
        self.tile_smote.value_label.setText(generalization["smote_scope"])
        self.tile_regularization.value_label.setText(generalization["regularization"])
        self.tile_guard.value_label.setText(generalization["leakage_guard"])
        draw_loss_curve(self.loss_canvas.axes, result["history"]["train_loss"], result["history"]["val_loss"])
        self.loss_canvas.draw()
        self.log_panel.append_line(f"\u6a21\u578b\u6587\u4ef6: {result['model_path']}")
        self.log_panel.append_line(f"\u9884\u5904\u7406\u5668: {result['preprocessor_path']}")
        self.log_panel.append_line(f"\u6d4b\u8bd5\u96c6\u6307\u6807 -> accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['f1_score']:.4f}, recall={test_metrics['recall']:.4f}")
        self.artifacts_ready.emit(result)
        show_info(self, "\u8bad\u7ec3\u5b8c\u6210", "\u9a8c\u8bc1\u96c6\u4e0e\u72ec\u7acb\u6d4b\u8bd5\u96c6\u6307\u6807\u5df2\u751f\u6210\u3002")

    def on_training_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.log_panel.append_line(f"\u8bad\u7ec3\u5931\u8d25: {message}")
        show_error(self, "\u8bad\u7ec3\u5931\u8d25", message)
