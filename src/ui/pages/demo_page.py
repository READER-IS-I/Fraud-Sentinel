from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, MetricCard, show_error, show_info
from src.workers.demo_worker import DemoWorker


class DemoPage(QWidget):
    artifacts_ready = Signal(dict)
    page_title = "\u793a\u4f8b\u6f14\u793a"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: DemoWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        intro_card = CardFrame("\u4e00\u952e\u6f14\u793a", "\u81ea\u52a8\u8bad\u7ec3\u5185\u7f6e PaySim \u6a21\u578b\u5e76\u5bf9\u6279\u91cf\u6837\u672c\u6267\u884c\u63a8\u7406\u3002")
        intro_label = QLabel("\u6f14\u793a\u94fe\u8def\uff1asample_demo.csv -> SMOTE + MLP \u8bad\u7ec3 -> \u6a21\u578b\u4ea7\u7269 -> sample_inference.csv \u6279\u91cf\u63a8\u7406")
        intro_label.setObjectName("bodyText")
        intro_label.setWordWrap(True)
        self.start_button = HighlightButton("\u5f00\u59cb\u6f14\u793a", variant="primary")
        self.start_button.clicked.connect(self.start_demo)
        intro_card.layout.addWidget(intro_label)
        intro_card.layout.addWidget(self.start_button)
        root.addWidget(intro_card)

        status_card = CardFrame("\u6f14\u793a\u7ed3\u679c")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        status_card.layout.addWidget(self.progress)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.high_risk_card = MetricCard("\u9ad8\u98ce\u9669\u6837\u672c", "--", "\u6279\u91cf\u7ed3\u679c\u4e2d\u7684\u9ad8\u98ce\u9669\u6837\u672c\u6570")
        self.max_prob_card = MetricCard("\u6700\u9ad8\u6b3a\u8bc8\u6982\u7387", "--", "\u793a\u4f8b\u7ed3\u679c\u4e2d\u7684\u6700\u9ad8\u6982\u7387")
        self.result_path_card = MetricCard("\u7ed3\u679c\u6587\u4ef6", "--", "\u6f14\u793a\u7ed3\u679c CSV \u8def\u5f84")
        for index, card in enumerate([self.high_risk_card, self.max_prob_card, self.result_path_card]):
            grid.addWidget(card, 0, index)
        status_card.layout.addLayout(grid)
        root.addWidget(status_card)

        table_card = CardFrame("\u98ce\u9669\u6837\u672c\u9884\u89c8", "\u6309\u6b3a\u8bc8\u6982\u7387\u4ece\u9ad8\u5230\u4f4e\u5c55\u793a\u524d 12 \u6761\u8bb0\u5f55\u3002")
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        table_card.layout.addWidget(self.table)
        root.addWidget(table_card)

        log_card = CardFrame("\u6f14\u793a\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def start_demo(self) -> None:
        try:
            self.worker = DemoWorker()
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self.log_panel.append_line)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.start_button.setEnabled(False)
            self.progress.setValue(0)
            self.worker.start()
        except Exception as exc:
            show_error(self, "\u6f14\u793a\u542f\u52a8\u5931\u8d25", str(exc))

    def on_finished(self, result: dict) -> None:
        self.start_button.setEnabled(True)
        self.high_risk_card.set_value(str(result["high_risk_count"]))
        self.max_prob_card.set_value(f"{result['max_probability']:.4f}")
        self.result_path_card.set_value("\u5df2\u751f\u6210", result["result_csv"])
        self.populate_table(result["preview"])
        self.artifacts_ready.emit(result)
        show_info(self, "\u6f14\u793a\u5b8c\u6210", "\u793a\u4f8b\u8bad\u7ec3\u4e0e\u6279\u91cf\u63a8\u7406\u5df2\u5b8c\u6210\u3002")

    def on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.log_panel.append_line(f"\u6f14\u793a\u5931\u8d25: {message}")
        show_error(self, "\u6f14\u793a\u5931\u8d25", message)

    def populate_table(self, records: list[dict]) -> None:
        if not records:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        columns = list(records[0].keys())
        self.table.clear()
        self.table.setRowCount(len(records))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        for row, record in enumerate(records):
            for col, key in enumerate(columns):
                value = record[key]
                display = f"{value:.6f}" if isinstance(value, float) else str(value)
                self.table.setItem(row, col, QTableWidgetItem(display))
        self.table.resizeColumnsToContents()
