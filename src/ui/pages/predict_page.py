from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets import CardFrame, HighlightButton, LogPanel, create_path_row, show_error, show_info
from src.utils.file_utils import DATA_DIR, EXAMPLES_DIR, OUTPUTS_DIR, get_dialog_start_dir


def _default_batch_csv() -> str:
    return str(EXAMPLES_DIR / "sample_inference.csv")


class PredictPage(QWidget):
    page_title = "\u98ce\u9669\u63a8\u7406"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.predictor: FraudPredictor | None = None
        self.batch_result: pd.DataFrame | None = None
        self.input_fields: dict[str, QLineEdit] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        artifact_card = CardFrame("\u63a8\u7406\u8d44\u6e90", "\u52a0\u8f7d\u6a21\u578b\u548c\u9884\u5904\u7406\u5668\u540e\uff0c\u5373\u53ef\u5bf9\u5355\u6761\u6216\u6279\u91cf\u4ea4\u6613\u8fdb\u884c\u98ce\u9669\u5224\u65ad\u3002")
        self.model_edit = QLineEdit()
        self.preprocessor_edit = QLineEdit()
        artifact_card.layout.addWidget(create_path_row("\u6a21\u578b", self.model_edit, self._browse_button(self.model_edit, "\u9009\u62e9\u6a21\u578b\u6587\u4ef6", "PyTorch Files (*.pt)")))
        artifact_card.layout.addWidget(create_path_row("\u9884\u5904\u7406\u5668", self.preprocessor_edit, self._browse_button(self.preprocessor_edit, "\u9009\u62e9\u9884\u5904\u7406\u5668", "Joblib Files (*.joblib)")))
        load_button = HighlightButton("\u52a0\u8f7d\u63a8\u7406\u8d44\u6e90", variant="primary")
        load_button.clicked.connect(self.load_predictor)
        artifact_card.layout.addWidget(load_button)
        root.addWidget(artifact_card)

        tabs = QTabWidget()
        tabs.addTab(self._build_single_tab(), "\u5355\u6761\u63a8\u7406")
        tabs.addTab(self._build_batch_tab(), "\u6279\u91cf\u63a8\u7406")
        root.addWidget(tabs)

        log_card = CardFrame("\u63a8\u7406\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def _browse_button(self, target: QLineEdit, title: str, pattern: str) -> HighlightButton:
        button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        button.clicked.connect(lambda: self._browse_file(target, title, pattern))
        return button

    def _browse_file(self, target: QLineEdit, title: str, pattern: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, get_dialog_start_dir(EXAMPLES_DIR), pattern)
        if path:
            target.setText(path)

    def set_artifact_paths(self, model_path: str, preprocessor_path: str) -> None:
        self.model_edit.setText(model_path)
        self.preprocessor_edit.setText(preprocessor_path)

    def load_predictor(self) -> None:
        try:
            from src.core.predictor import FraudPredictor

            self.predictor = FraudPredictor(Path(self.model_edit.text().strip()), Path(self.preprocessor_edit.text().strip()))
            self.log_panel.append_line("\u63a8\u7406\u8d44\u6e90\u52a0\u8f7d\u5b8c\u6210\u3002")
            show_info(self, "\u52a0\u8f7d\u6210\u529f", "\u6a21\u578b\u4e0e\u9884\u5904\u7406\u5668\u5df2\u5c31\u7eea\u3002")
        except Exception as exc:
            show_error(self, "\u52a0\u8f7d\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u52a0\u8f7d\u5931\u8d25: {exc}")

    def _build_single_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        card = CardFrame("\u4ea4\u6613\u5f55\u5165\u8868\u5355", "\u6309 PaySim \u539f\u59cb\u4ea4\u6613\u5b57\u6bb5\u5f55\u5165\u5355\u6761\u6837\u672c\u3002")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setSpacing(10)

        self.input_fields["step"] = self._create_numeric_field("1")
        self.input_fields["amount"] = self._create_numeric_field("1200")
        self.input_fields["nameOrig"] = self._create_text_field("C1231000001")
        self.input_fields["oldbalanceOrg"] = self._create_numeric_field("5000")
        self.input_fields["newbalanceOrig"] = self._create_numeric_field("3800")
        self.input_fields["nameDest"] = self._create_text_field("M1978000001")
        self.input_fields["oldbalanceDest"] = self._create_numeric_field("0")
        self.input_fields["newbalanceDest"] = self._create_numeric_field("0")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])

        widgets = [
            ("step", QLabel("step"), self.input_fields["step"]),
            ("type", QLabel("type"), self.type_combo),
            ("amount", QLabel("amount"), self.input_fields["amount"]),
            ("nameOrig", QLabel("nameOrig"), self.input_fields["nameOrig"]),
            ("oldbalanceOrg", QLabel("oldbalanceOrg"), self.input_fields["oldbalanceOrg"]),
            ("newbalanceOrig", QLabel("newbalanceOrig"), self.input_fields["newbalanceOrig"]),
            ("nameDest", QLabel("nameDest"), self.input_fields["nameDest"]),
            ("oldbalanceDest", QLabel("oldbalanceDest"), self.input_fields["oldbalanceDest"]),
            ("newbalanceDest", QLabel("newbalanceDest"), self.input_fields["newbalanceDest"]),
        ]
        for index, (_key, label, field) in enumerate(widgets):
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(4)
            wrapper_layout.addWidget(label)
            wrapper_layout.addWidget(field)
            form_layout.addWidget(wrapper, index // 3, index % 3)

        scroll.setWidget(form_widget)
        card.layout.addWidget(scroll)
        button_row = QHBoxLayout()
        safe_button = HighlightButton("\u586b\u5145\u4f4e\u98ce\u9669\u793a\u4f8b", variant="secondary")
        safe_button.clicked.connect(self.fill_safe_sample)
        risky_button = HighlightButton("\u586b\u5145\u9ad8\u98ce\u9669\u793a\u4f8b", variant="ghost")
        risky_button.clicked.connect(self.fill_risky_sample)
        predict_button = HighlightButton("\u6267\u884c\u63a8\u7406", variant="primary")
        predict_button.clicked.connect(self.run_single_prediction)
        button_row.addWidget(safe_button)
        button_row.addWidget(risky_button)
        button_row.addWidget(predict_button)
        button_row.addStretch(1)
        card.layout.addLayout(button_row)
        layout.addWidget(card)

        self.result_card = CardFrame("\u63a8\u7406\u7ed3\u679c", "\u9ad8\u98ce\u9669\u4ea4\u6613\u4f1a\u4ee5\u9192\u76ee\u9884\u8b66\u989c\u8272\u5c55\u793a\u3002")
        self.result_label = QLabel("\u7ed3\u679c: --")
        self.result_label.setObjectName("resultValue")
        self.prob_label = QLabel("\u6b3a\u8bc8\u6982\u7387: --")
        self.prob_label.setObjectName("bodyText")
        self.risk_label = QLabel("\u98ce\u9669\u7b49\u7ea7: --")
        self.risk_label.setObjectName("riskBadge")
        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setValue(0)
        for widget in [self.result_label, self.prob_label, self.risk_label, self.risk_bar]:
            self.result_card.layout.addWidget(widget)
        layout.addWidget(self.result_card)
        return page

    def _build_batch_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        card = CardFrame("\u6279\u91cf CSV \u63a8\u7406", "\u5bfc\u5165 PaySim \u4ea4\u6613 CSV \u540e\u76f4\u63a5\u751f\u6210\u9884\u6d4b\u7ed3\u679c\u8868\u3002")
        self.batch_csv_edit = QLineEdit(_default_batch_csv())
        browse_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        browse_button.clicked.connect(lambda: self._browse_file(self.batch_csv_edit, "\u9009\u62e9\u6279\u91cf CSV", "CSV Files (*.csv)"))
        card.layout.addWidget(create_path_row("CSV", self.batch_csv_edit, browse_button))
        button_row = QHBoxLayout()
        run_button = HighlightButton("\u6279\u91cf\u63a8\u7406", variant="primary")
        run_button.clicked.connect(self.run_batch_prediction)
        export_button = HighlightButton("\u5bfc\u51fa CSV", variant="ghost")
        export_button.clicked.connect(self.export_batch_result)
        button_row.addWidget(run_button)
        button_row.addWidget(export_button)
        button_row.addStretch(1)
        card.layout.addLayout(button_row)
        self.batch_table = QTableWidget(0, 0)
        self.batch_table.setAlternatingRowColors(True)
        card.layout.addWidget(self.batch_table)
        layout.addWidget(card)
        return page

    def _create_numeric_field(self, default: str) -> QLineEdit:
        field = QLineEdit(default)
        field.setPlaceholderText(default)
        return field

    def _create_text_field(self, default: str) -> QLineEdit:
        field = QLineEdit(default)
        field.setPlaceholderText(default)
        return field

    def fill_safe_sample(self) -> None:
        self.type_combo.setCurrentText("PAYMENT")
        self.input_fields["step"].setText("3")
        self.input_fields["amount"].setText("520")
        self.input_fields["nameOrig"].setText("C1000001001")
        self.input_fields["oldbalanceOrg"].setText("12000")
        self.input_fields["newbalanceOrig"].setText("11480")
        self.input_fields["nameDest"].setText("M4000000021")
        self.input_fields["oldbalanceDest"].setText("0")
        self.input_fields["newbalanceDest"].setText("0")

    def fill_risky_sample(self) -> None:
        self.type_combo.setCurrentText("TRANSFER")
        self.input_fields["step"].setText("533")
        self.input_fields["amount"].setText("980000")
        self.input_fields["nameOrig"].setText("C2000009091")
        self.input_fields["oldbalanceOrg"].setText("980000")
        self.input_fields["newbalanceOrig"].setText("0")
        self.input_fields["nameDest"].setText("C9000091000")
        self.input_fields["oldbalanceDest"].setText("0")
        self.input_fields["newbalanceDest"].setText("0")

    def _collect_single_record(self) -> dict[str, object]:
        record: dict[str, object] = {"type": self.type_combo.currentText()}
        numeric_fields = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        text_fields = ["nameOrig", "nameDest"]
        for column in numeric_fields:
            try:
                record[column] = float(self.input_fields[column].text().strip())
            except ValueError as exc:
                raise ValueError(f"\u5b57\u6bb5 {column} \u5fc5\u987b\u4e3a\u6570\u503c\u3002") from exc
        for column in text_fields:
            value = self.input_fields[column].text().strip()
            if not value:
                raise ValueError(f"\u5b57\u6bb5 {column} \u4e0d\u80fd\u4e3a\u7a7a\u3002")
            record[column] = value
        record["step"] = int(record["step"])
        return record

    def run_single_prediction(self) -> None:
        try:
            if self.predictor is None:
                self.load_predictor()
            if self.predictor is None:
                return
            result = self.predictor.predict_single(self._collect_single_record())
            probability = result["fraud_probability"]
            self.result_label.setText(f"\u7ed3\u679c: {result['label']}")
            self.prob_label.setText(f"\u6b3a\u8bc8\u6982\u7387: {probability:.4f}")
            self.risk_label.setText(f"\u98ce\u9669\u7b49\u7ea7: {result['risk_level']}")
            self.risk_bar.setValue(int(probability * 100))
            if result["risk_level"] == "\u9ad8":
                self.risk_label.setProperty("level", "high")
            elif result["risk_level"] == "\u4e2d":
                self.risk_label.setProperty("level", "mid")
            else:
                self.risk_label.setProperty("level", "low")
            self.style().unpolish(self.risk_label)
            self.style().polish(self.risk_label)
            self.log_panel.append_line(f"\u5355\u6761\u63a8\u7406\u5b8c\u6210\uff0c\u6982\u7387={probability:.4f}")
        except Exception as exc:
            show_error(self, "\u5355\u6761\u63a8\u7406\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u5355\u6761\u63a8\u7406\u5931\u8d25: {exc}")

    def run_batch_prediction(self) -> None:
        try:
            if self.predictor is None:
                self.load_predictor()
            if self.predictor is None:
                return
            self.batch_result = self.predictor.predict_csv(self.batch_csv_edit.text().strip())
            self.populate_batch_table(self.batch_result)
            self.log_panel.append_line(f"\u6279\u91cf\u63a8\u7406\u5b8c\u6210: {len(self.batch_result)} \u884c")
            show_info(self, "\u6279\u91cf\u63a8\u7406\u5b8c\u6210", "\u6279\u91cf\u9884\u6d4b\u7ed3\u679c\u5df2\u66f4\u65b0\u3002")
        except Exception as exc:
            show_error(self, "\u6279\u91cf\u63a8\u7406\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u6279\u91cf\u63a8\u7406\u5931\u8d25: {exc}")

    def populate_batch_table(self, frame: pd.DataFrame) -> None:
        self.batch_table.clear()
        self.batch_table.setRowCount(len(frame))
        self.batch_table.setColumnCount(len(frame.columns))
        self.batch_table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        for row in range(len(frame)):
            risk_value = str(frame.iloc[row]["RiskLevel"]) if "RiskLevel" in frame.columns else "\u4f4e"
            for column, _column_name in enumerate(frame.columns):
                value = frame.iloc[row, column]
                display = f"{value:.6f}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(display)
                if risk_value == "\u9ad8":
                    item.setBackground(QColor("#5d1d2c"))
                elif risk_value == "\u4e2d":
                    item.setBackground(QColor("#5d5120"))
                self.batch_table.setItem(row, column, item)
        self.batch_table.resizeColumnsToContents()

    def export_batch_result(self) -> None:
        try:
            if self.batch_result is None:
                raise ValueError("\u8bf7\u5148\u6267\u884c\u6279\u91cf\u63a8\u7406\u3002")
            path, _ = QFileDialog.getSaveFileName(self, "\u5bfc\u51fa CSV", str(OUTPUTS_DIR / "predictions" / "batch_result.csv"), "CSV Files (*.csv)")
            if not path:
                return
            self.batch_result.to_csv(path, index=False, encoding="utf-8-sig")
            self.log_panel.append_line(f"\u6279\u91cf\u7ed3\u679c\u5df2\u5bfc\u51fa: {path}")
            show_info(self, "\u5bfc\u51fa\u6210\u529f", f"\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230:\n{path}")
        except Exception as exc:
            show_error(self, "\u5bfc\u51fa\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u5bfc\u51fa\u5931\u8d25: {exc}")

