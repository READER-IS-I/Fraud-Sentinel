from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.core.preprocessing import FraudPreprocessor, build_data_profile, inspect_csv_schema, load_dataframe, profile_csv
from src.ui.widgets import CardFrame, HighlightButton, LogPanel, create_path_row, labeled_value, show_error, show_info
from src.utils.file_utils import OUTPUTS_DIR, get_dialog_start_dir
from src.utils.plot_utils import PlotCanvas, draw_class_distribution, draw_empty


class DataPage(QWidget):
    page_title = "\u6570\u636e\u5bfc\u5165\u4e0e\u9884\u5904\u7406"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_frame: pd.DataFrame | None = None
        self.preprocessor: FraudPreprocessor | None = None
        self.transformed_frame: pd.DataFrame | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        control_card = CardFrame("\u6570\u636e\u6587\u4ef6", "\u652f\u6301 PaySim \u539f\u59cb\u4ea4\u6613\u6570\u636e\u96c6\u4e0e\u517c\u5bb9\u65e7\u7248 PCA \u793a\u4f8b\u6570\u636e\u3002")
        self.csv_edit = QLineEdit()
        self.csv_edit.setPlaceholderText("\u9009\u62e9 CSV \u6587\u4ef6\uff0c\u4f8b\u5982 PaySim \u4ea4\u6613\u6d41\u6c34")
        browse_button = HighlightButton("\u6d4f\u89c8", variant="secondary")
        browse_button.clicked.connect(self.browse_csv)
        control_card.layout.addWidget(create_path_row("CSV", self.csv_edit, browse_button))

        control_row = QHBoxLayout()
        self.scaler_combo = QComboBox()
        self.scaler_combo.addItems(["standard", "robust"])
        self.scaler_combo.setMinimumWidth(180)
        analyze_button = HighlightButton("\u8bfb\u53d6\u5206\u6790", variant="primary")
        analyze_button.clicked.connect(self.load_profile)
        preprocess_button = HighlightButton("\u6267\u884c\u9884\u5904\u7406", variant="secondary")
        preprocess_button.clicked.connect(self.run_preprocess)
        save_button = HighlightButton("\u4fdd\u5b58\u9884\u5904\u7406\u5668", variant="ghost")
        save_button.clicked.connect(self.save_preprocessor)
        control_row.addWidget(QLabel("\u7f29\u653e\u5668"))
        control_row.addWidget(self.scaler_combo)
        control_row.addWidget(analyze_button)
        control_row.addWidget(preprocess_button)
        control_row.addWidget(save_button)
        control_row.addStretch(1)
        control_card.layout.addLayout(control_row)
        root.addWidget(control_card)

        info_card = CardFrame("\u6570\u636e\u6982\u89c8", "\u5927\u6587\u4ef6\u4f1a\u81ea\u52a8\u5207\u6362\u4e3a\u5206\u5757\u7edf\u8ba1\u6a21\u5f0f\uff0c\u907f\u514d\u754c\u9762\u5361\u6b7b\u3002")
        info_grid = QGridLayout()
        info_grid.setSpacing(12)
        self.schema_tile = labeled_value("\u6570\u636e\u7ed3\u6784")
        self.rows_tile = labeled_value("\u6837\u672c\u884c\u6570")
        self.cols_tile = labeled_value("\u5b57\u6bb5\u5217\u6570")
        self.class_tile = labeled_value("\u7c7b\u522b\u5206\u5e03")
        self.missing_tile = labeled_value("\u7f3a\u5931\u503c")
        for index, tile in enumerate([self.schema_tile, self.rows_tile, self.cols_tile, self.class_tile, self.missing_tile]):
            info_grid.addWidget(tile, 0, index)
        info_card.layout.addLayout(info_grid)
        self.distribution_canvas = PlotCanvas(width=4.8, height=2.8)
        draw_empty(self.distribution_canvas.axes, "\u7c7b\u522b\u5206\u5e03", "\u8bfb\u53d6 CSV \u540e\u663e\u793a\u6837\u672c\u5206\u5e03")
        self.distribution_canvas.draw()
        info_card.layout.addWidget(self.distribution_canvas)
        root.addWidget(info_card)

        table_card = CardFrame("\u6837\u672c\u9884\u89c8", "\u9884\u89c8\u524d 10 \u884c\u539f\u59cb\u6216\u9884\u5904\u7406\u540e\u6570\u636e\u3002")
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        table_card.layout.addWidget(self.table)
        root.addWidget(table_card)

        log_card = CardFrame("\u5904\u7406\u65e5\u5fd7", "\u5b57\u6bb5\u6821\u9a8c\u3001\u9884\u5904\u7406\u4e0e\u4fdd\u5b58\u64cd\u4f5c\u90fd\u4f1a\u5728\u8fd9\u91cc\u8bb0\u5f55\u3002")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9 CSV", get_dialog_start_dir(), "CSV Files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def _append_log(self, text: str) -> None:
        self.log_panel.append_line(text)

    def _is_large_file(self, csv_path: str) -> bool:
        try:
            return Path(csv_path).stat().st_size >= 64 * 1024 * 1024
        except FileNotFoundError:
            return False

    def load_profile(self) -> None:
        try:
            csv_path = self.csv_edit.text().strip()
            if not csv_path:
                raise ValueError("\u8bf7\u5148\u9009\u62e9 CSV \u6587\u4ef6\u3002")
            schema = inspect_csv_schema(csv_path)
            large_file = self._is_large_file(csv_path)
            if large_file:
                profile = profile_csv(csv_path)
                preview = load_dataframe(csv_path, nrows=10)
                self._append_log("\u5df2\u542f\u7528\u5927\u6587\u4ef6\u5206\u5757\u7edf\u8ba1\u6a21\u5f0f\u3002")
            else:
                preview = load_dataframe(csv_path)
                profile = build_data_profile(preview, schema=schema)
            self.current_frame = preview.copy()
            self.schema_tile.value_label.setText(schema.display_name)
            self.rows_tile.value_label.setText(str(profile.rows))
            self.cols_tile.value_label.setText(str(profile.columns))
            class_text = " / ".join(f"{key}: {value}" for key, value in profile.class_distribution.items()) if profile.class_distribution else "--"
            self.class_tile.value_label.setText(class_text)
            missing_text = str(profile.missing_total) if not profile.missing_by_column else ", ".join(f"{k}:{v}" for k, v in profile.missing_by_column.items())
            self.missing_tile.value_label.setText(missing_text)
            if profile.class_distribution:
                actual = {"\u6b63\u5e38": profile.class_distribution.get("0", 0), "\u6b3a\u8bc8": profile.class_distribution.get("1", 0)}
                draw_class_distribution(self.distribution_canvas.axes, actual)
            else:
                draw_empty(self.distribution_canvas.axes, "\u7c7b\u522b\u5206\u5e03", "\u5f53\u524d\u6587\u4ef6\u4e0d\u542b\u76ee\u6807\u6807\u7b7e")
            self.distribution_canvas.draw()
            self.populate_table(preview.head(10))
            self._append_log(f"\u5df2\u8bfb\u53d6\u6587\u4ef6: {csv_path}")
            self._append_log(f"\u8bc6\u522b\u5230\u6570\u636e\u7ed3\u6784: {schema.display_name}")
        except Exception as exc:
            show_error(self, "\u6570\u636e\u8bfb\u53d6\u5931\u8d25", str(exc))
            self._append_log(f"\u8bfb\u53d6\u5931\u8d25: {exc}")

    def run_preprocess(self) -> None:
        try:
            if self.current_frame is None:
                self.load_profile()
            if self.current_frame is None:
                return
            self.preprocessor = FraudPreprocessor(self.scaler_combo.currentText())
            self.preprocessor.fit(self.current_frame)
            self.transformed_frame = self.preprocessor.transform_to_dataframe(self.current_frame)
            preview = self.transformed_frame.head(10).copy()
            target_column = getattr(self.preprocessor.schema, "target_column", "")
            if target_column and target_column in self.current_frame.columns:
                preview[target_column] = self.current_frame[target_column].head(10).tolist()
            self.populate_table(preview)
            self._append_log(f"\u9884\u5904\u7406\u5b8c\u6210\uff0cscaler={self.scaler_combo.currentText()}")
            show_info(self, "\u9884\u5904\u7406\u5b8c\u6210", "\u5df2\u5b8c\u6210\u5b57\u6bb5\u7f29\u653e\u4e0e\u7279\u5f81\u6784\u5efa\u3002")
        except Exception as exc:
            show_error(self, "\u9884\u5904\u7406\u5931\u8d25", str(exc))
            self._append_log(f"\u9884\u5904\u7406\u5931\u8d25: {exc}")

    def save_preprocessor(self) -> None:
        try:
            if self.preprocessor is None:
                raise ValueError("\u8bf7\u5148\u6267\u884c\u9884\u5904\u7406\u3002")
            default_path = OUTPUTS_DIR / "preprocessors" / "manual_preprocessor.joblib"
            path, _ = QFileDialog.getSaveFileName(self, "\u4fdd\u5b58\u9884\u5904\u7406\u5668", str(default_path), "Joblib Files (*.joblib)")
            if not path:
                return
            self.preprocessor.save(path)
            self._append_log(f"\u9884\u5904\u7406\u5668\u5df2\u4fdd\u5b58: {path}")
            show_info(self, "\u4fdd\u5b58\u6210\u529f", f"\u9884\u5904\u7406\u5668\u5df2\u4fdd\u5b58\u5230:\n{path}")
        except Exception as exc:
            show_error(self, "\u4fdd\u5b58\u5931\u8d25", str(exc))
            self._append_log(f"\u4fdd\u5b58\u5931\u8d25: {exc}")

    def populate_table(self, frame: pd.DataFrame) -> None:
        self.table.clear()
        self.table.setColumnCount(len(frame.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        self.table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, _column_name in enumerate(frame.columns):
                value = frame.iloc[row, column]
                display = f"{value:.6f}" if isinstance(value, float) else str(value)
                self.table.setItem(row, column, QTableWidgetItem(display))
        self.table.resizeColumnsToContents()
