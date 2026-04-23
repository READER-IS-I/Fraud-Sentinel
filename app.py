from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from src.utils.file_utils import LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, USER_DATA_DIR, ensure_dir, resource_path
from src.utils.logger import install_exception_hook, log_exception, setup_logging
from src.utils.runtime_env import run_environment_self_check


FONT_CANDIDATES = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "Segoe UI",
]


def load_stylesheet() -> str:
    qss_path = resource_path("assets/styles/dark_finance.qss")
    return qss_path.read_text(encoding="utf-8-sig") if qss_path.exists() else ""


def bootstrap_directories() -> None:
    paths = [
        USER_DATA_DIR,
        LOGS_DIR,
        MODELS_DIR,
        MODELS_DIR / "keys",
        MODELS_DIR / "training_runs",
        MODELS_DIR / "demo_runs",
        OUTPUTS_DIR,
        OUTPUTS_DIR / "encrypted",
        OUTPUTS_DIR / "decrypted",
        OUTPUTS_DIR / "preprocessors",
        OUTPUTS_DIR / "predictions",
        OUTPUTS_DIR / "logs",
    ]
    for path in paths:
        ensure_dir(path)


def build_app_font() -> QFont:
    families = set(QFontDatabase.families())
    for family in FONT_CANDIDATES:
        if family in families:
            font = QFont(family, 10)
            font.setHintingPreference(QFont.PreferFullHinting)
            return font
    fallback = QFont()
    fallback.setPointSize(10)
    fallback.setHintingPreference(QFont.PreferFullHinting)
    return fallback


def main() -> int:
    setup_logging()
    install_exception_hook()
    bootstrap_directories()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("\u91d1\u9274\u667a\u9632 FraudShield")
    app.setFont(build_app_font())
    try:
        from src.ui.main_window import MainWindow

        stylesheet = load_stylesheet()
        if stylesheet:
            app.setStyleSheet(stylesheet)
        report = run_environment_self_check()
        window = MainWindow()
        window.report_startup_self_check(report)
        window.show()
        return app.exec()
    except Exception as exc:
        log_exception("Application startup failed", exc)
        QMessageBox.critical(None, "启动失败", f"程序启动失败，请查看 logs\\error.log。\n\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
