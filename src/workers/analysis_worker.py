from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from src.utils.logger import log_exception


class AnalysisWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, csv_path: str, sample_size: int, random_seed: int) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.random_seed = random_seed

    def run(self) -> None:
        try:
            from src.core.analysis import build_analysis_report

            self.log.emit("\u5f00\u59cb\u6267\u884c\u6570\u636e\u5206\u6790\u4efb\u52a1...")
            self.progress.emit(15)
            result = build_analysis_report(self.csv_path, sample_size=self.sample_size, random_seed=self.random_seed)
            self.progress.emit(100)
            self.log.emit("\u6570\u636e\u5206\u6790\u5b8c\u6210\u3002")
            self.finished_ok.emit(result)
        except Exception as exc:
            log_exception("AnalysisWorker failed", exc)
            self.failed.emit(str(exc))

