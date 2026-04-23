from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from src.utils.logger import log_exception


class CompareWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, csv_path: str, sample_size: int, random_seed: int, smote_ratio: float) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.smote_ratio = smote_ratio

    def run(self) -> None:
        try:
            from src.core.model_compare import compare_models

            self.log.emit("\u5f00\u59cb\u6267\u884c\u6a21\u578b\u5bf9\u6bd4\u4efb\u52a1...")
            self.progress.emit(15)
            result = compare_models(
                self.csv_path,
                sample_size=self.sample_size,
                random_seed=self.random_seed,
                smote_ratio=self.smote_ratio,
            )
            self.progress.emit(100)
            self.log.emit("\u6a21\u578b\u5bf9\u6bd4\u5b8c\u6210\u3002")
            self.finished_ok.emit(result)
        except Exception as exc:
            log_exception("CompareWorker failed", exc)
            self.failed.emit(str(exc))

