from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from src.utils.logger import log_exception


class EvalWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, model_path: str, preprocessor_path: str, csv_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.csv_path = csv_path
        self.device = device

    def run(self) -> None:
        try:
            from src.core.evaluator import evaluate_model

            self.log.emit("\u5f00\u59cb\u6267\u884c\u8bc4\u4f30\u4efb\u52a1...")
            self.progress.emit(20)
            result = evaluate_model(self.model_path, self.preprocessor_path, self.csv_path, device=self.device)
            self.progress.emit(100)
            self.log.emit("\u8bc4\u4f30\u5b8c\u6210\u3002")
            self.finished_ok.emit(result)
        except Exception as exc:
            log_exception("EvalWorker failed", exc)
            self.failed.emit(str(exc))

