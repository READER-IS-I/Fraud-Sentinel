from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from src.utils.logger import log_exception


class TrainWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            from src.core.trainer import train_model

            result = train_model(self.config, progress_callback=self.progress.emit, log_callback=self.log.emit)
            self.finished_ok.emit(result)
        except Exception as exc:
            log_exception("TrainWorker failed", exc)
            self.failed.emit(str(exc))
