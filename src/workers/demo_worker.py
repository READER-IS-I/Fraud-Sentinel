from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.utils.file_utils import DEMO_DIR, EXAMPLES_DIR, MODELS_DIR, create_timestamped_dir
from src.utils.logger import log_exception


class DemoWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def run(self) -> None:
        try:
            from src.core.predictor import FraudPredictor
            from src.core.trainer import TrainingConfig, train_model

            run_dir = create_timestamped_dir(MODELS_DIR / "demo_runs", prefix="demo")
            config = TrainingConfig(
                csv_path=str(DEMO_DIR / "sample_demo.csv"),
                output_dir=str(run_dir),
                epochs=12,
                batch_size=64,
                learning_rate=0.001,
                random_seed=42,
                val_size=0.2,
                test_size=0.2,
                smote_ratio=0.2,
            )

            def scaled_progress(value: int) -> None:
                self.progress.emit(10 + int(value * 0.6))

            self.log.emit("\u5f00\u59cb\u6267\u884c\u4e00\u952e\u6f14\u793a\u6d41\u7a0b\uff1a\u8bad\u7ec3 + \u6279\u91cf\u63a8\u7406")
            self.progress.emit(5)
            train_result = train_model(config, progress_callback=scaled_progress, log_callback=self.log.emit)
            self.progress.emit(80)
            predictor = FraudPredictor(Path(train_result["model_path"]), Path(train_result["preprocessor_path"]))
            export_path = run_dir / "demo_batch_result.csv"
            batch_result = predictor.predict_csv(EXAMPLES_DIR / "sample_inference.csv", export_path)
            preview = batch_result.sort_values("FraudProbability", ascending=False).head(12)
            high_risk_count = int((batch_result["RiskLevel"] == "\u9ad8").sum())
            self.progress.emit(100)
            self.log.emit("\u793a\u4f8b\u6f14\u793a\u5df2\u5b8c\u6210\u3002")
            self.finished_ok.emit(
                {
                    "model_path": train_result["model_path"],
                    "preprocessor_path": train_result["preprocessor_path"],
                    "result_csv": str(export_path),
                    "high_risk_count": high_risk_count,
                    "max_probability": float(batch_result["FraudProbability"].max()),
                    "preview": preview.to_dict(orient="records"),
                }
            )
        except Exception as exc:
            log_exception("DemoWorker failed", exc)
            self.failed.emit(str(exc))

