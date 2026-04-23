from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.evaluator import infer_probabilities
from src.core.model import load_model_checkpoint
from src.core.preprocessing import FraudPreprocessor, load_dataframe
from src.utils.validators import PAYSIM_SCHEMA, SchemaDefinition, detect_schema, validate_columns


LABEL_NORMAL = "\u6b63\u5e38"
LABEL_FRAUD = "\u6b3a\u8bc8"


def risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "\u9ad8"
    if probability >= 0.45:
        return "\u4e2d"
    return "\u4f4e"


@dataclass
class FraudPredictor:
    model_path: Path
    preprocessor_path: Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model, self.checkpoint = load_model_checkpoint(self.model_path, device=self.device)
        self.model.to(self.device)
        self.preprocessor = FraudPreprocessor.load(self.preprocessor_path)
        self.schema: SchemaDefinition = getattr(self.preprocessor, "schema", None) or PAYSIM_SCHEMA

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        schema = detect_schema(frame.columns)
        validation = validate_columns(frame.columns, require_target=False, schema=schema)
        if validation.missing_columns:
            raise ValueError(f"\u63a8\u7406\u8f93\u5165\u7f3a\u5c11\u5fc5\u9700\u5b57\u6bb5: {', '.join(validation.missing_columns)}")

        features = self.preprocessor.transform(frame)
        probabilities = infer_probabilities(self.model, features, device=self.device)
        predicted_classes = (probabilities >= 0.5).astype(int)
        result = frame.copy()
        result["PredictedClass"] = predicted_classes
        result["FraudProbability"] = np.round(probabilities, 6)
        result["RiskLevel"] = [risk_level(probability) for probability in probabilities]
        result["ResultLabel"] = [LABEL_FRAUD if value == 1 else LABEL_NORMAL for value in predicted_classes]
        return result

    def predict_single(self, record: dict[str, object]) -> dict:
        columns = list(getattr(self.preprocessor, "raw_input_columns", []) or self.schema.manual_input_columns)
        frame = pd.DataFrame([record], columns=columns)
        result = self.predict_frame(frame).iloc[0]
        probability = float(result["FraudProbability"])
        return {
            "predicted_class": int(result["PredictedClass"]),
            "label": str(result["ResultLabel"]),
            "fraud_probability": probability,
            "risk_level": str(result["RiskLevel"]),
        }

    def predict_csv(self, csv_path: Path | str, export_path: Path | str | None = None) -> pd.DataFrame:
        frame = load_dataframe(csv_path)
        result = self.predict_frame(frame)
        if export_path:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(export_path, index=False, encoding="utf-8-sig")
        return result

