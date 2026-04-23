from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.core.model import load_model_checkpoint
from src.core.preprocessing import FraudPreprocessor, load_dataframe
from src.utils.validators import detect_schema, validate_columns


LABEL_NORMAL = "\u6b63\u5e38"
LABEL_FRAUD = "\u6b3a\u8bc8"


def infer_probabilities(model, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probabilities


def calculate_metrics(y_true: np.ndarray, fraud_probabilities: np.ndarray) -> dict:
    predicted = (fraud_probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "predicted_distribution": {LABEL_NORMAL: int((predicted == 0).sum()), LABEL_FRAUD: int((predicted == 1).sum())},
        "actual_distribution": {LABEL_NORMAL: int((y_true == 0).sum()), LABEL_FRAUD: int((y_true == 1).sum())},
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, fraud_probabilities))
        fpr, tpr, _ = roc_curve(y_true, fraud_probabilities)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, fraud_probabilities)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()}
        metrics["pr_auc"] = float(auc(recall_curve, precision_curve))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["roc_curve"] = {"fpr": [], "tpr": []}
        metrics["pr_curve"] = {"precision": [], "recall": []}
    return metrics


def evaluate_model(model_path: Path | str, preprocessor_path: Path | str, csv_path: Path | str, device: str = "cpu") -> dict:
    frame = load_dataframe(csv_path)
    schema = detect_schema(frame.columns)
    validation = validate_columns(frame.columns, require_target=True, schema=schema)
    if not validation.valid:
        missing = ", ".join(validation.missing_columns)
        raise ValueError(f"\u8bc4\u4f30\u6570\u636e\u7f3a\u5c11\u5fc5\u9700\u5b57\u6bb5: {missing}")

    model, checkpoint = load_model_checkpoint(model_path, device=device)
    model.to(device)
    preprocessor = FraudPreprocessor.load(preprocessor_path)

    labels = frame[schema.target_column].to_numpy(dtype=int)
    features = preprocessor.transform(frame)
    probabilities = infer_probabilities(model, features, device=device)
    predicted = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(labels, probabilities)
    metrics["sample_count"] = int(len(frame))
    metrics["model_path"] = str(model_path)
    metrics["preprocessor_path"] = str(preprocessor_path)
    metrics["feature_columns"] = checkpoint.get("feature_columns", getattr(preprocessor, "feature_columns", []))
    metrics["schema_name"] = schema.name
    metrics["preview"] = pd.DataFrame(
        {
            "FraudProbability": np.round(probabilities, 6),
            "PredictedClass": predicted,
            "PredictedLabel": [LABEL_FRAUD if value == 1 else LABEL_NORMAL for value in predicted],
            "TrueClass": labels,
            "TrueLabel": [LABEL_FRAUD if value == 1 else LABEL_NORMAL for value in labels],
        }
    ).head(20).to_dict(orient="records")
    return metrics

