from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np

from src.core.analysis import sample_stratified_frame
from src.core.evaluator import calculate_metrics
from src.core.preprocessing import FraudPreprocessor


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_reference_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int = 8,
    batch_size: int = 256,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.core.dataset import FraudTensorDataset
    from src.core.model import FraudMLP

    _set_seed(seed)
    model = FraudMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(FraudTensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    model.train()
    for _epoch in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _evaluate_model(name: str, y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    metrics = calculate_metrics(y_true, probabilities)
    roc_fpr, roc_tpr, _ = roc_curve(y_true, probabilities)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, probabilities)
    metrics["name"] = name
    metrics["roc_curve"] = {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist()}
    metrics["pr_curve"] = {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()}
    return metrics


def compare_models(
    csv_path: Path | str,
    sample_size: int = 12000,
    random_seed: int = 42,
    test_size: float = 0.2,
    smote_ratio: float = 0.2,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    from src.core.trainer import _apply_smote

    sample_frame, profile = sample_stratified_frame(csv_path, sample_size=sample_size, random_seed=random_seed)
    target_column = "isFraud" if "isFraud" in sample_frame.columns else "Class"
    x_train, x_test, y_train, y_test = train_test_split(
        sample_frame,
        sample_frame[target_column].to_numpy(dtype=int),
        test_size=test_size,
        random_state=random_seed,
        stratify=sample_frame[target_column] if sample_frame[target_column].nunique() > 1 else None,
    )

    preprocessor = FraudPreprocessor(schema_name="paysim" if "isFraud" in sample_frame.columns else None)
    train_features = preprocessor.fit_transform(x_train)
    test_features = preprocessor.transform(x_test)
    train_labels = np.asarray(y_train, dtype=int)
    test_labels = np.asarray(y_test, dtype=int)
    train_features, train_labels = _apply_smote(train_features, train_labels, random_seed, smote_ratio)

    rows: list[dict] = []
    roc_curves: dict[str, dict] = {}
    pr_curves: dict[str, dict] = {}

    classical_models = [
        ("LogReg", LogisticRegression(max_iter=400, class_weight="balanced")),
        ("KNN", KNeighborsClassifier(n_neighbors=9, weights="distance")),
        ("SVM", SVC(kernel="rbf", class_weight="balanced", probability=True, gamma="scale", C=1.0)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=random_seed)),
    ]

    for name, model in classical_models:
        started = time.time()
        model.fit(train_features, train_labels)
        probabilities = model.predict_proba(test_features)[:, 1]
        metrics = _evaluate_model(name, test_labels, probabilities)
        metrics["seconds"] = time.time() - started
        rows.append(metrics)
        roc_curves[name] = {"fpr": metrics["roc_curve"]["fpr"], "tpr": metrics["roc_curve"]["tpr"], "auc": metrics["roc_auc"]}
        pr_curves[name] = {"recall": metrics["pr_curve"]["recall"], "precision": metrics["pr_curve"]["precision"], "auc": metrics["pr_auc"]}

    mlp_started = time.time()
    mlp_probabilities = _fit_reference_mlp(train_features, train_labels, test_features, seed=random_seed)
    mlp_metrics = _evaluate_model("MLP", test_labels, mlp_probabilities)
    mlp_metrics["seconds"] = time.time() - mlp_started
    rows.append(mlp_metrics)
    roc_curves["MLP"] = {"fpr": mlp_metrics["roc_curve"]["fpr"], "tpr": mlp_metrics["roc_curve"]["tpr"], "auc": mlp_metrics["roc_auc"]}
    pr_curves["MLP"] = {"recall": mlp_metrics["pr_curve"]["recall"], "precision": mlp_metrics["pr_curve"]["precision"], "auc": mlp_metrics["pr_auc"]}

    rows.sort(key=lambda item: item["f1_score"], reverse=True)
    summary_rows = [
        {
            "model": row["name"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "seconds": row["seconds"],
        }
        for row in rows
    ]
    return {
        "profile": profile,
        "rows": summary_rows,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "best_model": summary_rows[0]["model"] if summary_rows else "--",
    }

