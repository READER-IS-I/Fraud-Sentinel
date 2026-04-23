from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.dataset import FraudTensorDataset, create_train_val_test_split
from src.core.evaluator import calculate_metrics, infer_probabilities
from src.core.model import FraudMLP, save_model_checkpoint
from src.core.preprocessing import FraudPreprocessor, build_data_profile, load_dataframe
from src.utils.file_utils import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.validators import detect_schema


@dataclass
class TrainingConfig:
    csv_path: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 42
    val_size: float = 0.2
    test_size: float = 0.2
    smote_ratio: float = 0.2
    scaler_type: str = "standard"
    device: str = "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested_device: str) -> str:
    return "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"


def _log(message: str, callback=None) -> None:
    get_logger().info(message)
    if callback:
        callback(message)


def _apply_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    smote_ratio: float,
    log_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(y_train, return_counts=True)
    distribution = {int(label): int(count) for label, count in zip(values, counts)}
    _log(f"\u8bad\u7ec3\u96c6\u539f\u59cb\u7c7b\u522b\u5206\u5e03: {distribution}", log_callback)
    if smote_ratio <= 0:
        _log("SMOTE \u76ee\u6807\u6bd4\u4f8b <= 0\uff0c\u5c06\u8df3\u8fc7\u8fc7\u91c7\u6837\u3002", log_callback)
        return x_train, y_train
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    if len(values) < 2 or minority_count < 2:
        _log("\u5f53\u524d\u6837\u672c\u4e0d\u6ee1\u8db3 SMOTE \u6761\u4ef6\uff0c\u5c06\u8df3\u8fc7\u8fc7\u91c7\u6837\u3002", log_callback)
        return x_train, y_train
    current_ratio = minority_count / majority_count if majority_count else 0.0
    if smote_ratio <= current_ratio:
        _log(f"\u5f53\u524d\u5c11\u6570\u7c7b\u5360\u6bd4 {current_ratio:.4f} \u5df2\u4e0d\u4f4e\u4e8e\u76ee\u6807 {smote_ratio:.4f}\uff0c\u8df3\u8fc7 SMOTE\u3002", log_callback)
        return x_train, y_train
    k_neighbors = min(5, minority_count - 1)
    if k_neighbors < 1:
        _log("SMOTE \u53ef\u7528\u90bb\u5c45\u6570\u4e0d\u8db3\uff0c\u5c06\u8df3\u8fc7\u8fc7\u91c7\u6837\u3002", log_callback)
        return x_train, y_train
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        raise RuntimeError(
            "SMOTE \u4f9d\u8d56 imbalanced-learn \u5bfc\u5165\u5931\u8d25\uff0c"
            "\u8fd9\u901a\u5e38\u8868\u793a PyInstaller \u672a\u5b8c\u6574\u6536\u96c6 imblearn \u7684\u5305\u6570\u636e\u6216 metadata\u3002"
        ) from exc
    sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=smote_ratio)
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    values_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
    distribution_resampled = {int(label): int(count) for label, count in zip(values_resampled, counts_resampled)}
    _log(f"SMOTE \u540e\u7c7b\u522b\u5206\u5e03: {distribution_resampled} | target_ratio={smote_ratio:.2f}", log_callback)
    return x_resampled, y_resampled


def train_model(config: TrainingConfig, progress_callback=None, log_callback=None) -> dict:
    device = _resolve_device(config.device)
    set_global_seed(config.random_seed)
    output_dir = ensure_dir(config.output_dir)
    _log(f"\u5f00\u59cb\u52a0\u8f7d\u8bad\u7ec3\u6570\u636e: {config.csv_path}", log_callback)
    frame = load_dataframe(config.csv_path)
    profile = build_data_profile(frame)
    schema = detect_schema(frame.columns)
    _log(f"\u6570\u636e\u89c4\u6a21: {profile.rows} \u884c x {profile.columns} \u5217", log_callback)
    _log(f"\u68c0\u6d4b\u5230\u6570\u636e\u7ed3\u6784: {schema.display_name}", log_callback)
    _log(f"\u8bad\u7ec3\u8bbe\u5907: {device.upper()}", log_callback)

    split = create_train_val_test_split(
        frame,
        random_seed=config.random_seed,
        val_size=config.val_size,
        test_size=config.test_size,
    )
    split_summary = {
        "train": int(len(split.x_train)),
        "val": int(len(split.x_val)),
        "test": int(len(split.x_test)),
    }
    _log(
        f"\u6570\u636e\u5212\u5206: train={split_summary['train']} / val={split_summary['val']} / test={split_summary['test']}",
        log_callback,
    )

    preprocessor = FraudPreprocessor(scaler_type=config.scaler_type, schema_name=split.schema_name)
    x_train = preprocessor.fit_transform(split.x_train)
    x_val = preprocessor.transform(split.x_val)
    x_test = preprocessor.transform(split.x_test)
    y_train = split.y_train.to_numpy(dtype=int)
    y_val = split.y_val.to_numpy(dtype=int)
    y_test = split.y_test.to_numpy(dtype=int)

    x_train, y_train = _apply_smote(
        x_train,
        y_train,
        config.random_seed,
        config.smote_ratio,
        log_callback=log_callback,
    )

    train_loader = DataLoader(FraudTensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(FraudTensorDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)

    input_dim = int(x_train.shape[1])
    model = FraudMLP(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history_train: list[float] = []
    history_val: list[float] = []
    labels_buffer: list[int] = []
    probabilities: list[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        labels_buffer = []
        probabilities = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_losses.append(float(loss.item()))
                probabilities.extend(probs.cpu().numpy().tolist())
                labels_buffer.extend(labels.cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history_train.append(train_loss)
        history_val.append(val_loss)
        validation_metrics = calculate_metrics(np.array(labels_buffer), np.array(probabilities))
        _log(
            f"Epoch {epoch:02d}/{config.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | f1={validation_metrics['f1_score']:.4f} | "
            f"recall={validation_metrics['recall']:.4f}",
            log_callback,
        )
        if progress_callback:
            progress_callback(int(epoch / config.epochs * 100))

    checkpoint_config = asdict(config) | {"schema_name": schema.name}
    model_path = save_model_checkpoint(output_dir / "model.pt", model, config=checkpoint_config, feature_columns=preprocessor.feature_columns)
    preprocessor_path = preprocessor.save(output_dir / "preprocessor.joblib")

    validation_metrics = calculate_metrics(np.array(labels_buffer), np.array(probabilities))
    test_probabilities = infer_probabilities(model, x_test, device=device)
    test_metrics = calculate_metrics(y_test, test_probabilities)
    _log(
        f"\u6d4b\u8bd5\u96c6\u7ed3\u679c: accuracy={test_metrics['accuracy']:.4f} | f1={test_metrics['f1_score']:.4f} | "
        f"recall={test_metrics['recall']:.4f}",
        log_callback,
    )

    final_metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "history": {"train_loss": history_train, "val_loss": history_val},
        "config": checkpoint_config,
        "profile": profile.to_dict(),
        "split_summary": split_summary,
        "generalization": {
            "schema": schema.display_name,
            "split_strategy": f"train/val/test = {split_summary['train']}/{split_summary['val']}/{split_summary['test']}",
            "smote_scope": f"SMOTE only on the training split (target_ratio={config.smote_ratio:.2f})",
            "scaler": config.scaler_type,
            "regularization": f"Dropout + weight_decay={config.weight_decay}",
            "leakage_guard": "Validation and test sets are never resampled",
        },
    }
    metrics_path = save_json(output_dir / "metrics.json", final_metrics)
    _log(f"\u8bad\u7ec3\u5b8c\u6210\uff0c\u6a21\u578b\u5df2\u4fdd\u5b58\u5230: {model_path}", log_callback)
    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "metrics_path": str(metrics_path),
        "history": final_metrics["history"],
        "metrics": final_metrics,
        "device": device,
    }

