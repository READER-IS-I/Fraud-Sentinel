from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils.validators import detect_schema, feature_columns_for_schema, reorder_columns


@dataclass
class SplitResult:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    schema_name: str


class FraudTensorDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    ordered = reorder_columns(frame, require_target=True)
    schema = detect_schema(ordered.columns)
    feature_columns = feature_columns_for_schema(schema)
    return ordered[feature_columns], ordered[schema.target_column], schema.name


def create_train_val_test_split(
    frame: pd.DataFrame,
    random_seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> SplitResult:
    x, y, schema_name = split_features_and_target(frame)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y if y.nunique() > 1 else None,
    )

    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=y_train_val if y_train_val.nunique() > 1 else None,
    )

    return SplitResult(
        x_train=x_train.reset_index(drop=True),
        x_val=x_val.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        schema_name=schema_name,
    )

