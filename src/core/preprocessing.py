
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.utils.validators import (
    DEFAULT_SCHEMA,
    LEGACY_SCHEMA,
    PAYSIM_SCHEMA,
    SchemaDefinition,
    coerce_types,
    detect_schema,
    reorder_columns,
    validate_columns,
)

SCALER_MAP = {"standard": StandardScaler, "robust": RobustScaler}

PAYSIM_ENGINEERED_NUMERIC = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_delta",
    "destination_delta",
    "amount_to_origin_balance_ratio",
    "origin_balance_error",
    "destination_balance_error",
    "origin_zero_after",
    "destination_zero_before",
    "destination_zero_after",
    "is_dest_merchant",
    "is_dest_customer",
)


@dataclass
class DataProfile:
    rows: int
    columns: int
    missing_total: int
    missing_by_column: dict[str, int]
    class_distribution: dict[str, int]
    column_names: list[str]
    schema_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "missing_total": self.missing_total,
            "missing_by_column": self.missing_by_column,
            "class_distribution": self.class_distribution,
            "column_names": self.column_names,
            "schema_name": self.schema_name,
        }


class FraudPreprocessor:
    def __init__(self, scaler_type: str = "standard", schema_name: str | None = None) -> None:
        if scaler_type not in SCALER_MAP:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        self.scaler_type = scaler_type
        self.schema_name = schema_name
        self.schema: SchemaDefinition = DEFAULT_SCHEMA
        self.transformer: ColumnTransformer | None = None
        self.feature_columns: list[str] = []
        self.raw_input_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "FraudPreprocessor":
        schema = self._resolve_schema(frame)
        prepared = self._prepare_frame(frame, schema=schema)
        self.transformer = self._build_transformer(schema)
        self.transformer.fit(prepared)
        self.feature_columns = list(self.transformer.get_feature_names_out())
        self.raw_input_columns = list(schema.manual_input_columns)
        return self

    def transform(self, frame: pd.DataFrame):
        schema = self._resolve_schema(frame)
        prepared = self._prepare_frame(frame, schema=schema)
        if self.transformer is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.transformer.transform(prepared)

    def fit_transform(self, frame: pd.DataFrame):
        self.fit(frame)
        return self.transform(frame)

    def transform_to_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(frame)
        columns = self.feature_columns or [f"f_{i}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=columns)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "FraudPreprocessor":
        return joblib.load(path)

    def _resolve_schema(self, frame: pd.DataFrame) -> SchemaDefinition:
        if self.schema_name == LEGACY_SCHEMA.name:
            self.schema = LEGACY_SCHEMA
        elif self.schema_name == PAYSIM_SCHEMA.name:
            self.schema = PAYSIM_SCHEMA
        else:
            self.schema = detect_schema(frame.columns)
        return self.schema

    def _build_transformer(self, schema: SchemaDefinition) -> ColumnTransformer:
        scaler_cls = SCALER_MAP[self.scaler_type]
        if schema.name == LEGACY_SCHEMA.name:
            numeric_columns = list(LEGACY_SCHEMA.numeric_columns)
            categorical_columns: list[str] = []
        else:
            numeric_columns = list(PAYSIM_ENGINEERED_NUMERIC)
            categorical_columns = ["type"]

        transformers: list[tuple[str, object, list[str]]] = [("num", scaler_cls(), numeric_columns)]
        if categorical_columns:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    def _prepare_frame(self, frame: pd.DataFrame, schema: SchemaDefinition) -> pd.DataFrame:
        ordered = reorder_columns(frame, require_target=False, schema=schema)
        typed = coerce_types(ordered, schema=schema, require_target=False)
        if schema.name == LEGACY_SCHEMA.name:
            return typed.loc[:, list(schema.numeric_columns)].copy()
        prepared = typed.copy()
        prepared["origin_delta"] = prepared["oldbalanceOrg"] - prepared["newbalanceOrig"]
        prepared["destination_delta"] = prepared["newbalanceDest"] - prepared["oldbalanceDest"]
        prepared["amount_to_origin_balance_ratio"] = np.where(
            prepared["oldbalanceOrg"] > 0,
            prepared["amount"] / prepared["oldbalanceOrg"],
            0.0,
        )
        prepared["origin_balance_error"] = np.abs(prepared["oldbalanceOrg"] - prepared["newbalanceOrig"] - prepared["amount"])
        prepared["destination_balance_error"] = np.abs(prepared["newbalanceDest"] - prepared["oldbalanceDest"] - prepared["amount"])
        prepared["origin_zero_after"] = (prepared["newbalanceOrig"] == 0).astype(int)
        prepared["destination_zero_before"] = (prepared["oldbalanceDest"] == 0).astype(int)
        prepared["destination_zero_after"] = (prepared["newbalanceDest"] == 0).astype(int)
        prepared["is_dest_merchant"] = prepared["nameDest"].str.startswith("M").astype(int)
        prepared["is_dest_customer"] = prepared["nameDest"].str.startswith("C").astype(int)
        return prepared.loc[:, list(PAYSIM_ENGINEERED_NUMERIC) + ["type"]].copy()

def prepare_feature_frame(frame: pd.DataFrame, schema_name: str | None = None) -> pd.DataFrame:
    preprocessor = FraudPreprocessor(schema_name=schema_name)
    schema = preprocessor._resolve_schema(frame)
    return preprocessor._prepare_frame(frame, schema=schema)


def inspect_csv_schema(path: Path | str) -> SchemaDefinition:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return detect_schema(header)


def load_dataframe(path: Path | str, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    schema = inspect_csv_schema(path)
    if schema.name == PAYSIM_SCHEMA.name:
        dtype_map = {
            "step": "int64",
            "type": "string",
            "amount": "float64",
            "nameOrig": "string",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "nameDest": "string",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int64",
            "isFlaggedFraud": "int64",
        }
        frame = pd.read_csv(path, dtype=dtype_map, nrows=nrows, usecols=usecols, low_memory=False)
    else:
        frame = pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)
    validation = validate_columns(frame.columns, require_target=False, schema=schema)
    if validation.duplicated_columns:
        raise ValueError(f"CSV contains duplicated columns: {', ' .join(validation.duplicated_columns)}")
    return coerce_types(frame, schema=schema, require_target=schema.target_column in frame.columns)


def build_data_profile(frame: pd.DataFrame, schema: SchemaDefinition | None = None) -> DataProfile:
    schema = schema or detect_schema(frame.columns)
    class_distribution: dict[str, int] = {}
    if schema.target_column in frame.columns:
        counts = frame[schema.target_column].value_counts().sort_index()
        class_distribution = {str(int(key)): int(value) for key, value in counts.items()}
    missing_by_column = {column: int(value) for column, value in frame.isna().sum().items() if int(value) > 0}
    return DataProfile(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        missing_total=int(frame.isna().sum().sum()),
        missing_by_column=missing_by_column,
        class_distribution=class_distribution,
        column_names=list(frame.columns),
        schema_name=schema.name,
    )


def profile_csv(path: Path | str, chunk_size: int = 200_000) -> DataProfile:
    schema = inspect_csv_schema(path)
    total_rows = 0
    total_missing = 0
    missing_by_column: dict[str, int] = {}
    class_distribution: dict[str, int] = {}
    columns: list[str] | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk = coerce_types(chunk, schema=schema, require_target=schema.target_column in chunk.columns)
        total_rows += len(chunk)
        total_missing += int(chunk.isna().sum().sum())
        chunk_missing = chunk.isna().sum()
        for column, value in chunk_missing.items():
            if int(value) > 0:
                missing_by_column[column] = missing_by_column.get(column, 0) + int(value)
        if schema.target_column in chunk.columns:
            counts = chunk[schema.target_column].value_counts().sort_index()
            for key, value in counts.items():
                label = str(int(key))
                class_distribution[label] = class_distribution.get(label, 0) + int(value)
        if columns is None:
            columns = list(chunk.columns)
    return DataProfile(
        rows=total_rows,
        columns=len(columns or []),
        missing_total=total_missing,
        missing_by_column=missing_by_column,
        class_distribution=class_distribution,
        column_names=columns or [],
        schema_name=schema.name,
    )
