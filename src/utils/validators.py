from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SchemaDefinition:
    name: str
    display_name: str
    required_columns: tuple[str, ...]
    target_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    id_columns: tuple[str, ...]
    manual_input_columns: tuple[str, ...]


LEGACY_SCHEMA = SchemaDefinition(
    name="legacy_creditcard",
    display_name="Credit Card PCA Schema",
    required_columns=("Time", "Amount", *tuple(f"V{i}" for i in range(1, 29)), "Class"),
    target_column="Class",
    numeric_columns=("Time", "Amount", *tuple(f"V{i}" for i in range(1, 29))),
    categorical_columns=(),
    id_columns=(),
    manual_input_columns=("Time", "Amount", *tuple(f"V{i}" for i in range(1, 29))),
)

PAYSIM_SCHEMA = SchemaDefinition(
    name="paysim",
    display_name="PaySim Transaction Schema",
    required_columns=(
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ),
    target_column="isFraud",
    numeric_columns=("step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"),
    categorical_columns=("type",),
    id_columns=("nameOrig", "nameDest"),
    manual_input_columns=("step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"),
)

SUPPORTED_SCHEMAS = (PAYSIM_SCHEMA, LEGACY_SCHEMA)
DEFAULT_SCHEMA = PAYSIM_SCHEMA
FEATURE_COLUMNS = list(DEFAULT_SCHEMA.numeric_columns)
TARGET_COLUMN = DEFAULT_SCHEMA.target_column


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    missing_columns: list[str]
    duplicated_columns: list[str]
    has_target: bool
    schema: SchemaDefinition


def detect_schema(columns: Iterable[str]) -> SchemaDefinition:
    column_set = set(columns)
    for schema in SUPPORTED_SCHEMAS:
        if all(column in column_set for column in schema.required_columns):
            return schema
    if all(column in column_set for column in PAYSIM_SCHEMA.manual_input_columns):
        return PAYSIM_SCHEMA
    if all(column in column_set for column in LEGACY_SCHEMA.manual_input_columns):
        return LEGACY_SCHEMA
    return DEFAULT_SCHEMA


def validate_columns(columns: Iterable[str], require_target: bool = True, schema: SchemaDefinition | None = None) -> ValidationResult:
    column_list = list(columns)
    schema = schema or detect_schema(column_list)
    duplicated_columns = sorted({col for col in column_list if column_list.count(col) > 1})
    required_columns = list(schema.required_columns)
    if not require_target:
        required_columns = [column for column in required_columns if column != schema.target_column]
    missing_columns = [column for column in required_columns if column not in column_list]
    has_target = schema.target_column in column_list
    return ValidationResult(
        valid=not missing_columns and not duplicated_columns,
        missing_columns=missing_columns,
        duplicated_columns=duplicated_columns,
        has_target=has_target,
        schema=schema,
    )


def build_validation_error(validation: ValidationResult) -> str:
    messages: list[str] = []
    if validation.missing_columns:
        messages.append(f"\u7f3a\u5c11\u5fc5\u9700\u5b57\u6bb5: {', '.join(validation.missing_columns)}")
    if validation.duplicated_columns:
        messages.append(f"\u5b58\u5728\u91cd\u590d\u5b57\u6bb5: {', '.join(validation.duplicated_columns)}")
    return "\uff1b".join(messages) if messages else "\u5b57\u6bb5\u6821\u9a8c\u5931\u8d25"


def reorder_columns(df: pd.DataFrame, require_target: bool = True, schema: SchemaDefinition | None = None) -> pd.DataFrame:
    validation = validate_columns(df.columns, require_target=require_target, schema=schema)
    if not validation.valid:
        raise ValueError(build_validation_error(validation))
    ordered_columns = [
        column
        for column in validation.schema.required_columns
        if require_target or column != validation.schema.target_column
    ]
    return df.loc[:, ordered_columns].copy()


def feature_columns_for_schema(schema: SchemaDefinition) -> list[str]:
    return [column for column in schema.required_columns if column != schema.target_column]


def coerce_types(df: pd.DataFrame, schema: SchemaDefinition | None = None, require_target: bool = True) -> pd.DataFrame:
    schema = schema or detect_schema(df.columns)
    output = df.copy()
    for column in schema.numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="raise")
    for column in schema.categorical_columns + schema.id_columns:
        if column in output.columns:
            output[column] = output[column].fillna("").astype(str)
    if require_target and schema.target_column in output.columns:
        output[schema.target_column] = pd.to_numeric(output[schema.target_column], errors="raise").astype(int)
    return output

