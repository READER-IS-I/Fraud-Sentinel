from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.core.preprocessing import FraudPreprocessor, build_data_profile, coerce_types, inspect_csv_schema, prepare_feature_frame, profile_csv


def sample_stratified_frame(
    csv_path: Path | str,
    sample_size: int = 20000,
    random_seed: int = 42,
    chunk_size: int = 200000,
) -> tuple[pd.DataFrame, dict]:
    schema = inspect_csv_schema(csv_path)
    profile = profile_csv(csv_path, chunk_size=chunk_size)
    if profile.rows <= sample_size:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame = coerce_types(frame, schema=schema, require_target=schema.target_column in frame.columns)
        return frame.reset_index(drop=True), profile.to_dict()

    target_column = schema.target_column
    if target_column not in profile.column_names:
        sample_fraction = min(1.0, sample_size / max(profile.rows, 1))
        chunks: list[pd.DataFrame] = []
        for index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
            chunk = coerce_types(chunk, schema=schema, require_target=False)
            take = min(len(chunk), max(1, int(round(len(chunk) * sample_fraction))))
            chunks.append(chunk.sample(n=take, random_state=random_seed + index))
        sample = pd.concat(chunks, ignore_index=True)
        if len(sample) > sample_size:
            sample = sample.sample(n=sample_size, random_state=random_seed)
        return sample.reset_index(drop=True), profile.to_dict()

    total_fraud = int(profile.class_distribution.get("1", 0))
    total_normal = int(profile.class_distribution.get("0", 0))
    fraud_quota = min(total_fraud, max(min(sample_size // 4, 5000), min(total_fraud, 300)))
    normal_quota = min(total_normal, max(sample_size - fraud_quota, 0))
    if fraud_quota + normal_quota == 0:
        raise ValueError("No samples available for analysis.")

    fraud_fraction = fraud_quota / total_fraud if total_fraud else 0.0
    normal_fraction = normal_quota / total_normal if total_normal else 0.0
    fraud_parts: list[pd.DataFrame] = []
    normal_parts: list[pd.DataFrame] = []

    for index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        chunk = coerce_types(chunk, schema=schema, require_target=True)
        fraud_chunk = chunk[chunk[target_column] == 1]
        normal_chunk = chunk[chunk[target_column] == 0]
        if not fraud_chunk.empty and fraud_fraction > 0:
            take = min(len(fraud_chunk), max(1, int(round(len(fraud_chunk) * fraud_fraction))))
            fraud_parts.append(fraud_chunk.sample(n=take, random_state=random_seed + index))
        if not normal_chunk.empty and normal_fraction > 0:
            take = min(len(normal_chunk), max(1, int(round(len(normal_chunk) * normal_fraction))))
            normal_parts.append(normal_chunk.sample(n=take, random_state=random_seed + 1000 + index))

    fraud_frame = pd.concat(fraud_parts, ignore_index=True) if fraud_parts else pd.DataFrame(columns=profile.column_names)
    normal_frame = pd.concat(normal_parts, ignore_index=True) if normal_parts else pd.DataFrame(columns=profile.column_names)
    if len(fraud_frame) > fraud_quota:
        fraud_frame = fraud_frame.sample(n=fraud_quota, random_state=random_seed)
    if len(normal_frame) > normal_quota:
        normal_frame = normal_frame.sample(n=normal_quota, random_state=random_seed)

    sample = pd.concat([normal_frame, fraud_frame], ignore_index=True)
    if sample.empty:
        raise ValueError("Failed to build an analysis sample.")
    sample = sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return sample, profile.to_dict()


def build_analysis_report(csv_path: Path | str, sample_size: int = 20000, random_seed: int = 42) -> dict:
    sample_frame, profile = sample_stratified_frame(csv_path, sample_size=sample_size, random_seed=random_seed)
    report_profile = build_data_profile(sample_frame)
    schema_name = report_profile.schema_name
    target_column = "isFraud" if schema_name == "paysim" else "Class"

    prepared = prepare_feature_frame(sample_frame, schema_name=schema_name)
    preprocessor = FraudPreprocessor(schema_name=schema_name)
    transformed = preprocessor.fit_transform(sample_frame)

    labels = sample_frame[target_column].to_numpy(dtype=int) if target_column in sample_frame.columns else np.zeros(len(sample_frame), dtype=int)
    amount_values = np.log1p(sample_frame["amount"].to_numpy()) if "amount" in sample_frame.columns else np.log1p(sample_frame["Amount"].to_numpy())
    normal_amounts = amount_values[labels == 0].tolist()
    fraud_amounts = amount_values[labels == 1].tolist()

    type_counts = {}
    fraud_rate_by_type = {}
    if "type" in sample_frame.columns:
        counts = sample_frame.groupby("type").size().sort_values(ascending=False)
        type_counts = {str(key): int(value) for key, value in counts.items()}
        fraud_rates = sample_frame.groupby("type")[target_column].mean().sort_values(ascending=False)
        fraud_rate_by_type = {str(key): float(value) for key, value in fraud_rates.items()}

    corr_columns = [
        column
        for column in [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "origin_delta",
            "destination_delta",
            "amount_to_origin_balance_ratio",
        ]
        if column in prepared.columns
    ]
    corr_matrix = prepared[corr_columns].corr().fillna(0.0).to_numpy().tolist() if corr_columns else []

    pca = PCA(n_components=2, random_state=random_seed)
    pca_points = pca.fit_transform(transformed)
    variance_ratio = pca.explained_variance_ratio_.tolist()

    tsne_count = min(len(transformed), 2500)
    tsne_indices = np.random.default_rng(random_seed).choice(len(transformed), size=tsne_count, replace=False)
    tsne_input = transformed[tsne_indices]
    tsne_labels = labels[tsne_indices]
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=max(10, min(35, tsne_count // 20)), random_state=random_seed)
    tsne_points = tsne.fit_transform(tsne_input)

    return {
        "profile": profile,
        "sample_profile": report_profile.to_dict(),
        "class_distribution": {
            "\u6b63\u5e38": int((labels == 0).sum()),
            "\u6b3a\u8bc8": int((labels == 1).sum()),
        },
        "type_distribution": type_counts,
        "fraud_rate_by_type": fraud_rate_by_type,
        "amount_hist": {
            "normal": normal_amounts,
            "fraud": fraud_amounts,
        },
        "correlation": {
            "labels": corr_columns,
            "matrix": corr_matrix,
        },
        "pca": {
            "x": pca_points[:, 0].tolist(),
            "y": pca_points[:, 1].tolist(),
            "labels": labels.tolist(),
            "variance_ratio": variance_ratio,
        },
        "tsne": {
            "x": tsne_points[:, 0].tolist(),
            "y": tsne_points[:, 1].tolist(),
            "labels": tsne_labels.tolist(),
        },
    }

