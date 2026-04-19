"""Geometry helpers for projecting real UDC update vectors into 3D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TokenGeometry3D:
    token_index: int
    token: str
    points: list[list[float]]
    deltas: list[list[float]]
    segment_cosines: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryProjection3D:
    method: str
    num_components: int
    explained_variance_ratio: list[float]
    num_layers: int
    token_paths: list[TokenGeometry3D]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_paths"] = [path.to_dict() for path in self.token_paths]
        return payload


@dataclass(frozen=True)
class ResponseUpdateVectors:
    response_tokens: list[str]
    deltas_per_token: list[np.ndarray]
    num_layers: int


def _safe_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    value = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    if not np.isfinite(value):
        return 0.0
    return value


def _fit_pca_basis(matrix: np.ndarray, num_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("PCA input must be a 2D matrix.")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] == 0 or centered.shape[1] == 0:
        basis = np.eye(num_components, dtype=np.float64)
        variance = np.zeros(num_components, dtype=np.float64)
        return basis, variance

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = min(num_components, vh.shape[0], vh.shape[1])
    basis = vh[:components].T

    if components < num_components:
        pad = np.zeros((basis.shape[0], num_components - components), dtype=np.float64)
        basis = np.concatenate([basis, pad], axis=1)

    total = float(np.sum(singular_values ** 2))
    if total <= 1e-12:
        variance = np.zeros(num_components, dtype=np.float64)
    else:
        explained = (singular_values[:components] ** 2) / total
        variance = np.zeros(num_components, dtype=np.float64)
        variance[:components] = explained

    return basis, variance


def extract_response_update_vectors(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    response_start: int,
    response_end: int,
    *,
    response_tokens: list[str] | None = None,
) -> ResponseUpdateVectors:
    if response_end <= response_start:
        raise ValueError("Empty response span.")

    num_layers = len(hidden_states) - 1
    if num_layers < 1:
        raise ValueError("Need at least one update vector to build geometry.")

    token_delta_arrays: list[np.ndarray] = []
    for token_index in range(response_start, response_end):
        trajectory = [
            hidden_states[layer_index][0, token_index, :].detach().float().cpu().numpy()
            for layer_index in range(num_layers + 1)
        ]
        deltas = np.stack(
            [trajectory[layer_index + 1] - trajectory[layer_index] for layer_index in range(num_layers)],
            axis=0,
        )
        token_delta_arrays.append(deltas.astype(np.float64, copy=False))

    return ResponseUpdateVectors(
        response_tokens=list(response_tokens or []),
        deltas_per_token=token_delta_arrays,
        num_layers=num_layers,
    )


def project_update_vectors_with_basis(
    update_vectors: ResponseUpdateVectors,
    basis: np.ndarray,
    *,
    explained_variance_ratio: list[float] | np.ndarray | None = None,
) -> GeometryProjection3D:
    num_components = int(basis.shape[1])
    token_paths: list[TokenGeometry3D] = []
    provided_tokens = update_vectors.response_tokens

    for local_index, deltas in enumerate(update_vectors.deltas_per_token):
        projected_deltas = deltas @ basis
        origin = np.zeros((1, num_components), dtype=np.float64)
        points = np.concatenate([origin, np.cumsum(projected_deltas, axis=0)], axis=0)
        segment_cosines = [
            _safe_cosine(deltas[layer_index], deltas[layer_index + 1])
            for layer_index in range(max(0, deltas.shape[0] - 1))
        ]
        token_paths.append(
            TokenGeometry3D(
                token_index=local_index,
                token=provided_tokens[local_index] if local_index < len(provided_tokens) else f"token_{local_index}",
                points=points.tolist(),
                deltas=projected_deltas.tolist(),
                segment_cosines=segment_cosines,
            )
        )

    variance = np.asarray(explained_variance_ratio if explained_variance_ratio is not None else np.zeros(num_components), dtype=np.float64)
    if variance.shape[0] < num_components:
        padded = np.zeros(num_components, dtype=np.float64)
        padded[: variance.shape[0]] = variance
        variance = padded

    return GeometryProjection3D(
        method="pca",
        num_components=num_components,
        explained_variance_ratio=variance.tolist(),
        num_layers=update_vectors.num_layers,
        token_paths=token_paths,
    )


def fit_joint_pca_basis(
    update_vector_groups: list[ResponseUpdateVectors],
    *,
    num_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    for group in update_vector_groups:
        if group.deltas_per_token:
            matrices.append(np.concatenate(group.deltas_per_token, axis=0))
    if not matrices:
        raise ValueError("No update vectors available to fit a joint PCA basis.")
    stacked = np.concatenate(matrices, axis=0)
    return _fit_pca_basis(stacked, num_components=num_components)


def project_response_update_geometry(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    response_start: int,
    response_end: int,
    *,
    response_tokens: list[str] | None = None,
    num_components: int = 3,
) -> GeometryProjection3D:
    """Project the real layer-update vectors for one response span into 3D.

    The projection is PCA over the stacked update vectors for all response
    tokens in the analyzed answer. This preserves as much variance as possible
    in a common basis while staying faithful to the actual vectors used by UDC.
    """

    update_vectors = extract_response_update_vectors(
        hidden_states,
        response_start,
        response_end,
        response_tokens=response_tokens,
    )
    basis, explained_variance = fit_joint_pca_basis([update_vectors], num_components=num_components)
    return project_update_vectors_with_basis(
        update_vectors,
        basis,
        explained_variance_ratio=explained_variance,
    )


__all__ = [
    "GeometryProjection3D",
    "ResponseUpdateVectors",
    "TokenGeometry3D",
    "extract_response_update_vectors",
    "fit_joint_pca_basis",
    "project_update_vectors_with_basis",
    "project_response_update_geometry",
]
