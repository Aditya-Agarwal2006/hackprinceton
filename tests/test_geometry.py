import math

import torch

from app.geometry import project_response_update_geometry


def _stack_hidden_states(layer_vectors):
    tensors = []
    for layer in layer_vectors:
        tensors.append(torch.tensor([layer], dtype=torch.float32))
    return tuple(tensors)


def test_project_response_update_geometry_returns_paths_for_each_token():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ]
    )

    geometry = project_response_update_geometry(
        hidden_states,
        0,
        2,
        response_tokens=["Paris", "France"],
    )

    assert geometry.method == "pca"
    assert geometry.num_components == 3
    assert geometry.num_layers == 3
    assert len(geometry.token_paths) == 2
    assert geometry.token_paths[0].token == "Paris"
    assert len(geometry.token_paths[0].points) == geometry.num_layers + 1
    assert len(geometry.token_paths[0].segment_cosines) == geometry.num_layers - 1


def test_project_response_update_geometry_preserves_same_direction_pattern():
    hidden_states = _stack_hidden_states(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
            [[3.0, 0.0]],
        ]
    )

    geometry = project_response_update_geometry(
        hidden_states,
        0,
        1,
        response_tokens=["Paris"],
    )

    cosines = geometry.token_paths[0].segment_cosines
    assert len(cosines) == 2
    assert all(math.isclose(value, 1.0, rel_tol=1e-6, abs_tol=1e-6) for value in cosines)
