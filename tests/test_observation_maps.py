import torch

from psef.dynamics.observation_maps import partial_obs_operator


def test_partial_square_observation_operator():
    obs, info = partial_obs_operator(
        ori_dim=3,
        obs_inds=[0, 2],
        device=torch.device("cpu"),
        obs_fn="square",
    )
    x = torch.tensor([[2.0, 10.0, -3.0]])

    y = obs(x)

    assert y.shape == (1, 2)
    assert torch.allclose(y, torch.tensor([[4.0, 9.0]]))
    assert info["obs_fn"] == "square"


def test_partial_identity_observation_operator_returns_matrix_info():
    obs, matrix = partial_obs_operator(
        ori_dim=3,
        obs_inds=[1],
        device=torch.device("cpu"),
        obs_fn="identity",
    )
    x = torch.tensor([[2.0, 10.0, -3.0]])

    y = obs(x)

    assert y.shape == (1, 1)
    assert torch.allclose(y, torch.tensor([[10.0]]))
    assert matrix.shape == (3, 1)
