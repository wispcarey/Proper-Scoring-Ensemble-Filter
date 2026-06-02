import torch

from psef.filters.analysis import ensemble_kalman_filter_analysis


def test_esrf_analysis_shape_and_finiteness():
    ensemble_f = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float32,
    )
    observation_y = torch.tensor([[1.5]], dtype=torch.float32)

    def observation_operator_ens(x):
        return x[..., :1]

    analysis, _ = ensemble_kalman_filter_analysis(
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=observation_operator_ens,
        sigma_y=0.1,
        sigma_v=0.0,
        method="ESRF",
    )

    assert analysis.shape == ensemble_f.shape
    assert torch.isfinite(analysis).all()
