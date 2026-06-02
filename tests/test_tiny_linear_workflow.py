import torch

from psef.filters.analysis import ensemble_kalman_filter_analysis
from psef.losses.scoring_rules import compute_loss


def test_tiny_cpu_analysis_then_loss_workflow():
    ensemble_f = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float32,
    )
    observation_y = torch.tensor([[1.0]], dtype=torch.float32)

    def observation_operator_ens(x):
        return x[..., :1]

    analysis, _ = ensemble_kalman_filter_analysis(
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=observation_operator_ens,
        sigma_y=0.2,
        sigma_v=0.0,
        method="EnKF-PertObs",
    )

    truth = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    loss = compute_loss(analysis.unsqueeze(0), truth, "es")

    assert loss.ndim == 0
    assert torch.isfinite(loss)
