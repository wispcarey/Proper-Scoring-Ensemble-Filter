import torch

from psef.losses.scoring_rules import compute_ensemble_rank_histogram, compute_es, compute_loss


def test_energy_score_known_tiny_case():
    ens = torch.tensor([[[[0.0, 0.0], [2.0, 0.0]]]])
    truth = torch.tensor([[[1.0, 0.0]]])

    score = compute_es(ens, truth)

    assert score.shape == (1, 1)
    assert torch.allclose(score, torch.tensor([[0.5]]))


def test_required_loss_names_are_available():
    ens = torch.tensor(
        [
            [[[0.0, 0.0], [2.0, 0.0]]],
            [[[1.0, 1.0], [1.0, -1.0]]],
        ]
    )
    truth = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])

    for loss_name in ["es", "nl2", "l2"]:
        value = compute_loss(ens, truth, loss_name)
        assert value.ndim == 0
        assert torch.isfinite(value)

    assert compute_loss(ens, truth, "l2").item() == 0.0
    assert compute_loss(ens, truth, "nl2").item() == 0.0


def test_rank_hist_freq_var_is_normalized_per_projection():
    ens = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 1.0]],
                [[0.0, 1.0], [2.0, 1.0]],
                [[0.0, 1.0], [2.0, 1.0]],
                [[0.0, 1.0], [2.0, 1.0]],
            ]
        ]
    )
    truth = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0], [1.5, 0.0]]])
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    stats = compute_ensemble_rank_histogram(
        ens_states=ens,
        true_states=truth,
        projection_directions=directions,
        tie_break="midpoint",
    )

    assert stats["projection_sample_counts"].tolist() == [4, 4]
    assert stats["counts_by_projection"].tolist() == [[1, 2, 1], [4, 0, 0]]
    assert torch.allclose(stats["freq_var_by_projection"], torch.tensor([0.25, 4.0]))
    assert abs(stats["freq_var"] - 2.125) < 1e-6
