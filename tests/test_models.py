import torch

from psef.models.ete_lres import apply_ete_lres_update
from psef.models.set_transformer import SetTransformer


def test_set_transformer_forward_shape():
    torch.manual_seed(0)
    model = SetTransformer(
        input_dim=2,
        num_heads=1,
        num_inds=1,
        output_dim=3,
        hidden_dim=4,
        num_layers=1,
    )
    x = torch.randn(5, 7, 2)

    out = model(x)

    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_ete_lres_residual_update():
    forecast = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    correction = torch.tensor([[[0.5, -0.5], [1.0, -1.0]]])

    analyzed = apply_ete_lres_update(forecast, correction)

    assert torch.allclose(analyzed, forecast + correction)
