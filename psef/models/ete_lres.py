"""EtE-LRes helpers.

The full model construction remains in ``psef.training.legacy.set_models`` so
the public helper uses the same architecture setup as the training entry points.
"""

from psef.training.legacy import set_models


def apply_ete_lres_update(forecast_ensemble, correction):
    """Apply the EtE-LRes residual analysis update."""
    return forecast_ensemble + correction


__all__ = ["apply_ete_lres_update", "set_models"]
