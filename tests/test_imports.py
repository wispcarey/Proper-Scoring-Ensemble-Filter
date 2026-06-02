import importlib


def test_public_package_imports():
    modules = [
        "psef",
        "psef.losses.scoring_rules",
        "psef.losses.diagnostics",
        "psef.models.architectures",
        "psef.models.set_transformer",
        "psef.models.ete_lres",
        "psef.models.corr_terms",
        "psef.filters.analysis",
        "psef.filters.enkf",
        "psef.filters.esrf",
        "psef.filters.letkf",
        "psef.filters.ienkf",
        "psef.filters.bpf",
        "psef.dynamics.lorenz63",
        "psef.dynamics.lorenz96",
        "psef.dynamics.doubling",
        "psef.dynamics.observation_maps",
        "psef.data.datasets",
        "psef.data.linear",
        "psef.training.train_loop",
        "psef.training.eval_loop",
        "psef.training.checkpoints",
        "psef.config.defaults",
    ]

    for module in modules:
        importlib.import_module(module)
