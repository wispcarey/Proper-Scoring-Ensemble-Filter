import copy


LOW_DIM_SEARCH_SPECS_BY_N = {
    5: {"infl_num": 8, "infl_step": 0.05, "loc_num": 1, "loc_step": None, "loc_min": None},
    10: {"infl_num": 8, "infl_step": 0.04, "loc_num": 1, "loc_step": None, "loc_min": None},
    15: {"infl_num": 8, "infl_step": 0.03, "loc_num": 1, "loc_step": None, "loc_min": None},
    20: {"infl_num": 8, "infl_step": 0.025, "loc_num": 1, "loc_step": None, "loc_min": None},
    40: {"infl_num": 8, "infl_step": 0.02, "loc_num": 1, "loc_step": None, "loc_min": None},
    60: {"infl_num": 8, "infl_step": 0.015, "loc_num": 1, "loc_step": None, "loc_min": None},
    100: {"infl_num": 8, "infl_step": 0.01, "loc_num": 1, "loc_step": None, "loc_min": None},
}

L96_LOCAL_SEARCH_SPECS_BY_N = {
    5: {"infl_num": 8, "infl_step": 0.05, "loc_num": 6, "loc_step": 2.0, "loc_min": 0.001},
    10: {"infl_num": 8, "infl_step": 0.04, "loc_num": 6, "loc_step": 2.0, "loc_min": 0.001},
    15: {"infl_num": 8, "infl_step": 0.03, "loc_num": 6, "loc_step": 1.5, "loc_min": 0.001},
    20: {"infl_num": 8, "infl_step": 0.025, "loc_num": 6, "loc_step": 1.0, "loc_min": 0.001},
    40: {"infl_num": 8, "infl_step": 0.02, "loc_num": 6, "loc_step": 0.8, "loc_min": 0.001},
    60: {"infl_num": 8, "infl_step": 0.015, "loc_num": 6, "loc_step": 0.6, "loc_min": 0.001},
    100: {"infl_num": 8, "infl_step": 0.01, "loc_num": 6, "loc_step": 0.5, "loc_min": 0.001},
}

KS_LOCAL_SEARCH_SPECS_BY_N = {
    5: {"infl_num": 8, "infl_step": 0.05, "loc_num": 6, "loc_step": 4.0, "loc_min": 0.001},
    10: {"infl_num": 8, "infl_step": 0.04, "loc_num": 6, "loc_step": 4.0, "loc_min": 0.001},
    15: {"infl_num": 8, "infl_step": 0.03, "loc_num": 6, "loc_step": 3.0, "loc_min": 0.001},
    20: {"infl_num": 8, "infl_step": 0.025, "loc_num": 6, "loc_step": 2.0, "loc_min": 0.001},
    40: {"infl_num": 8, "infl_step": 0.02, "loc_num": 6, "loc_step": 1.5, "loc_min": 0.001},
    60: {"infl_num": 8, "infl_step": 0.015, "loc_num": 6, "loc_step": 1.2, "loc_min": 0.001},
    100: {"infl_num": 8, "infl_step": 0.01, "loc_num": 6, "loc_step": 1.0, "loc_min": 0.001},
}


LOW_DIM_METHOD_GRIDSEARCH_INFO = {
    "EnKF": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "ESRF": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-PertObs": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Sqrt": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Order1": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
}

L96_METHOD_GRIDSEARCH_INFO = {
    "EnKF": {
        "use_localization": True,
        "localization_fn": "GC",
        "search_specs_by_N": copy.deepcopy(L96_LOCAL_SEARCH_SPECS_BY_N),
    },
    "ESRF": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-PertObs": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Sqrt": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Order1": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "LETKF": {
        "use_localization": True,
        "localization_fn": "GC",
        "search_specs_by_N": copy.deepcopy(L96_LOCAL_SEARCH_SPECS_BY_N),
    },
}

KS_METHOD_GRIDSEARCH_INFO = {
    "EnKF": {
        "use_localization": True,
        "localization_fn": "GC",
        "search_specs_by_N": copy.deepcopy(KS_LOCAL_SEARCH_SPECS_BY_N),
    },
    "ESRF": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-PertObs": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Sqrt": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "iEnKS-Order1": {
        "use_localization": False,
        "localization_fn": None,
        "search_specs_by_N": copy.deepcopy(LOW_DIM_SEARCH_SPECS_BY_N),
    },
    "LETKF": {
        "use_localization": True,
        "localization_fn": "GC",
        "search_specs_by_N": copy.deepcopy(KS_LOCAL_SEARCH_SPECS_BY_N),
    },
}


BENCHMARK_GRIDSEARCH_INFO = {
    "lorenz63": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "rossler": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "linear": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "circle": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "hdoublewell": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "doubling1d": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "complex2d": copy.deepcopy(LOW_DIM_METHOD_GRIDSEARCH_INFO),
    "lorenz96": copy.deepcopy(L96_METHOD_GRIDSEARCH_INFO),
    "ks": copy.deepcopy(KS_METHOD_GRIDSEARCH_INFO),
}


def _build_infl_range(infl_num, infl_step):
    infl_num = int(infl_num)
    if infl_num <= 0:
        raise ValueError("infl_num must be positive.")
    infl_step = float(infl_step)
    return [round(1.0 + idx * infl_step, 6) for idx in range(infl_num)]


def _build_loc_range(loc_num, loc_step=None, loc_min=None):
    loc_num = int(loc_num)
    if loc_num <= 0:
        raise ValueError("loc_num must be positive.")
    if loc_num == 1 and loc_step is None:
        return [None]
    if loc_step is None or loc_min is None:
        raise ValueError("loc_step and loc_min must be set when localization search is enabled.")

    loc_vals = [round(float(loc_min), 6)]
    loc_step = float(loc_step)
    for idx in range(1, loc_num):
        loc_vals.append(round(idx * loc_step, 6))
    return loc_vals


def _resolve_search_specs_for_n(search_specs_by_n, ensemble_size):
    if len(search_specs_by_n) == 0:
        raise ValueError("search_specs_by_N cannot be empty.")

    ensemble_size = int(ensemble_size)
    if ensemble_size in search_specs_by_n:
        resolved_n = ensemble_size
    else:
        resolved_n = min(search_specs_by_n.keys(), key=lambda n_val: abs(int(n_val) - ensemble_size))

    resolved_cfg = copy.deepcopy(search_specs_by_n[resolved_n])
    resolved_cfg["resolved_search_N"] = int(resolved_n)
    resolved_cfg["requested_search_N"] = int(ensemble_size)
    resolved_cfg["infl_range"] = _build_infl_range(
        infl_num=resolved_cfg["infl_num"],
        infl_step=resolved_cfg["infl_step"],
    )
    resolved_cfg["loc_radius_range"] = _build_loc_range(
        loc_num=resolved_cfg["loc_num"],
        loc_step=resolved_cfg.get("loc_step"),
        loc_min=resolved_cfg.get("loc_min"),
    )
    return resolved_cfg


def get_supported_benchmark_methods(dataset):
    dataset_key = str(dataset or "").lower()
    dataset_cfg = BENCHMARK_GRIDSEARCH_INFO.get(dataset_key)
    if dataset_cfg is None:
        raise KeyError(
            f"Dataset '{dataset}' is not configured in BENCHMARK_GRIDSEARCH_INFO."
        )
    return tuple(dataset_cfg.keys())


def get_benchmark_gridsearch_config(dataset, method, ensemble_size, force_no_localization=False):
    dataset_key = str(dataset or "").lower()
    dataset_cfg = BENCHMARK_GRIDSEARCH_INFO.get(dataset_key)
    if dataset_cfg is None:
        raise KeyError(
            f"Dataset '{dataset}' is not configured in BENCHMARK_GRIDSEARCH_INFO."
        )

    if method not in dataset_cfg:
        supported_methods = ", ".join(dataset_cfg.keys())
        raise KeyError(
            f"Method '{method}' is not configured for dataset '{dataset}'. "
            f"Supported methods: {supported_methods}"
        )

    method_cfg = copy.deepcopy(dataset_cfg[method])
    search_specs_by_n = method_cfg.pop("search_specs_by_N")
    resolved_cfg = _resolve_search_specs_for_n(search_specs_by_n, ensemble_size=ensemble_size)
    method_cfg.update(resolved_cfg)

    if force_no_localization:
        method_cfg["loc_radius_range"] = [None]
        method_cfg["use_localization"] = False
        method_cfg["localization_fn"] = None
    return method_cfg
