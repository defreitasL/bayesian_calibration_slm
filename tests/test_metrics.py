import numpy as np

from slmcal.metrics import kge, spbias, spearman_r, objective_vector


def test_metrics_sane_values():
    obs = np.array([0.0, 1.0, 2.0, 3.0])
    sim = np.array([0.0, 1.0, 2.0, 3.0])

    assert np.isfinite(kge(obs, sim))
    assert abs(spbias(obs, sim)) < 1e-9
    assert abs(spearman_r(obs, sim) - 1.0) < 1e-9


def test_objective_vector_shape():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.0, 3.0])
    v = objective_vector(obs, sim, ("kge", "spbias"))
    assert v.shape == (2,)
