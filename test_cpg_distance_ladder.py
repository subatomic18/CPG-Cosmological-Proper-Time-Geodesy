import numpy as np

from cpg_distance_ladder import fit_gls, h0_from_5logh0


def test_h0_transform():
    h0, sigma = h0_from_5logh0(5.0 * np.log10(73.04), 0.03)
    assert np.isclose(h0, 73.04, rtol=0, atol=1e-10)
    expected_sigma = np.log(10.0) / 5.0 * 73.04 * 0.03
    assert np.isclose(sigma, expected_sigma)


def test_gls_recovers_synthetic_solution():
    # A small nontrivial correlated linear problem with an H0 parameter.
    names = ("offset", "5logH0")
    q_true = np.array([2.5, 5.0 * np.log10(73.04)])
    L = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.3],
            [1.0, 0.8],
            [1.0, 1.2],
        ]
    )
    C = np.array(
        [
            [0.04, 0.006, 0.0, 0.0],
            [0.006, 0.05, 0.004, 0.0],
            [0.0, 0.004, 0.06, 0.005],
            [0.0, 0.0, 0.005, 0.07],
        ]
    )
    y = L @ q_true

    fit = fit_gls(y, L, C, names)
    assert np.allclose(fit.parameter_values, q_true, rtol=0, atol=1e-10)
    assert np.isclose(fit.h0, 73.04, rtol=0, atol=1e-9)
    assert np.isclose(fit.chi2, 0.0, atol=1e-20)
    assert fit.dof == 2
