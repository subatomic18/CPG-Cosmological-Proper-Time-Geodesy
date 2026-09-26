import math

import numpy as np

from cpg_lightcone import (
    FiniteDifferenceSpacetime,
    adm_metric,
    endpoint_redshift,
    eulerian_four_velocity,
    solve_future_null_k0,
    trace_null_geodesic,
)


def test_adm_metric_and_four_velocity_normalization():
    gamma = np.diag([1.2, 0.9, 1.1])
    beta = np.array([0.03, -0.02, 0.01])
    alpha = 0.97
    g = adm_metric(alpha, beta, gamma)
    assert np.allclose(g, g.T)
    assert np.linalg.eigvalsh(gamma).min() > 0

    v = np.array([0.10, 0.04, -0.02])
    u = eulerian_four_velocity(alpha, beta, gamma, v)
    assert math.isclose(float(u @ g @ u), -1.0, rel_tol=0, abs_tol=1e-10)


def test_null_root_is_future_directed_and_null():
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    ki = np.array([0.3, 0.4, 0.0])
    k0 = solve_future_null_k0(eta, ki)
    k = np.array([k0, *ki])
    assert k0 > 0
    assert math.isclose(float(k @ eta @ k), 0.0, rel_tol=0, abs_tol=1e-14)


def test_minkowski_ray_and_zero_redshift():
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    spacetime = FiniteDifferenceSpacetime(
        lambda _x: eta,
        spatial_step=1e-4,
        time_step=1e-4,
        t_bounds=(-1.0, 5.0),
    )
    ray = trace_null_geodesic(
        spacetime,
        emit_event=np.array([0.0, 0.0, 0.0, 0.0]),
        spatial_k=np.array([1.0, 0.0, 0.0]),
        target_time=2.0,
        max_step=0.2,
    )
    assert ray.reached_target_time
    assert np.allclose(ray.obs_event, [2.0, 2.0, 0.0, 0.0], atol=1e-8)
    assert ray.max_abs_null_residual < 1e-10

    u = np.array([1.0, 0.0, 0.0, 0.0])
    z = endpoint_redshift(eta, u, ray.k_emit, eta, u, ray.k_obs)
    assert abs(z) < 1e-10


def test_minkowski_endpoint_doppler_redshift():
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    k = np.array([1.0, 1.0, 0.0, 0.0])
    u_emit = np.array([1.0, 0.0, 0.0, 0.0])
    v = 0.2
    lorentz = 1.0 / math.sqrt(1.0 - v * v)
    u_obs = np.array([lorentz, lorentz * v, 0.0, 0.0])
    z = endpoint_redshift(eta, u_emit, k, eta, u_obs, k)
    expected = math.sqrt((1.0 + v) / (1.0 - v)) - 1.0
    assert math.isclose(z, expected, rel_tol=1e-12)


def test_flat_flrw_recovers_scale_factor_redshift():
    # Analytic flat FLRW benchmark: ds^2=-dt^2+a(t)^2 dx^2 with a=e^(H t).
    H = 0.05

    def metric_fn(x):
        a = math.exp(H * float(x[0]))
        return np.diag([-1.0, a * a, a * a, a * a])

    spacetime = FiniteDifferenceSpacetime(
        metric_fn,
        spatial_step=1e-4,
        time_step=1e-5,
        t_bounds=(-1.0, 20.0),
    )
    ray = trace_null_geodesic(
        spacetime,
        emit_event=np.array([0.0, 0.0, 0.0, 0.0]),
        spatial_k=np.array([1.0, 0.0, 0.0]),
        target_time=10.0,
        max_step=0.1,
        rtol=1e-9,
        atol=1e-11,
    )
    u = np.array([1.0, 0.0, 0.0, 0.0])
    z = endpoint_redshift(
        spacetime.metric(ray.emit_event), u, ray.k_emit,
        spacetime.metric(ray.obs_event), u, ray.k_obs,
    )
    expected = math.exp(H * 10.0) - 1.0
    assert math.isclose(z, expected, rel_tol=2e-5, abs_tol=2e-6)
    assert ray.max_abs_null_residual < 1e-7
