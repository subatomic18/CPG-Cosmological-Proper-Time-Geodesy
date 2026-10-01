import math

import numpy as np

from cpg_lightcone import FiniteDifferenceSpacetime, endpoint_redshift
from cpg_lightcone_shooting import (
    TrackedSpatialWorldline,
    minimum_image_displacement,
    shoot_to_observer_worldline,
)


def minkowski_spacetime():
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    return eta, FiniteDifferenceSpacetime(
        lambda _x: eta,
        spatial_step=1e-4,
        time_step=1e-4,
        t_bounds=(-1.0, 5.0),
    )


def test_static_observer_shooting_recovers_light_travel_time_and_direction():
    _eta, spacetime = minkowski_spacetime()
    emit = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.array([1.2, 0.4, -0.3])
    expected_time = float(np.linalg.norm(target))
    expected_direction = target / expected_time

    result = shoot_to_observer_worldline(
        spacetime,
        emit,
        lambda _t: target,
        (0.6 * expected_time, 1.5 * expected_time),
        initial_arrival_time=1.1 * expected_time,
        position_tolerance=1e-8,
        ray_max_step=0.1,
        max_nfev=60,
    )

    assert result.converged
    assert result.spatial_miss_distance < 1e-8
    assert math.isclose(result.arrival_time, expected_time, rel_tol=0, abs_tol=2e-7)
    assert np.allclose(result.initial_spatial_k, expected_direction, atol=2e-7)
    assert result.ray.max_abs_null_residual < 1e-10


def test_receding_observer_shooting_recovers_intersection_and_doppler_redshift():
    eta, spacetime = minkowski_spacetime()
    emit = np.array([0.0, 0.0, 0.0, 0.0])
    v = 0.2

    def observer(t):
        return np.array([1.0 + v * t, 0.0, 0.0])

    # Photon x=t and observer x=1+v t, so t_arr=1/(1-v).
    expected_time = 1.0 / (1.0 - v)
    result = shoot_to_observer_worldline(
        spacetime,
        emit,
        observer,
        (0.8, 1.8),
        initial_arrival_time=1.3,
        position_tolerance=1e-8,
        ray_max_step=0.1,
        max_nfev=60,
    )

    assert result.converged
    assert math.isclose(result.arrival_time, expected_time, rel_tol=0, abs_tol=2e-7)
    assert np.allclose(result.initial_spatial_k, [1.0, 0.0, 0.0], atol=2e-7)
    assert result.spatial_miss_distance < 1e-8

    u_emit = np.array([1.0, 0.0, 0.0, 0.0])
    lorentz = 1.0 / math.sqrt(1.0 - v * v)
    u_obs = np.array([lorentz, lorentz * v, 0.0, 0.0])
    z = endpoint_redshift(
        eta,
        u_emit,
        result.ray.k_emit,
        eta,
        u_obs,
        result.ray.k_obs,
    )
    expected_z = math.sqrt((1.0 + v) / (1.0 - v)) - 1.0
    assert math.isclose(z, expected_z, rel_tol=1e-10, abs_tol=1e-10)


def test_periodic_worldline_unwraps_across_box_boundary():
    times = np.array([0.0, 1.0, 2.0])
    wrapped = np.array([
        [9.8, 1.0, 2.0],
        [0.1, 1.0, 2.0],
        [0.4, 1.0, 2.0],
    ])
    worldline = TrackedSpatialWorldline.from_periodic_samples(times, wrapped, 10.0)
    assert np.allclose(worldline.positions[:, 0], [9.8, 10.1, 10.4], atol=1e-12)
    assert np.allclose(worldline.position(0.5), [9.95, 1.0, 2.0], atol=1e-12)
    assert np.allclose(worldline.position(1.5), [10.25, 1.0, 2.0], atol=1e-12)


def test_minimum_image_displacement():
    d = minimum_image_displacement(
        np.array([0.1, 9.8, 5.0]),
        np.array([9.9, 0.2, 5.0]),
        10.0,
    )
    assert np.allclose(d, [0.2, -0.4, 0.0], atol=1e-12)
