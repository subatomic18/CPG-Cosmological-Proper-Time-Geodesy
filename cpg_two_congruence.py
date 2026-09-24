#!/usr/bin/env python3
"""Two-congruence weak-field chronometric diagnostic for CPG.

This module connects a physically specified pair of weak-field timelike
histories to the Hubble-closure diagnostic.  It keeps two distinct quantities
separate:

1. the instantaneous relative clock rate at the present epoch,
       Gamma_AB(0) = (d tau_A/dt) / (d tau_B/dt),
   which is the quantity used by the simple Hubble mapping
       H_A,pred = H_B / Gamma_AB(0);

2. the accumulated proper-time difference between the two histories,
       Delta tau_AB = tau_A - tau_B.

The weak-field clock rate is
    d tau / dt ~= 1 + Phi/c^2 - v^2/(2 c^2).

This is a conservative GR benchmark and simulation-ready diagnostic.  It is
not an Einstein-equation solver, and the Hubble mapping is phenomenological:
a physical cosmological interpretation still requires a self-consistent
metric, congruences, invariant boundary conditions, and light-cone mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import numpy as np
from scipy import integrate

from cpg_v0_3 import C_KM_S, HUBBLE_CONVERSION_GYR, FiducialCosmology
from cpg_hubble_closure import HubbleClosureAuditor, GammaTest


@dataclass(frozen=True)
class WeakFieldState:
    """Instantaneous state of one timelike history in a common weak-field gauge.

    ``phi_over_c2`` is Phi/c^2 and ``v_km_s`` is the peculiar speed relative
    to the chosen reference slicing.
    """

    label: str
    phi_over_c2: float
    v_km_s: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.phi_over_c2):
            raise ValueError(f"{self.label}: phi_over_c2 must be finite")
        if not np.isfinite(self.v_km_s) or self.v_km_s < 0:
            raise ValueError(f"{self.label}: v_km_s must be finite and >= 0")
        if self.v_km_s >= C_KM_S:
            raise ValueError(f"{self.label}: v_km_s must be < c")


@dataclass(frozen=True)
class TwoCongruenceAccumulation:
    z_low: float
    z_high: float
    tau_a_gyr: float
    tau_b_gyr: float
    delta_tau_a_minus_b_gyr: float
    delta_tau_a_minus_b_years: float
    accumulated_ratio_a_over_b: float


@dataclass(frozen=True)
class TwoCongruenceHubbleResult:
    gamma0_a_over_b: float
    clock_departure_percent: float
    hubble_test: GammaTest


def weak_field_clock_rate(state: WeakFieldState) -> float:
    """Return d tau/dt to first post-Newtonian order."""
    state.validate()
    beta2 = (state.v_km_s / C_KM_S) ** 2
    rate = 1.0 + state.phi_over_c2 - 0.5 * beta2
    if rate <= 0:
        raise ValueError(
            f"{state.label}: weak-field expression produced non-positive d tau/dt"
        )
    return float(rate)


def relative_gamma(state_a: WeakFieldState, state_b: WeakFieldState) -> float:
    """Instantaneous relative clock factor (d tau_A/dt)/(d tau_B/dt)."""
    return weak_field_clock_rate(state_a) / weak_field_clock_rate(state_b)


def linear_scale_factor_envelope(z: float, z_start: float) -> float:
    """Toy structure-growth envelope that is zero at z_start and one at z=0.

    It is intentionally phenomenological.  It is useful for conservative
    accumulation benchmarks before replacing the histories with simulation
    output.
    """
    if z_start <= 0:
        raise ValueError("z_start must be > 0")
    if z >= z_start:
        return 0.0
    if z <= 0:
        return 1.0
    a = 1.0 / (1.0 + z)
    a_start = 1.0 / (1.0 + z_start)
    return float((a - a_start) / (1.0 - a_start))


def scaled_state(
    present_state: WeakFieldState, z: float, z_start: float
) -> WeakFieldState:
    """Scale a present-day weak-field state with the toy growth envelope.

    Phi scales linearly with the envelope and v scales as sqrt(envelope), so
    both the potential and kinetic contributions to d tau/dt scale linearly.
    """
    g = linear_scale_factor_envelope(z, z_start)
    return WeakFieldState(
        label=present_state.label,
        phi_over_c2=present_state.phi_over_c2 * g,
        v_km_s=present_state.v_km_s * sqrt(g),
    )


class TwoCongruenceAuditor:
    """Integrate two weak-field histories and connect them to Hubble closure."""

    def __init__(self, cosmo: FiducialCosmology | None = None):
        self.cosmo = cosmo or FiducialCosmology()

    def _dt_dz_gyr(self, z: float) -> float:
        return HUBBLE_CONVERSION_GYR / (
            (1.0 + z) * self.cosmo.H(z)
        )

    def integrate_functions(
        self,
        z_low: float,
        z_high: float,
        state_a_fn: Callable[[float], WeakFieldState],
        state_b_fn: Callable[[float], WeakFieldState],
        epsabs: float = 1e-12,
        epsrel: float = 1e-12,
    ) -> TwoCongruenceAccumulation:
        """Integrate proper time for both histories between common z surfaces."""
        if z_low < 0 or z_high <= z_low:
            raise ValueError("Require 0 <= z_low < z_high")

        def f_a(z: float) -> float:
            return weak_field_clock_rate(state_a_fn(z)) * self._dt_dz_gyr(z)

        def f_b(z: float) -> float:
            return weak_field_clock_rate(state_b_fn(z)) * self._dt_dz_gyr(z)

        tau_a = integrate.quad(
            f_a, z_low, z_high, epsabs=epsabs, epsrel=epsrel, limit=500
        )[0]
        tau_b = integrate.quad(
            f_b, z_low, z_high, epsabs=epsabs, epsrel=epsrel, limit=500
        )[0]
        delta = tau_a - tau_b
        return TwoCongruenceAccumulation(
            z_low=float(z_low),
            z_high=float(z_high),
            tau_a_gyr=float(tau_a),
            tau_b_gyr=float(tau_b),
            delta_tau_a_minus_b_gyr=float(delta),
            delta_tau_a_minus_b_years=float(delta * 1e9),
            accumulated_ratio_a_over_b=float(tau_a / tau_b),
        )

    def integrate_scaled_present_states(
        self,
        state_a_today: WeakFieldState,
        state_b_today: WeakFieldState,
        z_start: float = 2.0,
    ) -> TwoCongruenceAccumulation:
        """Integrate a pair of present states grown from zero at z_start."""
        return self.integrate_functions(
            0.0,
            z_start,
            lambda z: scaled_state(state_a_today, z, z_start),
            lambda z: scaled_state(state_b_today, z, z_start),
        )

    def couple_present_to_hubble(
        self,
        local_today: WeakFieldState,
        reference_today: WeakFieldState,
        hubble_auditor: HubbleClosureAuditor,
    ) -> TwoCongruenceHubbleResult:
        """Feed the physically computed present relative rate to Hubble closure."""
        gamma0 = relative_gamma(local_today, reference_today)
        htest = hubble_auditor.evaluate_gamma(gamma0)
        return TwoCongruenceHubbleResult(
            gamma0_a_over_b=float(gamma0),
            clock_departure_percent=float(100.0 * (1.0 - gamma0)),
            hubble_test=htest,
        )
