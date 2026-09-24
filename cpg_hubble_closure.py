#!/usr/bin/env python3
"""Hubble-closure diagnostic for Cosmological Proper-Time Geodesy (CPG).

This module reframes a discrepancy between an early/reference H0 inference and
a local H0 measurement as the *chronometric mapping required* by the simple
phenomenological relation

    H_local,pred = H_reference / Gamma0.

It does not derive Gamma0 from Einstein's equations and should not be read as
evidence for RTD-EU.  Its role is diagnostic: quantify the clock mapping a
physical model would have to produce, and test how much of an H0 gap a proposed
Gamma0 would close.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class H0Measurement:
    label: str
    value: float
    sigma: float

    def validate(self) -> None:
        if self.value <= 0:
            raise ValueError(f"{self.label}: H0 must be positive")
        if self.sigma <= 0:
            raise ValueError(f"{self.label}: sigma must be positive")


@dataclass(frozen=True)
class RequiredClosure:
    gamma0_required: float
    gamma0_sigma: float
    clock_departure_percent: float
    h0_gap: float
    tension_sigma: float


@dataclass(frozen=True)
class GammaTest:
    gamma0: float
    gamma0_sigma: float
    h0_predicted_local: float
    h0_predicted_sigma: float
    residual_h0: float
    residual_tension_sigma: float
    gap_closed_fraction: float


class HubbleClosureAuditor:
    """Audit the mapping H_local = H_reference / Gamma0.

    ``rho`` is an optional correlation coefficient between the two H0
    measurements.  The default ``rho=0`` is appropriate when they are treated
    as statistically independent.  A model uncertainty on Gamma0 can be passed
    to :meth:`evaluate_gamma`.
    """

    def __init__(self, reference: H0Measurement, local: H0Measurement,
                 rho: float = 0.0):
        reference.validate()
        local.validate()
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must satisfy -1 < rho < 1")
        self.reference = reference
        self.local = local
        self.rho = float(rho)

    @property
    def covariance(self) -> float:
        return self.rho * self.reference.sigma * self.local.sigma

    def original_tension_sigma(self) -> float:
        var = (self.reference.sigma**2 + self.local.sigma**2
               - 2.0 * self.covariance)
        if var <= 0:
            raise ValueError("non-positive variance for H0 difference")
        return abs(self.local.value - self.reference.value) / sqrt(var)

    def required_closure(self) -> RequiredClosure:
        """Return the Gamma0 that exactly maps the central H0 values.

        The uncertainty uses first-order Gaussian error propagation for the
        ratio reference/local and includes the optional measurement covariance.
        """
        hr, sr = self.reference.value, self.reference.sigma
        hl, sl = self.local.value, self.local.sigma
        gamma = hr / hl
        cov = self.covariance
        var_gamma = (
            (sr / hl) ** 2
            + (hr * sl / hl**2) ** 2
            - 2.0 * hr * cov / hl**3
        )
        var_gamma = max(var_gamma, 0.0)
        return RequiredClosure(
            gamma0_required=gamma,
            gamma0_sigma=sqrt(var_gamma),
            clock_departure_percent=100.0 * (1.0 - gamma),
            h0_gap=hl - hr,
            tension_sigma=self.original_tension_sigma(),
        )

    def evaluate_gamma(self, gamma0: float, gamma0_sigma: float = 0.0) -> GammaTest:
        """Test a proposed present-day chronometric factor Gamma0.

        Gamma0 is treated as a model prediction independent of the H0 data.
        The predicted-local uncertainty therefore combines reference-H0 and
        model-Gamma uncertainty.  The residual tension also includes the
        supplied reference/local covariance.
        """
        if gamma0 <= 0:
            raise ValueError("gamma0 must be positive")
        if gamma0_sigma < 0:
            raise ValueError("gamma0_sigma must be non-negative")

        hr, sr = self.reference.value, self.reference.sigma
        hl, sl = self.local.value, self.local.sigma
        pred = hr / gamma0
        var_pred = ((sr / gamma0) ** 2
                    + (hr * gamma0_sigma / gamma0**2) ** 2)
        sigma_pred = sqrt(var_pred)

        residual = hl - pred
        # Cov(pred, local) from any covariance between reference and local.
        cov_pred_local = self.covariance / gamma0
        var_residual = var_pred + sl**2 - 2.0 * cov_pred_local
        if var_residual <= 0:
            raise ValueError("non-positive variance for residual H0 difference")
        residual_sigma = abs(residual) / sqrt(var_residual)

        original_gap = hl - hr
        if abs(original_gap) > 0:
            gap_closed = 1.0 - abs(residual) / abs(original_gap)
        else:
            gap_closed = 1.0 if abs(residual) == 0 else float("nan")

        return GammaTest(
            gamma0=float(gamma0),
            gamma0_sigma=float(gamma0_sigma),
            h0_predicted_local=pred,
            h0_predicted_sigma=sigma_pred,
            residual_h0=residual,
            residual_tension_sigma=residual_sigma,
            gap_closed_fraction=gap_closed,
        )

    def scan(self, gamma_values: Iterable[float]) -> list[GammaTest]:
        return [self.evaluate_gamma(float(g)) for g in gamma_values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPG Hubble-closure diagnostic: H_local = H_reference / Gamma0"
    )
    parser.add_argument("--reference", type=float, required=True,
                        help="reference/early H0 [km/s/Mpc]")
    parser.add_argument("--reference-sigma", type=float, required=True)
    parser.add_argument("--local", type=float, required=True,
                        help="local H0 [km/s/Mpc]")
    parser.add_argument("--local-sigma", type=float, required=True)
    parser.add_argument("--rho", type=float, default=0.0,
                        help="correlation coefficient between H0 measurements")
    parser.add_argument("--gamma", type=float, default=None,
                        help="optional model Gamma0 to test")
    parser.add_argument("--gamma-sigma", type=float, default=0.0)
    args = parser.parse_args()

    auditor = HubbleClosureAuditor(
        H0Measurement("reference", args.reference, args.reference_sigma),
        H0Measurement("local", args.local, args.local_sigma),
        rho=args.rho,
    )
    out = {"required_closure": asdict(auditor.required_closure())}
    if args.gamma is not None:
        out["gamma_test"] = asdict(
            auditor.evaluate_gamma(args.gamma, args.gamma_sigma)
        )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
