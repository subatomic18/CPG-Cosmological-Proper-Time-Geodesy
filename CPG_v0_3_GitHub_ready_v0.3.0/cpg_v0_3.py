#!/usr/bin/env python3
"""
Cosmological Proper-Time Geodesy (CPG v0.3)
High-Redshift Transient Closure Module

Author: Jeffery Barnes
Affiliation: Independent Researcher
License: MIT (code) / CC BY 4.0 (documentation)

CPG v0.3 extends the v0.2 dual-pathway engine with observational closure
analysis for future high-redshift standardized transient samples.

It tests temporal closure. It does NOT derive a physical RTD-EU lapse
Gamma(z,E) from Einstein's equations.
"""
from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, stats

C_KM_S = 299792.458
HUBBLE_CONVERSION_GYR = 977.79222168


class FiducialCosmology:
    def __init__(self, H0=67.40, Om0=0.315, Ode0=None, Or0=9.2e-5, Ok0=0.0):
        self.H0 = float(H0)
        self.Om0 = float(Om0)
        self.Or0 = float(Or0)
        self.Ok0 = float(Ok0)
        # By default enforce E(0)=1 exactly.  If Ode0 is supplied explicitly,
        # retain it but warn when the density parameters do not close.
        self.Ode0 = float(1.0 - self.Om0 - self.Or0 - self.Ok0) if Ode0 is None else float(Ode0)
        density_sum = self.Om0 + self.Ode0 + self.Or0 + self.Ok0
        if abs(density_sum - 1.0) > 1e-6:
            warnings.warn(
                f"Density parameters sum to {density_sum:.9f}, so E(0) != 1. "
                "For a normalized fiducial background, omit Ode0 or enforce closure explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )

    def E(self, z: float) -> float:
        zp1 = 1.0 + z
        return float(np.sqrt(
            self.Or0*zp1**4 + self.Om0*zp1**3 + self.Ok0*zp1**2 + self.Ode0
        ))

    def H(self, z: float) -> float:
        return self.H0 * self.E(z)

    def H_inv_gyr(self, z: float) -> float:
        return self.H(z) / HUBBLE_CONVERSION_GYR


class CPGEngine:
    """Dual-integrator engine retained from CPG v0.2."""
    def __init__(self, cosmo: Optional[FiducialCosmology] = None):
        self.cosmo = cosmo or FiducialCosmology()

    def pathway_a_z_quad(self, z1, z2, lapse_fn=None, epsabs=1.49e-12, epsrel=1.49e-12):
        if z2 <= z1:
            raise ValueError("Require z2 > z1")
        def integrand(z):
            gamma = lapse_fn(z) if lapse_fn is not None else 1.0
            return gamma / ((1.0 + z) * self.cosmo.H_inv_gyr(z))
        return integrate.quad(integrand, z1, z2, epsabs=epsabs, epsrel=epsrel, limit=500)

    def pathway_b_lna_collocation(self, z1, z2, lapse_fn=None, rtol=1e-11, atol=1e-12):
        if z2 <= z1:
            raise ValueError("Require z2 > z1")
        xi_start, xi_end = -np.log(1.0 + z2), -np.log(1.0 + z1)
        def rhs(xi, y):
            z = np.exp(-xi) - 1.0
            gamma = lapse_fn(z) if lapse_fn is not None else 1.0
            return np.array([gamma / self.cosmo.H_inv_gyr(z)])
        sol = integrate.solve_ivp(rhs, (xi_start, xi_end), [0.0], method="Radau", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return float(sol.y[0, -1])

    def verify_numerical_closure(self, z1=0.0, z2=1089.80, tolerance_gyr=1e-6):
        tau_a, err_a = self.pathway_a_z_quad(z1, z2)
        tau_b = self.pathway_b_lna_collocation(z1, z2)
        delta = abs(tau_a - tau_b)
        return {
            "tau_A_gyr": tau_a, "tau_B_gyr": tau_b,
            "quad_error_gyr": err_a, "delta_num_gyr": delta,
            "delta_num_years": delta*1e9, "tolerance_gyr": tolerance_gyr,
            "passed": bool(delta < tolerance_gyr)
        }


@dataclass(frozen=True)
class TransientEvent:
    event_id: str
    z: float
    t_obs: float
    t_rest: float
    sigma_t_obs: float
    sigma_t_rest: float = 0.0
    environment: str = "unknown"
    quality: int = 1

    def validate(self):
        if self.z < 0:
            raise ValueError(f"{self.event_id}: z must be >= 0")
        if self.t_obs <= 0 or self.t_rest <= 0:
            raise ValueError(f"{self.event_id}: timescales must be > 0")
        if self.sigma_t_obs < 0 or self.sigma_t_rest < 0:
            raise ValueError(f"{self.event_id}: uncertainties must be >= 0")


@dataclass
class GLSFit:
    model: str
    n: int
    params: np.ndarray
    covariance: np.ndarray
    chi2: float
    dof: int
    p_value: float
    z_min: float
    z_max: float

    def stderr(self):
        return np.sqrt(np.diag(self.covariance))


def load_transient_csv(path: str | Path) -> List[TransientEvent]:
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"event_id", "z", "t_obs", "t_rest", "sigma_t_obs"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in reader:
            e = TransientEvent(
                event_id=row["event_id"], z=float(row["z"]),
                t_obs=float(row["t_obs"]), t_rest=float(row["t_rest"]),
                sigma_t_obs=float(row["sigma_t_obs"]),
                sigma_t_rest=float(row.get("sigma_t_rest") or 0.0),
                environment=(row.get("environment") or "unknown").strip(),
                quality=int(row.get("quality") or 1),
            )
            e.validate()
            events.append(e)
    return events


def _prepare(events, systematic_fraction=0.0, quality_min=1):
    kept = [e for e in events if e.quality >= quality_min]
    if not kept:
        raise ValueError("No events remain after quality cuts")
    x = np.array([math.log1p(e.z) for e in kept])
    y = np.array([math.log(e.t_obs/e.t_rest) for e in kept])
    var = np.array([
        (e.sigma_t_obs/e.t_obs)**2 + (e.sigma_t_rest/e.t_rest)**2 + systematic_fraction**2
        for e in kept
    ])
    if np.any(var <= 0):
        raise ValueError("All retained events require non-zero total uncertainty")
    return x, y, var, kept


def _validate_cov(cov, n):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n, n):
        raise ValueError(f"Covariance shape {cov.shape} != ({n},{n})")
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Covariance must be symmetric")
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite")
    return cov


def _gls(X, y, C):
    W = np.linalg.pinv(C, hermitian=True)
    fisher = X.T @ W @ X
    cov_beta = np.linalg.pinv(fisher, hermitian=True)
    beta = cov_beta @ (X.T @ W @ y)
    resid = y - X @ beta
    chi2 = float(resid.T @ W @ resid)
    dof = int(len(y) - X.shape[1])
    p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")
    return beta, cov_beta, chi2, dof, p


class HighZTransientAuditor:
    """
    Fits y = alpha + b*x where
      y = ln(T_obs/T_rest), x = ln(1+z).
    Standard temporal closure: alpha=0 and b=1.
    """
    def __init__(self, events: Sequence[TransientEvent]):
        self.events = list(events)
        if len(self.events) < 2:
            raise ValueError("Need at least two events")
        for e in self.events:
            e.validate()

    def fit_global(self, systematic_fraction=0.0, cov_y=None, fit_intercept=True, quality_min=1):
        x, y, var, kept = _prepare(self.events, systematic_fraction, quality_min)
        C = np.diag(var) if cov_y is None else _validate_cov(cov_y, len(kept))
        if fit_intercept:
            X = np.column_stack([np.ones_like(x), x])
            model = "alpha + b ln(1+z)"
        else:
            X = x[:, None]
            model = "b ln(1+z); alpha=0"
        beta, cov, chi2, dof, p = _gls(X, y, C)
        return GLSFit(model, len(kept), beta, cov, chi2, dof, p,
                      min(e.z for e in kept), max(e.z for e in kept))

    def fit_high_z(self, z_threshold=2.5, **kwargs):
        subset = [e for e in self.events if e.z >= z_threshold]
        if len(subset) < 2:
            raise ValueError(f"Need >=2 events with z >= {z_threshold}; found {len(subset)}")
        return HighZTransientAuditor(subset).fit_global(**kwargs)

    def fit_binned(self, z_edges, min_count=3, systematic_fraction=0.0,
                   fit_intercept=True, quality_min=1):
        edges = np.asarray(z_edges, dtype=float)
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("z_edges must be strictly increasing")
        out = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if i == len(edges)-2:
                subset = [e for e in self.events if lo <= e.z <= hi]
            else:
                subset = [e for e in self.events if lo <= e.z < hi]
            if len(subset) < min_count:
                continue
            fit = HighZTransientAuditor(subset).fit_global(
                systematic_fraction=systematic_fraction,
                fit_intercept=fit_intercept, quality_min=quality_min)
            out.append(((float(lo), float(hi)), fit))
        return out

    def fit_by_environment(self, min_count=3, systematic_fraction=0.0,
                           fit_intercept=True, quality_min=1):
        groups: Dict[str, List[TransientEvent]] = {}
        for e in self.events:
            groups.setdefault(e.environment, []).append(e)
        out = {}
        for env, subset in groups.items():
            if len(subset) >= min_count:
                out[env] = HighZTransientAuditor(subset).fit_global(
                    systematic_fraction=systematic_fraction,
                    fit_intercept=fit_intercept, quality_min=quality_min)
        return out

    def fit_drift_model(self, systematic_fraction=0.0, cov_y=None,
                        fit_intercept=True, quality_min=1):
        """Fits y = alpha + b*x + c*x^2; standard closure is b=1,c=0."""
        x, y, var, kept = _prepare(self.events, systematic_fraction, quality_min)
        C = np.diag(var) if cov_y is None else _validate_cov(cov_y, len(kept))
        if fit_intercept:
            X = np.column_stack([np.ones_like(x), x, x**2])
            model = "alpha + b ln(1+z) + c ln^2(1+z)"
        else:
            X = np.column_stack([x, x**2])
            model = "b ln(1+z) + c ln^2(1+z); alpha=0"
        beta, cov, chi2, dof, p = _gls(X, y, C)
        return GLSFit(model, len(kept), beta, cov, chi2, dof, p,
                      min(e.z for e in kept), max(e.z for e in kept))

    def null_closure_test(self, systematic_fraction=0.0, cov_y=None,
                          quality_min=1, allow_calibration_offset=False):
        x, y, var, kept = _prepare(self.events, systematic_fraction, quality_min)
        C = np.diag(var) if cov_y is None else _validate_cov(cov_y, len(kept))
        W = np.linalg.pinv(C, hermitian=True)
        if allow_calibration_offset:
            one = np.ones_like(x)
            alpha = float((one @ W @ (y-x))/(one @ W @ one))
            model = alpha + x
            npar = 1
        else:
            alpha = 0.0
            model = x
            npar = 0
        resid = y - model
        chi2 = float(resid.T @ W @ resid)
        dof = len(y)-npar
        return {"n": len(y), "alpha_nuisance": alpha, "chi2": chi2,
                "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof))}

    @staticmethod
    def closure_from_fit(fit: GLSFit, z_eval: float, fit_intercept=True):
        x = math.log1p(z_eval)
        if fit_intercept:
            alpha, b = float(fit.params[0]), float(fit.params[1])
            g = np.array([1.0, x])
        else:
            alpha, b = 0.0, float(fit.params[0])
            g = np.array([x])
        log_ratio = alpha + (b-1.0)*x
        ratio = math.exp(log_ratio)
        R = ratio - 1.0
        var_log = float(g @ fit.covariance @ g)
        sigma_R = ratio*math.sqrt(max(var_log, 0.0))
        return {"z_eval": z_eval, "R_SN": R, "sigma_R_SN": sigma_R,
                "significance_sigma": R/sigma_R if sigma_R > 0 else float("nan"),
                "Gamma_eff_SN": 1.0+R}


def generate_synthetic_catalog(n=200, z_min=0.2, z_max=4.0, seed=42,
                               b_low=1.0, b_high=1.0, transition_z=2.5,
                               fractional_measurement_error=0.05,
                               environment_offset=None):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(z_min, z_max, n))
    if environment_offset is None:
        environment_offset = {"field": 0.0, "dense": 0.0}
    events = []
    for i, zi in enumerate(z):
        env = "field" if i % 2 == 0 else "dense"
        b = b_low if zi < transition_z else b_high
        t_rest = 20.0
        true_t_obs = t_rest*(1.0+environment_offset.get(env, 0.0))*(1.0+zi)**b
        sigma = fractional_measurement_error*true_t_obs
        measured = max(rng.normal(true_t_obs, sigma), 0.05*true_t_obs)
        events.append(TransientEvent(f"SYN-{i:04d}", float(zi), float(measured),
                                     t_rest, float(sigma), 0.0, env, 1))
    return events


def run_self_test():
    engine = CPGEngine()
    closure = engine.verify_numerical_closure()
    null_events = generate_synthetic_catalog(n=500, seed=11, b_high=1.0)
    inj_events = generate_synthetic_catalog(n=500, seed=11, b_high=1.08)
    null_fit = HighZTransientAuditor(null_events).fit_high_z(z_threshold=2.5, fit_intercept=False)
    inj_fit = HighZTransientAuditor(inj_events).fit_high_z(z_threshold=2.5, fit_intercept=False)
    return {
        "closure": closure,
        "null_b": float(null_fit.params[0]),
        "null_sigma_b": float(null_fit.stderr()[0]),
        "injected_b": float(inj_fit.params[0]),
        "injected_sigma_b": float(inj_fit.stderr()[0]),
    }


def fmt_fit(fit: GLSFit, fit_intercept=True):
    se = fit.stderr()
    if fit_intercept:
        head = f"alpha={fit.params[0]:+.6f}±{se[0]:.6f}, b={fit.params[1]:.6f}±{se[1]:.6f}"
    else:
        head = f"b={fit.params[0]:.6f}±{se[0]:.6f}"
    return f"{head}; chi2/dof={fit.chi2:.2f}/{fit.dof}; p={fit.p_value:.4g}; N={fit.n}; z=[{fit.z_min:.3f},{fit.z_max:.3f}]"


def analyze_catalog(path, systematic_fraction=0.0, z_threshold=2.5,
                    z_edges=None, fit_intercept=True):
    events = load_transient_csv(path)
    auditor = HighZTransientAuditor(events)
    print("CPG v0.3 HIGH-REDSHIFT TRANSIENT CLOSURE")
    print("="*72)
    print(f"Catalogue: {path} | N={len(events)}")
    global_fit = auditor.fit_global(systematic_fraction=systematic_fraction,
                                   fit_intercept=fit_intercept)
    print("Global:", fmt_fit(global_fit, fit_intercept))
    try:
        high = auditor.fit_high_z(z_threshold=z_threshold,
                                  systematic_fraction=systematic_fraction,
                                  fit_intercept=fit_intercept)
        print(f"High-z z>={z_threshold}:", fmt_fit(high, fit_intercept))
        print("Closure at z=3:", auditor.closure_from_fit(high, 3.0, fit_intercept))
    except ValueError as exc:
        print("High-z unavailable:", exc)
    if z_edges is not None:
        print("Binned fits:")
        for bounds, fit in auditor.fit_binned(z_edges, systematic_fraction=systematic_fraction,
                                              fit_intercept=fit_intercept):
            print(" ", bounds, fmt_fit(fit, fit_intercept))
    env = auditor.fit_by_environment(systematic_fraction=systematic_fraction,
                                     fit_intercept=fit_intercept)
    if env:
        print("Environment fits:")
        for name, fit in sorted(env.items()):
            print(" ", name, fmt_fit(fit, fit_intercept))
    null = auditor.null_closure_test(systematic_fraction=systematic_fraction,
                                    allow_calibration_offset=fit_intercept)
    print("Standard null:", null)


def main():
    p = argparse.ArgumentParser(description="CPG v0.3 high-redshift transient closure")
    p.add_argument("--catalog")
    p.add_argument("--systematic-fraction", type=float, default=0.0)
    p.add_argument("--z-threshold", type=float, default=2.5)
    p.add_argument("--z-bins", default="0,1,2,2.5,3,4,6")
    p.add_argument("--fix-intercept", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        for k, v in run_self_test().items():
            print(k, "=", v)
        return
    if not args.catalog:
        p.error("Provide --catalog PATH or use --self-test")
    edges = [float(x) for x in args.z_bins.split(",") if x.strip()]
    analyze_catalog(args.catalog, args.systematic_fraction, args.z_threshold,
                    edges, not args.fix_intercept)


if __name__ == "__main__":
    main()
