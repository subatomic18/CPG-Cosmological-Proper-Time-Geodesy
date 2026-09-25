#!/usr/bin/env python3
"""Small covariance-aware Gaussian-process utilities for CPG.

The implementation intentionally depends only on NumPy/SciPy. It uses a
Matern-3/2 kernel and a generalized linear mean. Observation covariance may
be fully non-diagonal.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg, optimize


@dataclass(frozen=True)
class GPPrediction:
    mean: np.ndarray
    covariance: np.ndarray
    amplitude: float
    length_scale: float
    mean_intercept: float
    mean_slope: float
    negative_log_likelihood: float


def _matern32(x1, x2, amplitude, length_scale):
    x1 = np.asarray(x1, dtype=float)[:, None]
    x2 = np.asarray(x2, dtype=float)[None, :]
    r = np.abs(x1 - x2)
    u = np.sqrt(3.0) * r / float(length_scale)
    return float(amplitude) ** 2 * (1.0 + u) * np.exp(-u)


class Matern32GP:
    """Matern-3/2 GP with full observational covariance and linear mean."""

    def __init__(self, x, y, covariance):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.C = np.asarray(covariance, dtype=float)
        n = len(self.x)
        if self.x.shape != (n,) or self.y.shape != (n,):
            raise ValueError("x and y must be one-dimensional and equal length")
        if n < 3:
            raise ValueError("Need at least three GP training points")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("GP x values must be strictly increasing")
        if self.C.shape != (n, n):
            raise ValueError("GP covariance shape mismatch")
        if not np.allclose(self.C, self.C.T, atol=1e-10, rtol=1e-10):
            raise ValueError("GP covariance must be symmetric")
        if np.linalg.eigvalsh(self.C).min() < -1e-8:
            raise ValueError("GP covariance must be positive semidefinite")
        self.x0 = float(np.mean(self.x))
        scale = float(np.ptp(self.x))
        self.xscale = scale if scale > 0 else 1.0
        self.X = np.column_stack((np.ones(n), (self.x - self.x0) / self.xscale))

    def _factor_terms(self, amplitude, length_scale):
        K = _matern32(self.x, self.x, amplitude, length_scale) + self.C
        jitter = max(1e-10, 1e-10 * float(np.median(np.diag(K))))
        K = K + np.eye(len(self.x)) * jitter
        L = linalg.cholesky(K, lower=True, check_finite=False)
        KiX = linalg.cho_solve((L, True), self.X, check_finite=False)
        Kiy = linalg.cho_solve((L, True), self.y, check_finite=False)
        normal = self.X.T @ KiX
        normal_inv = np.linalg.pinv(normal, hermitian=True)
        beta = normal_inv @ (self.X.T @ Kiy)
        resid = self.y - self.X @ beta
        alpha = linalg.cho_solve((L, True), resid, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * float(resid @ alpha) + 0.5 * logdet + 0.5 * len(self.x) * np.log(2*np.pi)
        return K, L, beta, alpha, normal_inv, nll

    def fit_predict(self, x_eval):
        xe = np.asarray(x_eval, dtype=float)
        if xe.ndim != 1:
            raise ValueError("x_eval must be one-dimensional")
        if np.any(xe < self.x[0]) or np.any(xe > self.x[-1]):
            raise ValueError("GP prediction does not extrapolate beyond training range")

        ystd = max(float(np.std(self.y)), 1.0)
        xrange = max(float(np.ptp(self.x)), 1e-3)
        amp0 = max(0.5 * ystd, 1.0)
        ell0 = max(0.35 * xrange, 0.05)
        bounds = [
            (np.log(max(0.05 * ystd, 0.1)), np.log(max(20.0 * ystd, 10.0))),
            (np.log(max(0.03 * xrange, 0.01)), np.log(max(5.0 * xrange, 0.2))),
        ]

        def objective(theta):
            try:
                return self._factor_terms(np.exp(theta[0]), np.exp(theta[1]))[-1]
            except (linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e100

        starts = [
            np.log([amp0, ell0]),
            np.log([ystd, max(0.15 * xrange, 0.02)]),
            np.log([ystd, max(0.8 * xrange, 0.05)]),
        ]
        fits = [optimize.minimize(objective, s, method="L-BFGS-B", bounds=bounds) for s in starts]
        best = min(fits, key=lambda r: float(r.fun))
        if not np.isfinite(best.fun):
            raise RuntimeError("GP hyperparameter optimization failed")
        amp, ell = np.exp(best.x)
        _, L, beta, alpha, normal_inv, nll = self._factor_terms(amp, ell)

        Xs = np.column_stack((np.ones(len(xe)), (xe - self.x0) / self.xscale))
        Kxs = _matern32(self.x, xe, amp, ell)
        mean = Xs @ beta + Kxs.T @ alpha

        KiKxs = linalg.cho_solve((L, True), Kxs, check_finite=False)
        Kss = _matern32(xe, xe, amp, ell)
        cov = Kss - Kxs.T @ KiKxs
        KiX = linalg.cho_solve((L, True), self.X, check_finite=False)
        V = Xs - Kxs.T @ KiX
        cov = cov + V @ normal_inv @ V.T
        cov = 0.5 * (cov + cov.T)

        w, v = np.linalg.eigh(cov)
        floor = max(1e-12, 1e-12 * float(np.max(np.diag(cov))))
        cov = (v * np.maximum(w, floor)) @ v.T
        cov = 0.5 * (cov + cov.T)

        return GPPrediction(
            mean=np.asarray(mean, float),
            covariance=np.asarray(cov, float),
            amplitude=float(amp),
            length_scale=float(ell),
            mean_intercept=float(beta[0]),
            mean_slope=float(beta[1] / self.xscale),
            negative_log_likelihood=float(nll),
        )
