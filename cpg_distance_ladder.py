"""CPG distance-ladder reproduction utilities.

This module reproduces the generalized least-squares (GLS) fit used by the
public SH0ES-2022 compact linear data release when supplied with the four
reformatted text files ``y_R22.txt``, ``C_R22.txt``, ``L_R22.txt`` and
``q_R22.txt``.

The standard fit is deliberately kept separate from any RTD-EU or
chronometric remapping. The first scientific requirement is that CPG recover
the published late-universe ladder result before altered physics is tested.

Data provenance
---------------
Original SH0ES high-level release:
    https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/SH0ES_Data
Convenient text reformat used by this reader:
    https://github.com/marcushogas/Cepheid-Distance-Ladder-Data/tree/main/SH0ES2022

The latter repository states that its SH0ES2022 files are a reformatted
version of the Riess et al. (2022) baseline data and provides a reference
notebook whose linear fit yields H0 = 73.04 +/- 1.01 km/s/Mpc.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import linalg


H0_PARAMETER = "5logH0"


@dataclass(frozen=True)
class DistanceLadderFit:
    """Result of the compact SH0ES-style linear distance-ladder fit."""

    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    parameter_errors: np.ndarray
    parameter_covariance: np.ndarray
    h0: float
    h0_error: float
    chi2: float
    dof: int
    n_data: int
    n_parameters: int

    def parameter(self, name: str) -> tuple[float, float]:
        """Return ``(value, 1-sigma error)`` for a named fit parameter."""
        try:
            i = self.parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return float(self.parameter_values[i]), float(self.parameter_errors[i])

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "h0_km_s_Mpc": self.h0,
            "h0_error_km_s_Mpc": self.h0_error,
            "chi2": self.chi2,
            "dof": self.dof,
            "n_data": self.n_data,
            "n_parameters": self.n_parameters,
            "parameters": {
                name: {
                    "value": float(value),
                    "error": float(error),
                }
                for name, value, error in zip(
                    self.parameter_names,
                    self.parameter_values,
                    self.parameter_errors,
                )
            },
        }


def h0_from_5logh0(value: float, sigma: float = 0.0) -> tuple[float, float]:
    """Convert the linear-fit parameter ``5 log10(H0)`` to H0 and sigma.

    H0 is returned in km s^-1 Mpc^-1 when the fitted parameter follows the
    SH0ES convention. The uncertainty is the first-order propagation of a
    Gaussian uncertainty in ``5 log10(H0)``.
    """

    h0 = 10.0 ** (float(value) / 5.0)
    h0_error = (np.log(10.0) / 5.0) * h0 * float(sigma)
    return float(h0), float(abs(h0_error))


def fit_gls(
    y: Sequence[float] | np.ndarray,
    design: np.ndarray,
    covariance: np.ndarray,
    parameter_names: Sequence[str],
) -> DistanceLadderFit:
    """Perform the SH0ES compact generalized least-squares fit.

    The model is ``y = L q + noise`` with data covariance ``C``. Rather than
    explicitly forming ``C^-1``, the implementation uses a Cholesky solve.
    This is algebraically equivalent to

        q = (L.T C^-1 L)^-1 L.T C^-1 y

    and is numerically preferable for a positive-definite covariance matrix.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    design = np.asarray(design, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    names = tuple(str(x) for x in parameter_names)

    if design.ndim != 2:
        raise ValueError("design must be a two-dimensional matrix")
    n_data, n_parameters = design.shape
    if y.size != n_data:
        raise ValueError("y length must match the number of design rows")
    if covariance.shape != (n_data, n_data):
        raise ValueError("covariance must have shape (n_data, n_data)")
    if len(names) != n_parameters:
        raise ValueError("parameter_names length must match design columns")
    if H0_PARAMETER not in names:
        raise ValueError(f"parameter_names must contain {H0_PARAMETER!r}")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(design)):
        raise ValueError("y and design must contain only finite values")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values")

    c_factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
    cinv_design = linalg.cho_solve(c_factor, design, check_finite=False)
    cinv_y = linalg.cho_solve(c_factor, y, check_finite=False)

    normal = design.T @ cinv_design
    rhs = design.T @ cinv_y

    n_factor = linalg.cho_factor(normal, lower=True, check_finite=False)
    q_fit = linalg.cho_solve(n_factor, rhs, check_finite=False)
    q_cov = linalg.cho_solve(
        n_factor, np.eye(n_parameters), check_finite=False
    )
    q_err = np.sqrt(np.clip(np.diag(q_cov), 0.0, None))

    residual = y - design @ q_fit
    cinv_residual = linalg.cho_solve(c_factor, residual, check_finite=False)
    chi2 = float(residual @ cinv_residual)
    dof = int(n_data - n_parameters)

    i_h0 = names.index(H0_PARAMETER)
    h0, h0_error = h0_from_5logh0(q_fit[i_h0], q_err[i_h0])

    return DistanceLadderFit(
        parameter_names=names,
        parameter_values=q_fit,
        parameter_errors=q_err,
        parameter_covariance=q_cov,
        h0=h0,
        h0_error=h0_error,
        chi2=chi2,
        dof=dof,
        n_data=n_data,
        n_parameters=n_parameters,
    )


def load_shoes2022_text(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load the reformatted SH0ES-2022 compact data products.

    Expected files in ``data_dir``:
      * y_R22.txt -- two columns, ``Source`` and ``Data``
      * C_R22.txt -- covariance matrix
      * L_R22.txt -- design matrix
      * q_R22.txt -- parameter names
    """

    data_dir = Path(data_dir)
    required = {
        "y": data_dir / "y_R22.txt",
        "C": data_dir / "C_R22.txt",
        "L": data_dir / "L_R22.txt",
        "q": data_dir / "q_R22.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing SH0ES-2022 file(s): " + ", ".join(missing))

    y = np.loadtxt(required["y"], usecols=(1,), skiprows=1, dtype=float)
    covariance = np.loadtxt(required["C"], delimiter="\t", dtype=float)
    design = np.loadtxt(required["L"], delimiter="\t", dtype=float)
    parameter_names = tuple(np.loadtxt(required["q"], dtype=str).tolist())
    return y, design, covariance, parameter_names


def fit_shoes2022_text(data_dir: str | Path) -> DistanceLadderFit:
    """Load and fit the reformatted SH0ES-2022 compact data release."""

    return fit_gls(*load_shoes2022_text(data_dir))


def published_baseline_check(
    fit: DistanceLadderFit,
    expected_h0: float = 73.04,
    expected_sigma: float = 1.01,
    h0_tolerance: float = 0.05,
    sigma_tolerance: float = 0.05,
) -> dict[str, float | bool]:
    """Compare a fit with the public reference notebook's linear-fit output.

    This is a reproduction check, not a statistical claim about which H0 value
    is correct. Defaults correspond to the reference notebook distributed
    with the reformatted SH0ES-2022 matrix data.
    """

    dh0 = fit.h0 - expected_h0
    dsigma = fit.h0_error - expected_sigma
    return {
        "expected_h0": float(expected_h0),
        "expected_sigma": float(expected_sigma),
        "delta_h0": float(dh0),
        "delta_sigma": float(dsigma),
        "within_tolerance": bool(
            abs(dh0) <= h0_tolerance and abs(dsigma) <= sigma_tolerance
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the SH0ES-2022 compact linear distance-ladder fit."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing y_R22.txt, C_R22.txt, L_R22.txt and q_R22.txt.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--show-parameters",
        action="store_true",
        help="Print all fitted ladder parameters.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fit = fit_shoes2022_text(args.data_dir)
    check = published_baseline_check(fit)

    if args.json:
        payload = fit.to_dict()
        payload["published_baseline_check"] = check
        print(json.dumps(payload, indent=2))
        return 0

    print("CPG SH0ES-2022 distance-ladder reproduction")
    print(f"N_data       : {fit.n_data}")
    print(f"N_parameters : {fit.n_parameters}")
    print(f"chi2 / dof   : {fit.chi2:.3f} / {fit.dof}")
    print(
        "H0           : "
        f"{fit.h0:.3f} +/- {fit.h0_error:.3f} km s^-1 Mpc^-1"
    )
    print(
        "Reference    : 73.04 +/- 1.01 km s^-1 Mpc^-1 "
        "(public reformatted-data notebook)"
    )
    print(f"Reproduced   : {check['within_tolerance']}")

    if args.show_parameters:
        print("\nParameters")
        for name, value, error in zip(
            fit.parameter_names, fit.parameter_values, fit.parameter_errors
        ):
            print(f"{name:12s} {value: .6f} +/- {error:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
