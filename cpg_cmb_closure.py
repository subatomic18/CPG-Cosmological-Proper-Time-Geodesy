"""CPG compressed-CMB acoustic-scale closure test.

This module provides a deliberately limited early-universe counterpart to the
CPG late-universe distance-ladder reproduction.  It takes compressed Planck
2018 base-LambdaCDM background quantities (physical baryon density, physical
cold-dark-matter density, and the accurately measured acoustic angular scale)
and solves for the Hubble constant required by a standard flat LambdaCDM
background using CAMB.

The purpose is *validation*, not a replacement for the full Planck likelihood.
The full Planck inference uses the complete TT/TE/EE spectra, low-l likelihood,
foreground/nuisance parameters, covariances, and parameter degeneracies.  A
compressed theta_* calculation can reproduce the central background scale but
must not be advertised as an independent Planck error-bar determination.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CMBCompressedInputs:
    ombh2: float
    omch2: float
    theta_star_100: float
    tau: float = 0.0544
    ns: float = 0.9649
    As: float = 2.1e-9
    sum_mnu_eV: float = 0.06
    Neff: float = 3.046
    Tcmb_K: float = 2.7255
    omk: float = 0.0


@dataclass(frozen=True)
class CMBAcousticClosureResult:
    h0: float
    theta_star_100_target: float
    theta_star_100_model: float
    z_star: float
    r_star_Mpc: float
    DA_star_Mpc: float
    age_Gyr: float
    ombh2: float
    omch2: float
    reference_h0: float | None = None
    reference_sigma: float | None = None

    @property
    def delta_h0(self) -> float | None:
        if self.reference_h0 is None:
            return None
        return self.h0 - self.reference_h0

    def to_dict(self) -> dict[str, float | None]:
        return {
            "H0_km_s_Mpc": self.h0,
            "theta_star_100_target": self.theta_star_100_target,
            "theta_star_100_model": self.theta_star_100_model,
            "z_star": self.z_star,
            "r_star_Mpc": self.r_star_Mpc,
            "DA_star_Mpc": self.DA_star_Mpc,
            "age_Gyr": self.age_Gyr,
            "ombh2": self.ombh2,
            "omch2": self.omch2,
            "reference_H0_km_s_Mpc": self.reference_h0,
            "reference_sigma_km_s_Mpc": self.reference_sigma,
            "delta_H0_km_s_Mpc": self.delta_h0,
        }


def _require_camb():
    try:
        import camb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CAMB is required for cpg_cmb_closure. Install with `pip install camb`."
        ) from exc
    return camb


def _camb_background(h0: float, inputs: CMBCompressedInputs) -> dict[str, float]:
    camb = _require_camb()
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=float(h0),
        ombh2=float(inputs.ombh2),
        omch2=float(inputs.omch2),
        omk=float(inputs.omk),
        mnu=float(inputs.sum_mnu_eV),
        nnu=float(inputs.Neff),
        TCMB=float(inputs.Tcmb_K),
        tau=float(inputs.tau),
    )
    pars.InitPower.set_params(As=float(inputs.As), ns=float(inputs.ns))
    # Background/thermodynamic quantities are sufficient; no C_l spectra needed.
    results = camb.get_background(pars)
    derived = results.get_derived_params()

    theta = float(derived["thetastar"])
    zstar = float(derived["zstar"])
    rstar = float(derived["rstar"])
    DAstar = float(derived["DAstar"])
    age = float(derived["age"])

    return {
        "theta_star_100": theta,
        "z_star": zstar,
        "r_star_Mpc": rstar,
        "DA_star_Mpc": DAstar,
        "age_Gyr": age,
    }


def solve_h0_from_acoustic_scale(
    inputs: CMBCompressedInputs,
    h0_bounds: tuple[float, float] = (50.0, 90.0),
    reference_h0: float | None = None,
    reference_sigma: float | None = None,
) -> CMBAcousticClosureResult:
    """Solve H0 so CAMB matches the supplied 100*theta_* acoustic scale.

    Physical densities ombh2 and omch2 are held fixed.  This is intentionally
    a compressed background-geometry closure calculation, not a full CMB
    parameter fit.
    """

    target = float(inputs.theta_star_100)
    lo, hi = map(float, h0_bounds)
    if not (0.0 < lo < hi):
        raise ValueError("h0_bounds must satisfy 0 < lower < upper")

    def residual(h0: float) -> float:
        return _camb_background(h0, inputs)["theta_star_100"] - target

    flo = residual(lo)
    fhi = residual(hi)
    if flo == 0.0:
        root = lo
    elif fhi == 0.0:
        root = hi
    elif flo * fhi > 0.0:
        raise ValueError(
            "H0 bracket does not contain an acoustic-scale root: "
            f"residual({lo})={flo:.6g}, residual({hi})={fhi:.6g}"
        )
    else:
        root = float(brentq(residual, lo, hi, xtol=1e-9, rtol=1e-11, maxiter=100))

    model = _camb_background(root, inputs)
    return CMBAcousticClosureResult(
        h0=root,
        theta_star_100_target=target,
        theta_star_100_model=model["theta_star_100"],
        z_star=model["z_star"],
        r_star_Mpc=model["r_star_Mpc"],
        DA_star_Mpc=model["DA_star_Mpc"],
        age_Gyr=model["age_Gyr"],
        ombh2=inputs.ombh2,
        omch2=inputs.omch2,
        reference_h0=reference_h0,
        reference_sigma=reference_sigma,
    )


def load_planck_compressed(path: str | Path) -> tuple[CMBCompressedInputs, dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload["inputs"]
    inputs = CMBCompressedInputs(
        ombh2=float(raw["ombh2"]),
        omch2=float(raw["omch2"]),
        theta_star_100=float(raw["theta_star_100"]),
        tau=float(raw.get("tau", 0.0544)),
        ns=float(raw.get("ns", 0.9649)),
        As=float(raw.get("As", 2.1e-9)),
        sum_mnu_eV=float(raw.get("sum_mnu_eV", 0.06)),
        Neff=float(raw.get("Neff", 3.046)),
        Tcmb_K=float(raw.get("Tcmb_K", 2.7255)),
        omk=float(raw.get("omk", 0.0)),
    )
    return inputs, payload


def run_planck_compressed(path: str | Path) -> CMBAcousticClosureResult:
    inputs, payload = load_planck_compressed(path)
    ref = payload.get("published_reference", {})
    return solve_h0_from_acoustic_scale(
        inputs,
        reference_h0=float(ref["H0_km_s_Mpc"]) if "H0_km_s_Mpc" in ref else None,
        reference_sigma=float(ref["H0_sigma_km_s_Mpc"]) if "H0_sigma_km_s_Mpc" in ref else None,
    )


def published_central_value_check(
    result: CMBAcousticClosureResult,
    tolerance_km_s_Mpc: float = 0.5,
) -> dict[str, float | bool | None]:
    delta = result.delta_h0
    return {
        "reference_h0": result.reference_h0,
        "delta_h0": delta,
        "tolerance_km_s_Mpc": float(tolerance_km_s_Mpc),
        "within_tolerance": bool(
            delta is not None and abs(delta) <= float(tolerance_km_s_Mpc)
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPG compressed Planck acoustic-scale H0 closure test."
    )
    p.add_argument(
        "data_file",
        nargs="?",
        default="data/planck2018_cmb_compressed.json",
        help="Path to compressed Planck JSON inputs.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON output.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_planck_compressed(args.data_file)
    check = published_central_value_check(result)

    if args.json:
        payload = result.to_dict()
        payload["published_central_value_check"] = check
        payload["scope_note"] = (
            "Compressed acoustic-scale background closure only; not a full Planck likelihood."
        )
        print(json.dumps(payload, indent=2))
        return 0

    print("CPG Planck-2018 compressed CMB acoustic-scale closure")
    print(f"100 theta* target : {result.theta_star_100_target:.7f}")
    print(f"100 theta* model  : {result.theta_star_100_model:.7f}")
    print(f"ombh2             : {result.ombh2:.6f}")
    print(f"omch2             : {result.omch2:.6f}")
    print(f"z*                 : {result.z_star:.3f}")
    print(f"r*                 : {result.r_star_Mpc:.3f} Mpc")
    print(f"D_A(z*)            : {result.DA_star_Mpc:.3f} Mpc")
    print(f"Age                : {result.age_Gyr:.4f} Gyr")
    print(f"H0                 : {result.h0:.3f} km s^-1 Mpc^-1")
    if result.reference_h0 is not None:
        print(f"Published reference: {result.reference_h0:.3f} km s^-1 Mpc^-1")
        print(f"Delta H0           : {result.delta_h0:+.3f} km s^-1 Mpc^-1")
        print(f"Central reproduced : {check['within_tolerance']}")
    print("NOTE: this reproduces the compressed central background scale only;")
    print("      the full Planck uncertainty requires the full CMB likelihood.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
