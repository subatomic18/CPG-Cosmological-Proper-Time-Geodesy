# CPG Prediction Register v1.0

**Cosmological Proper-Time Geodesy**  
**Author:** Jeffery Barnes  
**Affiliation:** Independent Theoretical Physics Research Group, Montreal, QC, Canada  
**Freeze date:** 28 September 2026  
**Status:** Prospective prediction and falsification register

## 1. Purpose

Cosmological Proper-Time Geodesy (CPG) tests whether proper time accumulated along physically specified cosmological worldlines closes consistently with the predictions of general relativity and standard cosmological structure formation.

CPG does not assume that environmental proper-time differences explain the Hubble tension.

The primary invariant statistic is

\[
\mathcal M_{AB} \equiv \frac{\tau_A-\tau_B}{\tau_{\rm ref}},
\]

where

\[
\tau_i=\frac{1}{c}\int_{\gamma_i}\sqrt{-g_{\mu\nu}\,dx^\mu dx^\nu}.
\]

Worldlines A and B must be compared between physically defined boundary hypersurfaces rather than arbitrary coordinate-time surfaces.

## 2. FLRW null prediction - CPG-P1

For equivalent comoving geodesics in an exactly homogeneous and isotropic FLRW spacetime,

\[
\mathcal M_{AB}=0.
\]

A homogeneous lapse transformation cannot by itself generate physical CPG misclosure. Any implementation producing nonzero misclosure solely through a reparameterization of FLRW time fails this null test.

## 3. Weak-field environmental prediction - CPG-P2

For nonrelativistic motion through weak cosmological gravitational fields,

\[
d\tau \simeq dt\left(1+\frac{\Phi}{c^2}-\frac{v^2}{2c^2}\right).
\]

For two worldlines,

\[
\Delta\tau_{AB}\simeq
\int\left[
\frac{\Phi_A-\Phi_B}{c^2}
-\frac{v_A^2-v_B^2}{2c^2}
\right]dt+\Delta\tau_\Sigma.
\]

Here \(\Delta\tau_\Sigma\) represents the contribution arising from the physical definition of the endpoint hypersurfaces.

Analytic and exploratory weak-field estimates completed before the freeze date indicate a characteristic environmental fractional scale of approximately

\[
|\mathcal M_{\rm env}|\sim10^{-7}-10^{-5},
\]

with unusually deep cluster worldlines potentially extending to several times \(10^{-5}\).

This is an order-of-magnitude forecast, not a rigorous universal upper bound.

## 4. Potential-depth prediction - CPG-P3

Holding other contributions fixed,

\[
\frac{\partial\tau}{\partial\Phi}>0.
\]

Worldlines residing for extended periods in deeper gravitational potentials should accumulate less proper time than otherwise comparable worldlines occupying shallower potentials.

The relevant environmental variable is reconstructed gravitational potential \(\Phi(\mathbf{x},z)\), rather than categorical labels such as void, field, group, or cluster.

CPG therefore predicts systematic dependence on halo mass, gravitational potential, normalized halo-centric radius \(r/R_{200}\), halo concentration, and formation history. For otherwise comparable halos, increasing potential depth and decreasing halo-centric radius should produce increasingly negative proper-time offsets relative to a common reference worldline.

## 5. Percent-level prediction withdrawn - CPG-P4

Earlier exploratory CPG development considered accumulated environmental proper-time differences of order

\[
|\Delta\tau|/\tau\sim10^{-2}-5\times10^{-2}.
\]

**CPG Prediction Register v1.0 does not adopt this as a prediction.**

Calculations completed before the freeze date did not derive such an amplitude from ordinary weak-field gravitational and kinematic effects. The prospective characteristic scale is instead approximately \(10^{-7}-10^{-5}\) for typical environmental effects, subject to future full numerical calculation.

A future observation with \(|\mathcal M|\gg10^{-5}\) is not automatically a confirmation of CPG. Systematics, missing conventional contributions, endpoint definitions, and the relativistic calculation must first be re-examined.

## 6. Hubble-residual prediction - CPG-P5

For every source \(i\), CPG requires that its worldline statistic \(\mathcal M_i\) be reconstructed **without using the source's Hubble residual**.

Only afterward is

\[
R_{H,i}=\frac{H_{0,i}-H_{0,\rm ref}}{H_{0,\rm ref}}
\]

examined.

The prospective test compares a prespecified relationship

\[
R_H=F(\mathcal M,z,\ldots)
\]

against the null hypothesis that \(R_H\) is independent of \(\mathcal M\).

The functional relationship and parameters used for prospective testing must be specified on training/calibration data before held-out future observations are examined.

## 7. Supernova time-dilation consistency - CPG-P6

CPG must reproduce the established cosmological time-dilation relation

\[
\Delta t_{\rm obs}\simeq(1+z)\Delta t_{\rm em}.
\]

Therefore CPG does not permit an unrestricted additional percent-level distortion of observed supernova light-curve timescales. Proper-time misclosure and photon/light-curve time dilation are distinct observables and must not be conflated.

## 8. Cross-probe closure - CPG-P7

If \(\mathcal M\) represents a physical worldline property, independent cosmological probes must be compatible with a common underlying spacetime/worldline reconstruction.

Schematically,

\[
\mathcal M_{\rm SN}\simeq
\mathcal M_{\rm TRGB}\simeq
\mathcal M_{\rm CC}\simeq
\mathcal M_{\rm BAO}\simeq
\mathcal M_{\rm GW},
\]

after accounting for each probe's distinct observational response.

If mutually incompatible correction functions are required for different probes, the universality hypothesis fails.

## 9. Prospective-data rule - CPG-P8

The theoretical framework, environmental variables, sign predictions, and forecast methodology described here are frozen on **28 September 2026**.

Measurements released subsequently are to be evaluated first using the frozen prediction.

The model must not first be refitted to a new dataset and then presented as having predicted that dataset. Refitting is permitted only after the prospective result has been recorded.

## 10. Explicit failure conditions

The tested CPG hypothesis is weakened or falsified in the relevant regime if:

1. equivalent FLRW comoving worldlines produce physical misclosure;
2. predicted environmental signs are systematically reversed;
3. the predicted relationship between independently calculated \(\mathcal M\) and the targeted observable is absent;
4. apparent correlations disappear after conventional peculiar-velocity, gravitational, or selection corrections;
5. different probes require incompatible proper-time mappings;
6. the framework violates established cosmological time dilation;
7. successful predictions require dataset-by-dataset retuning; or
8. a claimed effect cannot be expressed through coordinate-invariant observables.

Negative results are to be retained and reported.

## 11. Numerical benchmark status

As of the freeze date, analytic and exploratory weak-field calculations consistently indicate

\[
|\mathcal M_{\rm env}|\sim10^{-7}-10^{-5}
\]

for ordinary late-time structures, with the deepest cluster trajectories potentially reaching several times \(10^{-5}\).

These values are benchmarks, not a finalized simulation-calibrated probability distribution.

A reproducible numerical forecast using documented halo growth, concentration, velocity, and potential histories should supersede these benchmark estimates in a separately versioned prediction release. The v1.0 predictions themselves must not be silently rewritten.

## 12. Primary prospective statement

> **Frozen CPG prediction - 2026-09-28:** Proper time calculated along physically specified cosmological worldlines is expected under standard weak-field general relativity to exhibit environmental fractional misclosure characteristically of order \(10^{-7}\) to \(10^{-5}\), with deeper gravitational potentials producing smaller accumulated proper time, all else equal. No percent-level environmental proper-time lapse is assumed. Any proposed observable CPG signal must track an independently reconstructed worldline statistic, survive conventional corrections, remain consistent across cosmological probes, and be tested on future data without prior refitting.

## Version declaration

**CPG Prediction Register v1.0**  
Frozen: **2026-09-28**

This release establishes the prospective baseline. Subsequent scientific changes must be issued as new versions. Version 1.0 should remain permanently accessible for prospective comparison and must not be overwritten.

### Version history

- **v1.0 - 2026-09-28:** Initial frozen prospective prediction and falsification register. Records withdrawal of the earlier exploratory percent-level environmental lapse expectation and adopts the weak-field benchmark scale as an order-of-magnitude forecast pending a reproducible simulation-calibrated release.
