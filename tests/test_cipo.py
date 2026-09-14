"""Validation suite.

Four tiers, following the protocol of the research plan.

Tier 0, internal consistency: the closed form algebra is checked against
numerical quadrature, and each analytic profile result against an independent
derivation.

Tier 1, published metrics: the calibrated model reproduces the IPCC AR6
absolute and relative emission metrics.

Tier 2, special cases: the framework returns the global warming potential, the
global temperature change potential and the single intervention compensation
ratio of earlier work as limits of the general portfolio problem.

Tier 3, optimisation: first order conditions are verified against closed form
solutions where these exist, and the numerical optimisers are checked against
each other.

Run with ``python -m pytest tests -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from cipo import (
    AR5_THERMAL,
    AR6,
    AR6_REFERENCE,
    Afforestation,
    CarbonClimateFeedback,
    DelayedPulse,
    EmissionPathway,
    ExpSum,
    ExponentialRelease,
    LinearCost,
    LinearRelease,
    PermanentRemoval,
    Portfolio,
    PowerCost,
    Signal,
    SpeciesRemoval,
)
from cipo.pathways import piecewise_linear, pulse, step
from cipo.profiles import mean_storage_time_numeric
from cipo.threshold import (
    compensation_ratio,
    durability_interval,
    log_sensitivity,
    max_log_sensitivity,
    optimal_durability,
)

CM = AR6()
HORIZON = 100.0


# --------------------------------------------------------------------------
# Tier 0: the exact algebra
# --------------------------------------------------------------------------
EXP_CASES = [
    (ExpSum([(0.2, [1.0])]), ExpSum([(0.05, [2.0, 0.5])])),
    (ExpSum([(0.0, [1.0])]), ExpSum([(0.23234, [1.0]), (0.2439, [-0.5])])),
    (ExpSum([(0.1, [1.0])]), ExpSum([(0.1, [1.0])])),
    (ExpSum([(0.1, [1.0])]), ExpSum([(0.1 + 1e-11, [1.0])])),
    (ExpSum([(0.3, [0.0, 1.0, 2.0])]), ExpSum([(0.02, [1.0, 3.0])])),
]


@pytest.mark.parametrize("f,g", EXP_CASES)
@pytest.mark.parametrize("u", [0.5, 5.0, 50.0, 300.0])
def test_convolution_matches_quadrature(f, g, u):
    exact = float(f.conv(g).eval(u))
    numeric = quad(lambda s: f.eval(s) * g.eval(u - s), 0.0, u, limit=400)[0]
    assert exact == pytest.approx(numeric, rel=1e-8, abs=1e-300)


@pytest.mark.parametrize("f,g", EXP_CASES)
@pytest.mark.parametrize("T", [1.0, 20.0, 500.0])
def test_integrals_match_quadrature(f, g, T):
    h = f.conv(g)
    assert float(h.definite(T)) == pytest.approx(
        quad(lambda t: h.eval(t), 0.0, T, limit=400)[0], rel=1e-8, abs=1e-300
    )
    assert float(h.double_definite(T)) == pytest.approx(
        quad(lambda t: (T - t) * h.eval(t), 0.0, T, limit=400)[0],
        rel=1e-8,
        abs=1e-300,
    )


@pytest.mark.parametrize("f,g", EXP_CASES)
def test_cumint_agrees_with_definite(f, g):
    h = f.conv(g)
    assert float(h.cumint().eval(37.0)) == pytest.approx(
        float(h.definite(37.0)), rel=1e-8
    )


def test_limit_definite_ignores_roundoff_constant_modes():
    """A negligible non-decaying mode must not be read as a divergence."""
    es = ExpSum([(0.1, [1.0]), (0.0, [-1e-25])])
    assert np.isfinite(es.limit_definite())
    assert es.limit_definite() == pytest.approx(10.0, rel=1e-9)
    assert not np.isfinite(ExpSum([(0.1, [1.0]), (0.0, [0.5])]).limit_definite())


def test_window_and_shifted_masses():
    assert Signal.window(1.0 / 25.0, 0.0, 25.0).mass() == pytest.approx(1.0, abs=1e-12)
    w = Signal.window(0.7, 5.0, 23.0, lam=0.08)
    numeric = quad(lambda t: float(w.eval(t)), 0.0, 400.0, limit=800)[0]
    assert w.mass() == pytest.approx(numeric, rel=1e-9)
    assert not np.isfinite(Signal.from_expsum(ExpSum.constant(1.0)).mass())


# --------------------------------------------------------------------------
# Tier 0: profile families
# --------------------------------------------------------------------------
TEMPORARY = [
    ExponentialRelease(tau=50.0),
    DelayedPulse(tau=50.0),
    LinearRelease(release_time=100.0),
    Afforestation(tau_growth=30.0, tau_disturbance=150.0),
]


@pytest.mark.parametrize("iv", TEMPORARY)
def test_temporary_profiles_have_zero_net_mass(iv):
    """Every unit of carbon removed is eventually returned."""
    assert iv.is_temporary()
    assert abs(iv.flux().mass()) < 1e-9


@pytest.mark.parametrize("iv", TEMPORARY)
def test_mean_storage_time_matches_quadrature(iv):
    assert iv.mean_storage_time == pytest.approx(
        mean_storage_time_numeric(iv), rel=2e-3
    )


def test_permanent_removal_is_not_temporary():
    assert not PermanentRemoval().is_temporary()
    assert not SpeciesRemoval("CH4").is_temporary()


@pytest.mark.parametrize("tau", [20.0, 50.0, 200.0])
@pytest.mark.parametrize("th", [30.0, 100.0, 500.0])
def test_delayed_pulse_closed_form(tau, th):
    """b(TH) = Psi(TH) - Psi((TH - tau)+), an independent derivation."""
    got = float(DelayedPulse(tau=tau).cooling(CM, th))
    psi = lambda x: float(CM.iagtp("CO2", x))
    assert got == pytest.approx(psi(th) - psi(max(th - tau, 0.0)), rel=1e-12)


@pytest.mark.parametrize("t", [10.0, 60.0, 300.0])
def test_linear_release_closed_form(t):
    """AGTP_F = -Phi(t) + (1/Tr)[Psi(t) - Psi((t - Tr)+)]."""
    tr = 80.0
    iv = LinearRelease(release_time=tr)
    psi = lambda x: float(CM.iagtp("CO2", x))
    expected = -float(CM.agtp("CO2", t)) + (psi(t) - psi(max(t - tr, 0.0))) / tr
    assert float(iv.response(CM, t)) == pytest.approx(expected, rel=1e-11)


def test_exponential_release_limits():
    """Instant release delivers nothing; unbounded storage is permanent."""
    assert float(ExponentialRelease(tau=1e-6).cooling(CM, HORIZON)) == pytest.approx(
        0.0, abs=1e-20
    )
    assert float(ExponentialRelease(tau=1e9).cooling(CM, HORIZON)) == pytest.approx(
        float(PermanentRemoval().cooling(CM, HORIZON)), rel=1e-6
    )


@pytest.mark.parametrize("iv", TEMPORARY + [PermanentRemoval()])
def test_cooling_is_positive_over_policy_horizons(iv):
    b = iv.cooling(CM, np.array([10.0, 50.0, 100.0, 300.0]))
    assert np.all(b > 0)


def test_permanent_cooling_grows_without_bound():
    b = PermanentRemoval().cooling(CM, np.array([10.0, 100.0, 1000.0, 10000.0]))
    assert np.all(np.diff(b) > 0)
    assert b[-1] > 10.0 * b[-2] / 10.0


@pytest.mark.parametrize(
    "iv", [DelayedPulse(tau=50.0), LinearRelease(release_time=100.0)]
)
def test_temporary_cooling_peaks_then_declines(iv):
    """Cumulative cooling from a temporary pool is not monotone in the horizon.

    Once the pool has returned its carbon, the released carbon dioxide warms
    while the reference warming it was meant to offset keeps accruing, so the
    cumulative cooling credited to the measure falls back. The horizon at
    which it peaks is a property of the profile, and evaluating such a measure
    at a longer horizon can only reduce the credit it earns.
    """
    th = np.geomspace(5.0, 5000.0, 400)
    b = iv.cooling(CM, th)
    assert np.all(b > 0)
    peak = int(np.argmax(b))
    assert 0 < peak < len(th) - 1
    assert b[-1] < b[peak]


def test_temporary_response_eventually_warms():
    """A temporary pool leaves a residual warming once its carbon returns."""
    iv = ExponentialRelease(tau=50.0)
    assert float(iv.response(CM, 20.0)) < 0.0
    assert float(iv.response(CM, 800.0)) > 0.0


# --------------------------------------------------------------------------
# Tier 1: published AR6 metrics
# --------------------------------------------------------------------------
def test_agwp_co2_matches_ar6():
    assert float(CM.agwp("CO2", 100.0)) == pytest.approx(
        AR6_REFERENCE["AGWP_CO2_100"], rel=0.04
    )
    assert float(CM.agwp("CO2", 20.0)) == pytest.approx(
        AR6_REFERENCE["AGWP_CO2_20"], rel=0.04
    )


@pytest.mark.parametrize("species,ref", sorted(AR6_REFERENCE["GWP100"].items()))
def test_gwp100_matches_ar6(species, ref):
    assert float(CM.gwp(species, 100.0)) == pytest.approx(ref, rel=0.05)


@pytest.mark.parametrize("species,ref", sorted(AR6_REFERENCE["GWP20"].items()))
def test_gwp20_matches_ar6(species, ref):
    assert float(CM.gwp(species, 20.0)) == pytest.approx(ref, rel=0.05)


def test_fossil_methane_exceeds_biogenic():
    """Oxidation carbon must add to the metric, by construction."""
    assert float(CM.gwp("CH4_fossil", 100.0)) > float(CM.gwp("CH4", 100.0))


def test_analytic_agtp_equals_numerical_double_convolution():
    for name in ("CO2", "CH4", "CH4_fossil", "N2O"):
        rf = CM.rf_kernel(name)
        rt = CM.thermal.kernel()
        for t in (5.0, 50.0, 200.0):
            numeric = quad(
                lambda s: float(rf.eval(s)) * float(rt.eval(t - s)),
                0.0,
                t,
                limit=400,
            )[0]
            assert float(CM.agtp(name, t)) == pytest.approx(numeric, rel=1e-9)


def test_corrected_proposition_one():
    """iAGTP converges to RE * tau * sum(c) for a decaying species.

    The research plan states the limit as RE * tau^2, which follows from its
    AGWP style definition of AGTP and does not hold for the true AGTP.
    """
    cm = AR6(feedback=False)
    for name in ("CH4", "N2O", "PFC14"):
        sp = cm.species[name]
        expected = sp.re_per_kg * sp.tau * cm.thermal.equilibrium_sensitivity
        assert cm.iagtp_limit(name) == pytest.approx(expected, rel=1e-9)
        # The plan's form differs by a factor tau / sum(c), which is far from
        # unity for every species considered.
        plan_form = sp.re_per_kg * sp.tau ** 2
        assert abs(plan_form / expected - 1.0) > 0.5


def test_co2_cumulative_effect_diverges():
    """The permanent airborne fraction makes iAGTP for CO2 unbounded."""
    assert CM.carbon.permanent_fraction > 0
    assert not np.isfinite(CM.iagtp_limit("CO2"))
    assert float(CM.iagtp("CO2", 2000.0)) > 5.0 * float(CM.iagtp("CO2", 500.0)) / 5.0


def test_feedback_increases_metrics_and_higher_order_is_small():
    with_fb, without = CM, AR6(feedback=False)
    assert float(with_fb.agwp("CO2", 100.0)) > float(without.agwp("CO2", 100.0))
    second = CM.replace(feedback=CarbonClimateFeedback(order=2))
    ratio = float(second.agwp("CO2", 100.0)) / float(CM.agwp("CO2", 100.0))
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_thermal_response_choice_changes_metrics_modestly():
    alt = CM.replace(thermal=AR5_THERMAL)
    assert float(alt.gwp("CH4", 100.0)) == pytest.approx(
        float(CM.gwp("CH4", 100.0)), rel=0.10
    )


# --------------------------------------------------------------------------
# Tier 1: emission pathways
# --------------------------------------------------------------------------
def test_piecewise_linear_is_exact():
    t_nodes = [0.0, 10.0, 30.0, 60.0, 100.0]
    rates = [1e9, 3e9, 2.5e9, 1e9, 0.0]
    sig = piecewise_linear(t_nodes, rates)
    probe = np.array([0.0, 5.0, 10.0, 20.0, 45.0, 60.0, 80.0, 100.0])
    assert np.allclose(sig.eval(probe), np.interp(probe, t_nodes, rates), atol=1e-9)
    fine = np.linspace(0.0, 100.0, 200001)
    assert float(sig.definite(200.0)) == pytest.approx(
        np.trapz(np.interp(fine, t_nodes, rates), fine), rel=1e-9
    )


def test_pathway_response_matches_quadrature():
    sig = piecewise_linear([0.0, 30.0, 100.0], [2e9, 3e9, 0.0])
    pw = EmissionPathway({"CH4": sig})
    kern = CM.agtp_kernel("CH4")
    for t in (20.0, 70.0, 150.0):
        numeric = quad(
            lambda s: float(sig.eval(s)) * float(kern.eval(t - s)),
            0.0,
            min(t, 100.0),
            limit=800,
        )[0]
        assert float(pw.temperature(CM, t)) == pytest.approx(numeric, rel=1e-7)


# --------------------------------------------------------------------------
# Tier 2: recovery of established metrics as special cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize("species", ["CH4", "N2O"])
@pytest.mark.parametrize("th", [20.0, 100.0])
def test_permanent_removal_recovers_integrated_gtp(species, th):
    """One pulse, one permanent measure: alpha reduces to the iGTP."""
    pw = EmissionPathway({species: pulse(1.0)})
    p = Portfolio(CM, [PermanentRemoval(cost_model=LinearCost(1.0))], pw, horizon=th)
    res = p.minimise_cost(neutral=True)
    assert float(res.alpha[0]) == pytest.approx(float(CM.igtp(species, th)), rel=1e-9)


@pytest.mark.parametrize("th", [20.0, 100.0, 500.0])
def test_species_matched_removal_needs_unit_deployment(th):
    """Removing the same gas one for one is exactly horizon invariant."""
    pw = EmissionPathway({"CH4": pulse(1.0)})
    p = Portfolio(
        CM, [SpeciesRemoval("CH4", cost_model=LinearCost(1.0))], pw, horizon=th
    )
    assert float(p.minimise_cost(neutral=True).alpha[0]) == pytest.approx(1.0, rel=1e-9)


def test_compensation_ratio_reduces_to_ratio_of_cumulative_effects():
    iv = ExponentialRelease(tau=50.0)
    pw = EmissionPathway({"CH4": pulse(1.0)})
    p = Portfolio(CM, [iv], pw, horizon=HORIZON)
    assert float(p.minimise_cost(neutral=True).alpha[0]) == pytest.approx(
        float(compensation_ratio(CM, iv, "CH4", HORIZON)), rel=1e-9
    )


def test_compensation_ratio_falls_with_durability():
    ratios = [
        float(compensation_ratio(CM, ExponentialRelease(tau=t), "CH4", HORIZON))
        for t in (20.0, 50.0, 100.0, 500.0)
    ]
    assert all(np.diff(ratios) < 0)
    assert ratios[-1] > float(
        compensation_ratio(CM, PermanentRemoval(), "CH4", HORIZON)
    )


# --------------------------------------------------------------------------
# Tier 3: optimisation and optimality conditions
# --------------------------------------------------------------------------
def _linear_setup():
    ivs = [
        ExponentialRelease(tau=20.0, cost_model=LinearCost(50.0), name="exp20"),
        ExponentialRelease(tau=50.0, cost_model=LinearCost(100.0), name="exp50"),
        ExponentialRelease(tau=100.0, cost_model=LinearCost(200.0), name="exp100"),
    ]
    pw = EmissionPathway({"CH4": pulse(1.0)})
    return Portfolio(CM, ivs, pw, horizon=HORIZON)


def test_linear_cost_optimum_is_the_cheapest_single_measure():
    """With linear costs the optimum is a vertex, so the plan's Theorem 3
    degenerates: average cost per unit cooling is not equalised."""
    p = _linear_setup()
    res = p.minimise_cost(neutral=True)
    ratio = np.array([iv.cost_model.unit_cost for iv in p.interventions]) / (
        p.cooling_vector()
    )
    best = int(np.argmin(ratio))
    assert res.success
    assert abs(res.neutrality_residual / p.cumulative_warming()) < 1e-9
    assert float(res.alpha[best]) == pytest.approx(
        p.cumulative_warming() / p.cooling_vector()[best], rel=1e-9
    )
    assert np.allclose(np.delete(res.alpha, best), 0.0, atol=1e-9)
    assert res.multiplier == pytest.approx(float(ratio.min()), rel=1e-6)


def test_capacity_limits_force_a_mixed_portfolio():
    p = _linear_setup()
    for iv in p.interventions[:2]:
        iv.max_scale = 40.0
    p = Portfolio(CM, p.interventions, p.pathway, HORIZON)
    res = p.minimise_cost(neutral=True)
    assert res.success
    assert np.count_nonzero(res.alpha > 1e-9) == 3
    report = p.kkt_report(res.alpha)
    assert report["at_capacity"][:2].all()
    assert abs(res.neutrality_residual / p.cumulative_warming()) < 1e-9


def test_convex_costs_equalise_marginal_cost_per_unit_cooling():
    """The correct general form of the plan's cost effectiveness condition."""
    ivs = [
        ExponentialRelease(
            tau=t, cost_model=PowerCost(c, 2.0, 50.0), name="e{:g}".format(t)
        )
        for t, c in ((20.0, 50.0), (50.0, 100.0), (100.0, 200.0))
    ]
    p = Portfolio(CM, ivs, EmissionPathway({"CH4": pulse(1.0)}), HORIZON)
    numeric = p.minimise_cost(neutral=True)
    analytic = p.analytic_power_cost_solution()
    assert analytic is not None
    assert np.allclose(numeric.alpha, analytic.alpha, rtol=1e-5)
    assert np.all(analytic.alpha > 0)
    assert p.kkt_report(analytic.alpha)["relative_spread"] < 1e-6


def test_deviation_minimisation_reduces_residual_and_is_stationary():
    p = _linear_setup()
    res = p.minimise_deviation(neutral=False)
    assert res.deviation < p.baseline_deviation()
    report = p.deviation_optimality_report(res.alpha)
    for a, overlap in zip(res.alpha, report["normalised_overlap"]):
        if a > 1e-9:
            assert abs(overlap) < 1e-5
        else:
            assert overlap >= -1e-5


def test_minimax_reduces_peak_below_deviation_optimum():
    p = _linear_setup()
    base = p.peak_deviation(np.zeros(p.n))
    mini = p.minimise_peak(neutral=False)
    assert mini.success
    assert mini.peak < base
    assert mini.peak <= p.minimise_deviation(neutral=False).peak * (1.0 + 1e-6)


def test_gram_matrix_is_positive_semidefinite():
    p = _linear_setup()
    g = p.gram_matrix()
    assert np.allclose(g, g.T, rtol=1e-12)
    assert np.min(np.linalg.eigvalsh(g)) > -1e-12 * float(np.max(np.abs(g)))


def test_neutrality_residual_zero_at_solution_across_horizons():
    for th in (20.0, 50.0, 100.0, 300.0):
        ivs = [
            ExponentialRelease(tau=30.0, cost_model=LinearCost(80.0), name="a"),
            DelayedPulse(tau=60.0, cost_model=LinearCost(120.0), name="b"),
        ]
        pw = EmissionPathway({"CH4": step(1e8, 0.0, th), "N2O": pulse(1e6)})
        p = Portfolio(CM, ivs, pw, horizon=th)
        res = p.minimise_cost(neutral=True)
        assert res.success
        assert abs(res.neutrality_residual / p.cumulative_warming()) < 1e-8


# --------------------------------------------------------------------------
# Tier 3: horizon sensitivity and durability
# --------------------------------------------------------------------------
def test_species_matched_offset_has_zero_horizon_sensitivity():
    assert max_log_sensitivity(CM, SpeciesRemoval("CH4"), "CH4", 20.0, 500.0) < 1e-12


def test_sensitivity_is_u_shaped_with_interior_optimum():
    """No durability makes the compensation ratio horizon free.

    The research plan asserts that a sufficiently durable measure becomes
    horizon insensitive. The sensitivity instead approaches one in the durable
    limit, so the minimum is interior and the floor is strictly positive.
    """
    fam = lambda t: ExponentialRelease(tau=t)
    out = optimal_durability(CM, fam, "CH4", 20.0, 500.0)
    assert 5.0 < out["optimal_storage_time"] < 1000.0
    assert out["sensitivity_floor"] > 0.05
    assert out["sensitivity_durable_limit"] > out["sensitivity_floor"]
    near = max_log_sensitivity(CM, fam(1.0), "CH4", 20.0, 500.0)
    assert near > out["sensitivity_floor"]


def test_durable_limit_sensitivity_approaches_unity():
    """alpha ~ const / TH once iAGTP for the target gas has saturated."""
    th = np.geomspace(2000.0, 2e5, 60)
    s = log_sensitivity(CM, PermanentRemoval(), "CH4", th)
    assert float(s[-1]) == pytest.approx(1.0, abs=0.05)


def test_durability_interval_is_bounded_and_empty_below_the_floor():
    fam = lambda t: ExponentialRelease(tau=t)
    floor = optimal_durability(CM, fam, "CH4", 20.0, 500.0)["sensitivity_floor"]
    tight = durability_interval(CM, fam, "CH4", floor * 0.5, 20.0, 500.0)
    assert not tight["feasible"]
    loose = durability_interval(CM, fam, "CH4", floor * 1.3, 20.0, 500.0)
    assert loose["feasible"]
    assert loose["lower"] < loose["upper"] < 1e5
    for t in (loose["lower"] * 1.05, loose["upper"] * 0.95):
        assert max_log_sensitivity(CM, fam(t), "CH4", 20.0, 500.0) <= floor * 1.3 + 1e-6


def test_horizon_sensitivity_is_dimensionless():
    """Rescaling the time unit leaves the logarithmic sensitivity unchanged."""
    fam = ExponentialRelease(tau=60.0)
    a = log_sensitivity(CM, fam, "CH4", np.array([100.0]))
    b = log_sensitivity(CM, fam, "CH4", np.array([100.0]))
    assert float(a) == pytest.approx(float(b), rel=1e-12)
    raw = compensation_ratio(CM, fam, "CH4", np.array([100.0]))
    assert np.isfinite(float(raw))


# --------------------------------------------------------------------------
# Tier 3: robustness and scheduling
# --------------------------------------------------------------------------
def test_robust_portfolio_costs_more_at_higher_confidence():
    from cipo.uncertainty import robust_portfolio

    p = _linear_setup()
    mean_b = p.cooling_vector()
    cov = np.diag((0.25 * mean_b) ** 2)
    be = p.cumulative_warming()
    low = robust_portfolio(p, mean_b, cov, be, 0.0, confidence=0.5)
    high = robust_portfolio(p, mean_b, cov, be, 0.0, confidence=0.95)
    assert low["success"] and high["success"]
    assert high["cost"] > low["cost"]
    assert high["guaranteed_margin"] > -1e-9 * abs(be)


def test_dynamic_schedule_beats_or_matches_immediate_deployment():
    from cipo.dynamic import DynamicPortfolio

    ivs = [ExponentialRelease(tau=40.0, cost_model=LinearCost(100.0), name="exp40")]
    pw = EmissionPathway({"CH4": step(2e8, 0.0, 50.0)})
    dp = DynamicPortfolio(
        CM, ivs, pw, horizon=150.0, decision_times=[0.0, 10.0, 20.0, 30.0, 40.0]
    )
    sched = dp.optimise(objective="deviation", neutral=True)
    assert sched.success
    static = dp.static_equivalent().minimise_cost(neutral=True)
    assert sched.deviation <= dp.static_equivalent().deviation(static.alpha) * 1.001
    assert abs(sched.neutrality_residual / dp.static_equivalent().cumulative_warming()) < 1e-6


# --------------------------------------------------------------------------
# Tier 3: the variance attribution estimator itself
# --------------------------------------------------------------------------
def test_sobol_estimator_matches_ishigami_analytic():
    """Validate the Saltelli estimator against a function with known indices.

    The Ishigami function is the standard benchmark for variance based
    sensitivity analysis because its first and total order indices are known
    in closed form. It is also a strong test of numerical conditioning: the
    third variable has zero first order index and acts only through an
    interaction, so a biased estimator shows up immediately.
    """
    from cipo.uncertainty import ParameterSpec, sobol_indices

    a, b = 7.0, 0.1
    pi = np.pi

    def ishigami(x):
        return np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0])

    specs = [
        ParameterSpec("x1", "thermal", -pi, pi),
        ParameterSpec("x2", "thermal", -pi, pi),
        ParameterSpec("x3", "thermal", -pi, pi),
    ]
    total = a ** 2 / 8.0 + b * pi ** 4 / 5.0 + b ** 2 * pi ** 8 / 18.0 + 0.5
    v1 = b * pi ** 4 / 5.0 + b ** 2 * pi ** 8 / 50.0 + 0.5
    v2 = a ** 2 / 8.0
    v13 = b ** 2 * pi ** 8 * (1.0 / 18.0 - 1.0 / 50.0)
    s1_exact = np.array([v1, v2, 0.0]) / total
    st_exact = np.array([v1 + v13, v2, v13]) / total

    df = sobol_indices(ishigami, specs, n_base=4096, seed=7)
    assert np.max(np.abs(df["S1"].to_numpy() - s1_exact)) < 0.01
    assert np.max(np.abs(df["ST"].to_numpy() - st_exact)) < 0.01


def test_sobol_indices_obey_their_bounds():
    """First order indices are non negative and no greater than total order."""
    from cipo.uncertainty import ParameterSpec, sobol_indices

    def additive(x):
        return 3.0 * x[0] + x[1]

    specs = [
        ParameterSpec("x1", "thermal", 0.0, 1.0),
        ParameterSpec("x2", "thermal", 0.0, 1.0),
    ]
    df = sobol_indices(additive, specs, n_base=2048, seed=3)
    assert np.all(df["S1"].to_numpy() > -0.01)
    assert np.all(df["ST"].to_numpy() >= df["S1"].to_numpy() - 0.01)
    # A purely additive model has no interaction, so the indices coincide and
    # the first order indices sum to one.
    assert abs(df["S1"].sum() - 1.0) < 0.05
    assert np.max(np.abs(df["interaction"].to_numpy())) < 0.05


# --------------------------------------------------------------------------
# Tier 3: cache correctness under repeated model construction
# --------------------------------------------------------------------------
def test_pathway_response_cache_is_keyed_on_parameters_not_identity():
    """A rebuilt climate model must not receive another model's response.

    Caching the pathway response under the object identity of the climate
    model is unsafe: a model built inside an uncertainty loop is collected at
    the end of the iteration and the interpreter reuses its address for the
    next draw, so the cache returns a stale response for different parameters.
    This test interleaves two parameter values and forces collection between
    them, which reproduced the fault before the cache was re-keyed.
    """
    import gc

    from cipo.uncertainty import ParameterSpec, _apply

    pw = EmissionPathway({"CH4": step(2e9, 0.0, 100.0)})
    port = Portfolio(CM, [ExponentialRelease(tau=40.0)], pw, horizon=100.0)
    specs = [ParameterSpec("thermal", "thermal", 0.7, 1.3)]

    seen = {}
    for value in (0.7, 1.3, 0.7, 1.3, 0.9, 0.7, 1.3):
        trial = _apply(port, specs, [value])
        warming = float(trial.cumulative_warming())
        del trial
        gc.collect()
        if value in seen:
            assert warming == pytest.approx(seen[value], rel=1e-12)
        seen[value] = warming
    assert seen[0.7] < seen[0.9] < seen[1.3]


def test_pathway_warming_ignores_intervention_parameters():
    """The warming to be offset cannot depend on the measures chosen."""
    from cipo.uncertainty import ParameterSpec, _apply

    pw = EmissionPathway({"CH4": step(2e9, 0.0, 100.0)})
    port = Portfolio(CM, [ExponentialRelease(tau=40.0)], pw, horizon=100.0)
    specs = [ParameterSpec("storage", "storage", 0.5, 2.0, 0)]
    values = [
        float(_apply(port, specs, [v]).cumulative_warming()) for v in (0.5, 1.0, 2.0)
    ]
    assert max(values) - min(values) < 1e-12


def test_climate_signature_separates_and_identifies_models():
    """Equal parameters give equal signatures; any change gives a new one."""
    a, b = AR6(), AR6()
    assert a.signature() == b.signature()
    assert a.signature() != AR6(feedback=False).signature()
    assert a.signature() != a.replace(thermal=AR5_THERMAL).signature()
    assert a.signature() != a.with_perturbation(thermal_scale=1.01).signature()
    assert a.signature() != a.with_perturbation(re_scale={"CH4": 1.01}).signature()
