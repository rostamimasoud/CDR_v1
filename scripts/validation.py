"""Validation protocol: the three tiers of work package 3.

Tier 1 compares the calibrated model against the published IPCC AR6 emission
metrics. Tier 2 reproduces the single measure, single gas compensation ratios
of He et al. (2026), which is the closest published antecedent. Tier 3 checks
the optimality conditions of the portfolio problem against closed form
solutions.

The comparison with He et al. requires care over normalisation. Their
exponential release profile is written

    F(t) = -delta(t) + (3/d) exp(-3 t / d),

so the parameter ``d`` they report is three times the mean storage time. The
comparison below converts their parameter before evaluating, and is therefore
a test of the physics and not of a labelling convention.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from common import save_table, write_values

from cipo import (
    AR5_THERMAL,
    AR6,
    AR6_CH6_THERMAL,
    AR6_REFERENCE,
    EmissionPathway,
    ExponentialRelease,
    LinearCost,
    PermanentRemoval,
    Portfolio,
    PowerCost,
    SpeciesRemoval,
)
from cipo.pathways import pulse
from cipo.threshold import compensation_ratio

#: Compensation ratios reported by He et al. (2026) for temporary removal
#: offsetting a methane pulse over a hundred year horizon, keyed by their
#: release parameter in years.
HE_REFERENCE = {20.0: 498.0, 100.0: 101.0}


def tier1_metrics(cm) -> pd.DataFrame:
    rows = []
    for horizon, key in ((20.0, "AGWP_CO2_20"), (100.0, "AGWP_CO2_100")):
        got = float(cm.agwp("CO2", horizon))
        ref = AR6_REFERENCE[key]
        rows.append(
            {
                "quantity": "AGWP CO2",
                "horizon_yr": horizon,
                "this_study": got,
                "published": ref,
                "difference_pct": 100.0 * (got - ref) / ref,
                "source": "IPCC AR6 Table 7.SM.7",
            }
        )
    for key, horizon, fn in (
        ("GWP20", 20.0, cm.gwp),
        ("GWP100", 100.0, cm.gwp),
        ("GTP100", 100.0, cm.gtp),
    ):
        for species, ref in sorted(AR6_REFERENCE[key].items()):
            got = float(fn(species, horizon))
            rows.append(
                {
                    "quantity": "{} {}".format(key[:3], species),
                    "horizon_yr": horizon,
                    "this_study": got,
                    "published": ref,
                    "difference_pct": 100.0 * (got - ref) / ref,
                    "source": "IPCC AR6 Table 7.15",
                }
            )
    return pd.DataFrame(rows)


def tier2_he_benchmark(cm) -> pd.DataFrame:
    rows = []
    for decay, ref in sorted(HE_REFERENCE.items()):
        tau_bar = decay / 3.0
        got = float(
            compensation_ratio(cm, ExponentialRelease(tau=tau_bar), "CH4", 100.0)
        )
        rows.append(
            {
                "release_parameter_yr": decay,
                "mean_storage_time_yr": tau_bar,
                "this_study": got,
                "published": ref,
                "difference_pct": 100.0 * (got - ref) / ref,
                "source": "He et al. (2026), Nature 654, 391-397",
            }
        )
    return pd.DataFrame(rows)


def tier3_optimality(cm) -> pd.DataFrame:
    rows = []
    pw = EmissionPathway({"CH4": pulse(1.0)})

    # Permanent removal against a pulse must return the integrated GTP.
    for species in ("CH4", "N2O"):
        for th in (20.0, 100.0):
            p = Portfolio(
                cm, [PermanentRemoval(cost_model=LinearCost(1.0))],
                EmissionPathway({species: pulse(1.0)}), horizon=th,
            )
            got = float(p.minimise_cost(neutral=True).alpha[0])
            ref = float(cm.igtp(species, th))
            rows.append(
                {
                    "check": "permanent removal recovers iGTP ({})".format(species),
                    "horizon_yr": th,
                    "computed": got,
                    "reference": ref,
                    "relative_error": abs(got - ref) / abs(ref),
                }
            )

    # Species matched removal is exactly one for one at every horizon.
    for th in (20.0, 100.0, 500.0):
        p = Portfolio(
            cm, [SpeciesRemoval("CH4", cost_model=LinearCost(1.0))], pw, horizon=th
        )
        got = float(p.minimise_cost(neutral=True).alpha[0])
        rows.append(
            {
                "check": "species matched removal needs unit deployment",
                "horizon_yr": th,
                "computed": got,
                "reference": 1.0,
                "relative_error": abs(got - 1.0),
            }
        )

    # Convex costs: numerical optimum against the closed form solution.
    ivs = [
        ExponentialRelease(
            tau=t, cost_model=PowerCost(c, 2.0, 50.0), name="e{:g}".format(t)
        )
        for t, c in ((20.0, 50.0), (50.0, 100.0), (100.0, 200.0))
    ]
    p = Portfolio(cm, ivs, pw, horizon=100.0)
    num = p.minimise_cost(neutral=True)
    ana = p.analytic_power_cost_solution()
    rows.append(
        {
            "check": "convex cost optimum matches closed form",
            "horizon_yr": 100.0,
            "computed": float(num.cost),
            "reference": float(ana.cost),
            "relative_error": abs(num.cost - ana.cost) / ana.cost,
        }
    )
    rows.append(
        {
            "check": "marginal cost per unit cooling equalised",
            "horizon_yr": 100.0,
            "computed": float(p.kkt_report(ana.alpha)["relative_spread"]),
            "reference": 0.0,
            "relative_error": float(p.kkt_report(ana.alpha)["relative_spread"]),
        }
    )

    # Corrected Proposition 1.
    cm_nf = AR6(feedback=False)
    for species in ("CH4", "N2O"):
        sp = cm_nf.species[species]
        got = cm_nf.iagtp_limit(species)
        ref = sp.re_per_kg * sp.tau * cm_nf.thermal.equilibrium_sensitivity
        rows.append(
            {
                "check": "iAGTP limit equals RE tau sum(c) ({})".format(species),
                "horizon_yr": np.inf,
                "computed": got,
                "reference": ref,
                "relative_error": abs(got - ref) / abs(ref),
            }
        )
    return pd.DataFrame(rows)


def tier4_structural(cm) -> pd.DataFrame:
    """Sensitivity of the headline metrics to the thermal response chosen."""
    rows = []
    for name, thermal in (
        ("AR6 Chapter 7 (ECS 3.0 K)", None),
        ("AR6 Chapter 6 (ECS 3.5 K)", AR6_CH6_THERMAL),
        ("AR5 (Boucher and Reddy 2008)", AR5_THERMAL),
    ):
        model = cm if thermal is None else cm.replace(thermal=thermal)
        iv = ExponentialRelease(tau=50.0)
        rows.append(
            {
                "thermal_response": name,
                "equilibrium_sensitivity": model.thermal.equilibrium_sensitivity,
                "GWP100_CH4": float(model.gwp("CH4", 100.0)),
                "GTP100_CH4": float(model.gtp("CH4", 100.0)),
                "alpha_exp50_CH4_TH100": float(
                    compensation_ratio(model, iv, "CH4", 100.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build():
    cm = AR6()
    t1 = tier1_metrics(cm)
    t2 = tier2_he_benchmark(cm)
    t3 = tier3_optimality(cm)
    t4 = tier4_structural(cm)
    save_table(t1, "table_val1_ar6_metrics")
    save_table(t2, "table_val2_he_benchmark")
    save_table(t3, "table_val3_optimality")
    save_table(t4, "table_val4_structural")

    gwp = t1[t1["quantity"].str.startswith("GWP")]["difference_pct"].abs()
    vals = {
        "val_agwp_co2_100_error_pct": float(
            t1.loc[t1["quantity"].eq("AGWP CO2") & t1["horizon_yr"].eq(100.0),
                   "difference_pct"].abs().iloc[0]
        ),
        "val_gwp_max_error_pct": float(gwp.max()),
        "val_gwp_mean_error_pct": float(gwp.mean()),
        "val_he_max_error_pct": float(t2["difference_pct"].abs().max()),
        "val_he_mean_error_pct": float(t2["difference_pct"].abs().mean()),
        "val_optimality_max_relative_error": float(t3["relative_error"].max()),
        "val_alpha_spread_across_thermal_pct": float(
            100.0
            * (
                t4["alpha_exp50_CH4_TH100"].max() / t4["alpha_exp50_CH4_TH100"].min()
                - 1.0
            )
        ),
    }
    write_values(vals)
    return t1, t2, t3, t4, vals


if __name__ == "__main__":
    t1, t2, t3, t4, vals = build()
    pd.set_option("display.width", 200)
    print(t1.to_string(index=False, float_format=lambda x: "%.5g" % x))
    print()
    print(t2.to_string(index=False, float_format=lambda x: "%.5g" % x))
    print()
    print("  largest optimality error: {:.2e}".format(
        vals["val_optimality_max_relative_error"]
    ))
    print("  GWP agreement: mean {:.2f}%, worst {:.2f}%".format(
        vals["val_gwp_mean_error_pct"], vals["val_gwp_max_error_pct"]
    ))
