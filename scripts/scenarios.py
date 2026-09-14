"""Case study definitions: emission pathways, measures and cost assumptions.

Two scenarios are used throughout, matching work packages 4a and 4b of the
research plan.

**Scenario A, agricultural sector.** A national agricultural inventory
dominated by methane from enteric fermentation, rice cultivation and manure
management, together with nitrous oxide from managed soils. Emissions decline
modestly under continued productivity growth. This is the case where temporary
storage has a physical claim to relevance, because the target gases are short
lived.

**Scenario B, national net zero pathway.** Economy wide emissions falling to a
residual by 2050 and held there, with carbon dioxide, methane and nitrous
oxide all present. This is the harder case, because a permanent constituent
must be offset.

Emission magnitudes are of the order of a large agricultural economy and are
stated as round numbers: the framework is scale free in the deployment
variables, so the portfolio shares and the durability results do not depend on
the absolute scale, and only the reported absolute costs do.

Cost assumptions are central estimates from the assessed ranges in IPCC AR6
Working Group III Chapter 12 and in Chiquier et al. (2022), expressed in United
States dollars per tonne of carbon dioxide at peak storage, or per tonne of
methane for direct methane removal. Where a range is wide, the midpoint of the
assessed range is used and the sensitivity of the conclusions to that choice
is quantified by the Sobol analysis.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from cipo import (
    Afforestation,
    DelayedPulse,
    EmissionPathway,
    ExponentialRelease,
    LinearCost,
    PermanentRemoval,
    PowerCost,
    SpeciesRemoval,
)
from cipo.pathways import piecewise_linear

MT = 1.0e9  # kg per megatonne
GT = 1.0e12  # kg per gigatonne

# Cost of one tonne of carbon dioxide at peak storage, United States dollars.
UNIT_COSTS = {
    "soil_carbon": 45.0,
    "biochar": 110.0,
    "wood_products": 60.0,
    "afforestation": 30.0,
    "beccs": 150.0,
    "daccs": 400.0,
    "methane_removal": 2400.0,  # per tonne of methane removed
}

# Cumulative deployment ceilings over the century, gigatonnes of carbon
# dioxide at peak storage, reflecting land, feedstock and geological limits.
# Methane removal is capped in gigatonnes of methane.
#
# The scale is set by the problem itself. Offsetting the cumulative warming of
# Scenario A with permanent removal alone requires about 20 gigatonnes of
# carbon dioxide over the century, that is about 0.2 gigatonnes per year, so
# ceilings are assigned in that range. Their sum admits about 1.5 times the
# required cooling, which leaves the composition genuinely determined by cost
# and durability instead of by attainability alone.
CAPS_AGRICULTURE = {
    "soil_carbon": 12.0,
    "biochar": 6.0,
    "wood_products": 4.0,
    "afforestation": 10.0,
    "beccs": 5.0,
    "daccs": 3.0,
    "methane_removal": 0.15,  # gigatonnes of methane
}


# --------------------------------------------------------------------------
# Emission pathways
# --------------------------------------------------------------------------
def agriculture_pathway() -> EmissionPathway:
    """Scenario A: methane and nitrous oxide from agriculture, 2025 to 2125.

    Methane falls from 8.0 to 6.4 megatonnes per year by 2075 and is then
    held; nitrous oxide falls from 0.30 to 0.24 megatonnes per year on the
    same profile. Emissions continue beyond the policy horizon, which is the
    situation an inventory authority actually faces.
    """
    years = [0.0, 25.0, 50.0, 100.0, 200.0]
    ch4 = [8.0 * MT, 7.4 * MT, 6.8 * MT, 6.4 * MT, 6.4 * MT]
    n2o = [0.30 * MT, 0.28 * MT, 0.26 * MT, 0.24 * MT, 0.24 * MT]
    return EmissionPathway(
        {
            "CH4": piecewise_linear(years, ch4),
            "N2O": piecewise_linear(years, n2o),
        },
        "Agricultural sector",
    )


def net_zero_pathway() -> EmissionPathway:
    """Scenario B: economy wide emissions to a residual by 2050.

    Carbon dioxide falls from 400 to 40 megatonnes per year over 25 years and
    is then held at that residual. Methane and nitrous oxide fall to
    agricultural residuals on the same schedule. The retained carbon dioxide
    residual is what makes this scenario qualitatively harder for temporary
    measures.
    """
    years = [0.0, 25.0, 50.0, 100.0, 200.0]
    co2 = [400.0 * MT, 120.0 * MT, 40.0 * MT, 40.0 * MT, 40.0 * MT]
    ch4 = [5.0 * MT, 3.2 * MT, 2.6 * MT, 2.6 * MT, 2.6 * MT]
    n2o = [0.22 * MT, 0.17 * MT, 0.15 * MT, 0.15 * MT, 0.15 * MT]
    return EmissionPathway(
        {
            "CO2": piecewise_linear(years, co2),
            "CH4": piecewise_linear(years, ch4),
            "N2O": piecewise_linear(years, n2o),
        },
        "National net zero pathway",
    )


# --------------------------------------------------------------------------
# Intervention sets
# --------------------------------------------------------------------------
def _cost(key: str, convex: bool, cap: float, gamma: float = 1.6):
    """Cost model in dollars per kilogram, linear or strictly convex.

    A convex model represents the rising marginal cost of scaling a finite
    resource. The reference scale is set to half the deployment ceiling, so
    the unit cost is recovered at a mid range deployment and the assessed
    central estimate keeps its meaning.
    """
    per_kg = UNIT_COSTS[key] / 1000.0
    if not convex:
        return LinearCost(per_kg)
    return PowerCost(per_kg, gamma, 0.5 * cap * GT)


def agriculture_measures(convex: bool = True) -> List:
    """The portfolio of measures available in Scenario A."""
    c = CAPS_AGRICULTURE
    return [
        ExponentialRelease(
            tau=25.0,
            cost_model=_cost("soil_carbon", convex, c["soil_carbon"]),
            max_scale=c["soil_carbon"] * GT,
            name="soil_carbon",
            label="Soil carbon",
            side_effects={"land": 1.0},
        ),
        ExponentialRelease(
            tau=350.0,
            cost_model=_cost("biochar", convex, c["biochar"]),
            max_scale=c["biochar"] * GT,
            name="biochar",
            label="Biochar",
            side_effects={"land": 0.4},
        ),
        DelayedPulse(
            tau=55.0,
            cost_model=_cost("wood_products", convex, c["wood_products"]),
            max_scale=c["wood_products"] * GT,
            name="wood_products",
            label="Wood products",
            side_effects={"land": 0.7},
        ),
        Afforestation(
            tau_growth=28.0,
            tau_disturbance=160.0,
            cost_model=_cost("afforestation", convex, c["afforestation"]),
            max_scale=c["afforestation"] * GT,
            name="afforestation",
            label="Afforestation",
            side_effects={"land": 2.5},
        ),
        PermanentRemoval(
            cost_model=_cost("beccs", convex, c["beccs"]),
            max_scale=c["beccs"] * GT,
            name="beccs",
            label="Bioenergy with capture and storage",
            side_effects={"land": 1.8},
        ),
        SpeciesRemoval(
            species="CH4",
            efficiency=1.0,
            co2_penalty=0.0,
            cost_model=_cost("methane_removal", convex, c["methane_removal"]),
            max_scale=c["methane_removal"] * GT,
            name="methane_removal",
            label="Methane removal",
            side_effects={"land": 0.0},
        ),
    ]


def net_zero_measures(convex: bool = True) -> List:
    """Scenario B measures: as Scenario A plus direct air capture."""
    measures = agriculture_measures(convex)
    measures.insert(
        5,
        PermanentRemoval(
            cost_model=_cost("daccs", convex, CAPS_AGRICULTURE["daccs"]),
            max_scale=CAPS_AGRICULTURE["daccs"] * GT,
            name="daccs",
            label="Direct air capture and storage",
            side_effects={"land": 0.1},
        ),
    )
    return measures


# --------------------------------------------------------------------------
# Profile families indexed by mean storage time
# --------------------------------------------------------------------------
def _afforestation_for(tau_bar: float, tau_growth: float = 28.0):
    """Afforestation with a disturbance time giving the required mean storage."""
    from scipy.optimize import brentq

    def gap(td: float) -> float:
        return (
            Afforestation(tau_growth=tau_growth, tau_disturbance=td).mean_storage_time
            - tau_bar
        )

    try:
        td = brentq(gap, 1e-3, 1e9, xtol=1e-9, rtol=1e-12)
    except ValueError:
        td = np.nan
    return Afforestation(tau_growth=tau_growth, tau_disturbance=td)


#: Each family is parameterised by its mean storage time, which is the
#: invariant that makes shapes comparable.
FAMILIES = {
    "Exponential release": lambda tb: ExponentialRelease(tau=tb),
    "Delayed pulse": lambda tb: DelayedPulse(tau=tb),
    "Constant-rate release": lambda tb: LinearReleaseByMean(tb),
    "Afforestation": _afforestation_for,
}


def LinearReleaseByMean(tau_bar: float):
    """Constant rate release whose mean storage time is ``tau_bar``."""
    from cipo import LinearRelease

    return LinearRelease(release_time=2.0 * tau_bar)


REPRESENTATIVE = {
    "Soil carbon": 25.0,
    "Wood products": 55.0,
    "Afforestation": 120.0,
    "Biochar": 350.0,
}
