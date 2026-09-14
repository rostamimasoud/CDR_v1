"""Figure 6: the compensation ratio atlas.

For every combination of storage timescale and accounting horizon, how much
temporary carbon dioxide removal is required to offset a unit pulse of a given
gas. Each panel is one gas, and the eight panels run in ascending order of
perturbation lifetime, so the progression across the figure is the progression
from a forcer that decays within decades to one that persists for tens of
thousands of years.

The filled contours carry the compensation ratio on a logarithmic scale. Two
overlays carry the result that the ratio alone does not show: the dashed line
is the locus where the ratio is stationary in the horizon, and the marker is
the durability at which the ratio is least sensitive to the horizon over the
policy window. Reading the panels in order shows the locus moving out of the
plotted region as the target gas becomes longer lived, which is the graphical
form of the statement that a carbon store offsets a short lived forcer well
and a long lived one poorly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from common import (
    COLOURS,
    DOUBLE,
    configure,
    panel_label,
    save,
    save_table,
    write_values,
)

from cipo import AR6, ExponentialRelease
from cipo.threshold import compensation_ratio, log_sensitivity, optimal_durability

#: Panels in ascending order of perturbation lifetime.
SPECIES = [
    ("HFC32", "HFC-32", 5.4),
    ("CH4", "CH$_4$ (biogenic)", 11.8),
    ("CH4_fossil", "CH$_4$ (fossil)", 11.8),
    ("HFC134a", "HFC-134a", 14.0),
    ("CFC11", "CFC-11", 52.0),
    ("N2O", "N$_2$O", 109.0),
    ("SF6", "SF$_6$", 3200.0),
    ("PFC14", "PFC-14", 50000.0),
]

TAU = np.geomspace(2.0, 2000.0, 190)
TH = np.geomspace(10.0, 1000.0, 190)
TH_MIN, TH_MAX = 20.0, 500.0


def build():
    cm = AR6()
    configure()
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 0.52 * DOUBLE), sharex=True,
                             sharey=True)
    letters = "abcdefgh"
    rows = []
    optima = {}

    tau_grid, th_grid = np.meshgrid(TAU, TH, indexing="ij")
    # The cooling delivered by a store depends on its timescale and the
    # horizon alone, so it is computed once and reused for every gas.
    cooling = np.empty_like(tau_grid)
    for i, t in enumerate(TAU):
        cooling[i, :] = ExponentialRelease(tau=float(t)).cooling(cm, TH)

    for k, (name, label, lifetime) in enumerate(SPECIES):
        ax = axes.flat[k]
        alpha = cm.iagtp(name, th_grid) / cooling

        vmin = float(np.nanpercentile(alpha, 1))
        vmax = float(np.nanpercentile(alpha, 99))
        levels = np.geomspace(vmin, vmax, 24)
        cf = ax.contourf(
            TAU, TH, alpha.T, levels=levels, norm=LogNorm(vmin, vmax),
            cmap="viridis", extend="both",
        )
        for c in cf.collections:
            c.set_rasterized(True)
        # Decade contours give the eye a sense of gradient. They carry no
        # inline labels: the colourbar states the magnitude, and labels here
        # would collide with the panel annotation.
        ax.contour(
            TAU, TH, alpha.T, levels=_decade_levels(vmin, vmax),
            colors="white", linewidths=0.45, alpha=0.7,
        )

        # Locus where the compensation ratio is stationary in the horizon.
        sens = np.empty_like(alpha)
        for i, t in enumerate(TAU):
            sens[i, :] = _signed_log_sensitivity(cm, float(t), name, TH)
        ax.contour(
            TAU, TH, sens.T, levels=[0.0], colors=COLOURS["accent"],
            linewidths=0.9, linestyles='dashed',
        )

        # Durability least sensitive to the horizon over the policy window.
        opt = optimal_durability(
            cm, lambda x: ExponentialRelease(tau=x), name, TH_MIN, TH_MAX,
            search_range=(2.0, 2000.0), n_scan=40, n_refine=30,
        )
        optima[name] = opt
        t_opt = opt["optimal_storage_time"]
        interior = TAU[0] * 1.02 < t_opt < TAU[-1] * 0.98
        opt["interior"] = bool(interior)
        if interior:
            ax.plot(
                [t_opt], [np.sqrt(TH_MIN * TH_MAX)], marker="o", ms=3.4,
                mfc="white", mec=COLOURS["accent"], mew=0.9, zorder=6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(TAU[0], TAU[-1])
        ax.set_ylim(TH[0], TH[-1])
        ax.tick_params(labelsize=5.8)
        ax.text(
            0.045, 0.915,
            "{}\n$\\tau_x$ = {:g} yr".format(label, lifetime),
            transform=ax.transAxes, fontsize=5.9, va="top", ha="left",
            color="white",
            bbox=dict(boxstyle="round,pad=0.24", fc="#101010", ec="none", alpha=0.78),
        )
        panel_label(ax, letters[k], dx=-0.13, dy=1.19)

        cb = fig.colorbar(cf, ax=ax, pad=0.025, fraction=0.048, aspect=17)
        cb.ax.tick_params(labelsize=5.2, length=1.6, pad=1.2)
        cb.locator = LogLocator(base=10.0, subs=(1.0,), numticks=5)
        cb.formatter = LogFormatterMathtext()
        cb.update_ticks()
        cb.outline.set_linewidth(0.4)

        for th in (20.0, 100.0, 500.0):
            row = {"species": name, "lifetime_yr": lifetime, "horizon_yr": th}
            for tau in (25.0, 55.0, 120.0, 350.0):
                row["alpha_tau{:g}".format(tau)] = float(
                    compensation_ratio(cm, ExponentialRelease(tau=tau), name, th)
                )
            rows.append(row)

    for ax in axes[1, :]:
        ax.set_xlabel("Mean storage time $\\bar\\tau$ (yr)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time horizon $T_{\\rm H}$ (yr)")

    fig.subplots_adjust(hspace=0.36, wspace=0.46)
    save(fig, "fig6_atlas")

    df = pd.DataFrame(rows)
    save_table(df, "table6_atlas")
    vals = {
        "atlas_n_species": len(SPECIES),
        "atlas_lifetime_min_yr": min(s[2] for s in SPECIES),
        "atlas_lifetime_max_yr": max(s[2] for s in SPECIES),
    }
    for name, opt in optima.items():
        vals["atlas_optimum_" + name] = opt["optimal_storage_time"]
        vals["atlas_floor_" + name] = opt["sensitivity_floor"]
        vals["atlas_interior_" + name] = str(opt.get("interior", False))
    write_values(vals)
    return df, optima


def _signed_log_sensitivity(cm, tau, species, horizons):
    """Logarithmic horizon sensitivity, keeping its sign.

    The zero contour of this field is the locus on which the compensation
    ratio is stationary in the horizon, which is the boundary between the
    region where a longer horizon demands more compensation and the region
    where it demands less.
    """
    iv = ExponentialRelease(tau=tau)
    th = np.asarray(horizons, dtype=float)
    b = iv.cooling(cm, th)
    deriv = (
        cm.agtp(species, th) * b + cm.iagtp(species, th) * iv.response(cm, th)
    ) / (b * b)
    ratio = cm.iagtp(species, th) / b
    return th * deriv / ratio


def _decade_levels(vmin, vmax):
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil(np.log10(vmax)))
    out = [10.0 ** e for e in range(lo, hi + 1)]
    return [v for v in out if vmin < v < vmax]


def _fmt_level(v):
    if v >= 1000:
        return "{:.0e}".format(v).replace("e+0", "e")
    if v >= 1:
        return "{:.0f}".format(v)
    return "{:.2g}".format(v)


if __name__ == "__main__":
    df, optima = build()
    print("  panels: {} species".format(len(SPECIES)))
    for name, opt in optima.items():
        print(
            "    {:<11s} optimum {:8.1f} yr   floor {:.3f}".format(
                name, opt["optimal_storage_time"], opt["sensitivity_floor"]
            )
        )
