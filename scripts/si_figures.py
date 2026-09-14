"""Supplementary figures.

These figures support the main text without repeating it. Each one shows a
diagnostic that the main figures state as a single number or assert in words.

S1  Verification of the exact algebra against numerical quadrature, and the
    behaviour of the convolution formula at and near degenerate rates. This is
    the evidence behind the stated accuracy of the closed form construction.
S2  The impulse response ingredients of the climate model, and the size of the
    climate carbon feedback, which the main text quotes as a percentage.
S3  Structural sensitivity: how the compensation ratio moves when the thermal
    response function is replaced by the two published alternatives.
S4  The horizon at which the cooling credited to a temporary store reaches its
    maximum, as a function of storage time, for each profile family.
S5  Geometry of the portfolio problem: the correlation structure of the
    intervention responses, whose condition number governs how sharply the
    composition is determined.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from common import (
    COLOURS,
    DOUBLE,
    SEQUENCE,
    SINGLE,
    configure,
    halo,
    light_grid,
    panel_label,
    save,
    spine_style,
    write_values,
)
from scenarios import FAMILIES, agriculture_measures, agriculture_pathway

from cipo import (
    AR5_THERMAL,
    AR6,
    AR6_CH6_THERMAL,
    Afforestation,
    DelayedPulse,
    ExpSum,
    ExponentialRelease,
    LinearRelease,
    PermanentRemoval,
    Portfolio,
)
from cipo.threshold import compensation_ratio


def fig_s1_algebra():
    """Accuracy of the closed form algebra, including at degenerate rates."""
    configure()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.34 * DOUBLE))

    # a: relative error of the closed form convolution against quadrature.
    cases = [
        ("distinct rates", ExpSum([(0.20, [1.0])]), ExpSum([(0.05, [2.0, 0.5])])),
        ("polynomial factors", ExpSum([(0.30, [0.0, 1.0, 2.0])]),
         ExpSum([(0.02, [1.0, 3.0])])),
        ("constant mode", ExpSum([(0.0, [1.0])]),
         ExpSum([(0.2323, [1.0]), (0.2439, [-0.5])])),
    ]
    # Restricted to the range over which the reference quadrature is itself
    # reliable. Beyond it the convolution value falls below the absolute
    # tolerance of the quadrature, so the comparison measures the reference
    # and no longer the closed form.
    u = np.geomspace(0.1, 300.0, 40)
    for (label, f, g), colour in zip(cases, SEQUENCE[1::2]):
        conv = f.conv(g)
        err = []
        for x in u:
            exact = float(conv.eval(x))
            num = quad(lambda s: f.eval(s) * g.eval(x - s), 0.0, x, limit=400)[0]
            err.append(abs(exact - num) / max(abs(num), 1e-300))
        ax_a.plot(u, np.maximum(err, 1e-18), color=colour, label=label,
                  path_effects=halo(1.9))
    ax_a.axhline(1e-13, color=COLOURS["accent"], lw=0.8, ls=(0, (3, 2)))
    ax_a.text(0.115, 1.6e-13, "stated accuracy", fontsize=5.8,
              color=COLOURS["accent"], va="bottom")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Evaluation time (yr)")
    ax_a.set_ylabel("Relative difference from quadrature")
    ax_a.set_ylim(1e-18, 1e-8)
    ax_a.legend(loc="upper left")
    light_grid(ax_a)
    panel_label(ax_a, "a")

    # b: the convolution through a resonance, where two rates coincide.
    lam = 0.1
    offsets = np.concatenate([-np.geomspace(1e-3, 1e-12, 60), [0.0],
                              np.geomspace(1e-12, 1e-3, 60)])
    vals = []
    for d in offsets:
        f = ExpSum([(lam, [1.0])])
        g = ExpSum([(lam + d, [1.0])])
        vals.append(float(f.conv(g).eval(50.0)))
    exact = 50.0 * np.exp(-lam * 50.0)
    # A symmetric log axis over twelve decades on each side crowds its own
    # tick labels, so the magnitude of the rate difference is plotted and the
    # two branches are drawn as separate series.
    mag = np.abs(offsets)
    rel = np.array(vals) / exact
    neg = offsets < 0
    pos = offsets > 0
    ax_b.plot(mag[neg], rel[neg], color=COLOURS["exp_long"], marker="o",
              ms=1.8, lw=0.7, label="rates approached from below")
    ax_b.plot(mag[pos], rel[pos], color=COLOURS["delayed"], marker="s",
              ms=1.8, lw=0.7, label="rates approached from above")
    ax_b.axhline(1.0, color=COLOURS["accent"], lw=0.8, ls=(0, (3, 2)))
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=7))
    ax_b.xaxis.set_minor_locator(plt.NullLocator())
    ax_b.set_xlabel("Magnitude of the difference between\nthe two decay rates (yr$^{-1}$)")
    ax_b.set_ylabel("Convolution, relative to the\ncoincident rate limit")
    ax_b.set_ylim(0.9985, 1.0015)
    ax_b.legend(loc="lower left")
    light_grid(ax_b)
    panel_label(ax_b, "b")

    for ax in (ax_a, ax_b):
        spine_style(ax)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "figS1_algebra")


def fig_s2_ingredients():
    """The impulse response ingredients and the size of the feedback."""
    cm, cm0 = AR6(), AR6(feedback=False)
    configure()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(DOUBLE, 0.30 * DOUBLE))

    t = np.geomspace(0.1, 2000.0, 600)
    ax_a.plot(t, cm.carbon.irf().eval(t), color=COLOURS["permanent"],
              label="CO$_2$ airborne fraction", path_effects=halo(1.9))
    ax_a.axhline(cm.carbon.permanent_fraction, color=COLOURS["accent"],
                 lw=0.8, ls=(0, (3, 2)))
    ax_a.text(0.13, cm.carbon.permanent_fraction + 0.02,
              "$a_0$ = {:.4f}".format(cm.carbon.permanent_fraction),
              fontsize=5.8, color=COLOURS["accent"])
    for name, colour in (("CH4", COLOURS["removal"]), ("N2O", COLOURS["afforest"])):
        ax_a.plot(t, cm.irf(name).eval(t), color=colour,
                  label="{} airborne fraction".format(cm.species[name].display),
                  path_effects=halo(1.9))
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Time since emission (yr)")
    ax_a.set_ylabel("Fraction remaining airborne")
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(loc="lower left")
    light_grid(ax_a, "y")
    panel_label(ax_a, "a")

    tt = np.linspace(0.0, 400.0, 2000)
    ax_b.plot(tt, cm.thermal.kernel().eval(tt), color=COLOURS["exp_long"],
              label="AR6 Ch. 7 (ECS {:.2f} K)".format(cm.thermal.ecs()),
              path_effects=halo(1.9))
    for th, colour, lab in (
        (AR6_CH6_THERMAL, COLOURS["exp_mid"], "AR6 Ch. 6"),
        (AR5_THERMAL, COLOURS["delayed"], "AR5"),
    ):
        ax_b.plot(tt, th.kernel().eval(tt), color=colour, lw=0.9,
                  ls=(0, (4, 2)),
                  label="{} (ECS {:.2f} K)".format(lab, th.ecs()))
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Time since forcing (yr)")
    ax_b.set_ylabel("Thermal response\n(K per W m$^{-2}$ per yr)")
    ax_b.legend(loc="upper right")
    light_grid(ax_b, "y")
    panel_label(ax_b, "b")

    th2 = np.geomspace(5.0, 1000.0, 300)
    for name, colour in (("CO2", COLOURS["permanent"]), ("CH4", COLOURS["removal"])):
        ratio = 100.0 * (cm.agwp(name, th2) / cm0.agwp(name, th2) - 1.0)
        ax_c.plot(th2, ratio, color=colour,
                  label=cm.species[name].display, path_effects=halo(1.9))
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Time horizon (yr)")
    ax_c.set_ylabel("Contribution of the climate\ncarbon feedback (%)")
    ax_c.legend(loc="upper left")
    light_grid(ax_c)
    panel_label(ax_c, "c")

    for ax in (ax_a, ax_b, ax_c):
        spine_style(ax)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "figS2_ingredients")
    return {
        "feedback_share_co2_100_pct": float(
            100.0 * (cm.agwp("CO2", 100.0) / cm0.agwp("CO2", 100.0) - 1.0)
        ),
        "feedback_share_ch4_100_pct": float(
            100.0 * (cm.agwp("CH4", 100.0) / cm0.agwp("CH4", 100.0) - 1.0)
        ),
    }


def fig_s3_structural():
    """How much the results move with the choice of thermal response."""
    configure()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.34 * DOUBLE))
    base = AR6()
    variants = [
        ("AR6 Chapter 7", base, COLOURS["exp_long"]),
        ("AR6 Chapter 6", base.replace(thermal=AR6_CH6_THERMAL), COLOURS["exp_mid"]),
        ("AR5", base.replace(thermal=AR5_THERMAL), COLOURS["delayed"]),
    ]
    th = np.geomspace(10.0, 1000.0, 300)
    for label, model, colour in variants:
        ax_a.plot(th, compensation_ratio(model, ExponentialRelease(tau=50.0), "CH4", th),
                  color=colour, label=label, path_effects=halo(1.9))
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Time horizon (yr)")
    ax_a.set_ylabel("Compensation ratio, CH$_4$\n(kg CO$_2$ per kg CH$_4$)")
    ax_a.legend(loc="lower left")
    light_grid(ax_a, "y")
    panel_label(ax_a, "a")

    ref = compensation_ratio(base, ExponentialRelease(tau=50.0), "CH4", th)
    for label, model, colour in variants[1:]:
        dev = 100.0 * (
            compensation_ratio(model, ExponentialRelease(tau=50.0), "CH4", th) / ref
            - 1.0
        )
        ax_b.plot(th, dev, color=colour, label=label, path_effects=halo(1.9))
    ax_b.axhline(0.0, color="black", lw=0.5)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Time horizon (yr)")
    ax_b.set_ylabel("Difference from the AR6\nChapter 7 result (%)")
    ax_b.legend(loc="upper left")
    light_grid(ax_b)
    panel_label(ax_b, "b")

    for ax in (ax_a, ax_b):
        spine_style(ax)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "figS3_structural")


def fig_s4_credit_peak():
    """Where the cooling credited to a temporary store reaches its maximum."""
    cm = AR6()
    configure()
    fig, ax = plt.subplots(figsize=(SINGLE, 0.78 * SINGLE))
    taus = np.geomspace(5.0, 1000.0, 70)
    colours = {
        "Exponential release": COLOURS["exp_mid"],
        "Delayed pulse": COLOURS["delayed"],
        "Constant-rate release": COLOURS["linear"],
        "Afforestation": COLOURS["afforest"],
    }
    grid = np.geomspace(5.0, 3.0e4, 700)
    out = {}
    for name, fam in FAMILIES.items():
        peaks = []
        for tb in taus:
            b = fam(float(tb)).cooling(cm, grid)
            peaks.append(grid[int(np.argmax(b))])
        ax.plot(taus, peaks, color=colours[name], label=name,
                path_effects=halo(1.9))
        out[name] = float(np.interp(60.0, taus, peaks))
    ax.plot(taus, taus, color=COLOURS["grid"], lw=0.7, ls=":")
    ax.text(320, 250, "credit peaks at the\nstorage time itself",
            fontsize=5.6, color="#7a7a7a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean storage time $\\bar\\tau$ (yr)")
    ax.set_ylabel("Horizon of maximum credited cooling (yr)")
    ax.legend(loc="upper left")
    light_grid(ax)
    spine_style(ax)
    panel_label(ax, "a", dx=-0.19)
    save(fig, "figS4_credit_peak")
    return out


def fig_s5_geometry():
    """Correlation structure of the intervention responses."""
    cm = AR6()
    configure()
    port = Portfolio(cm, agriculture_measures(True), agriculture_pathway(), 100.0)
    g = port.gram_matrix()
    d = np.sqrt(np.diag(g))
    corr = g / np.outer(d, d)
    fig, ax = plt.subplots(figsize=(0.62 * DOUBLE, 0.50 * DOUBLE))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    labels = [iv.display.replace(" with ", "\nwith ") for iv in port.interventions]
    ax.set_xticks(range(port.n))
    ax.set_yticks(range(port.n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.4)
    ax.set_yticklabels(labels, fontsize=5.4)
    for i in range(port.n):
        for j in range(port.n):
            ax.text(j, i, "{:.2f}".format(corr[i, j]), ha="center", va="center",
                    fontsize=5.0,
                    color="white" if corr[i, j] > 0.72 else "#202020")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Correlation of temperature responses", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.4)
    panel_label(ax, "a", dx=-0.36, dy=1.06)
    save(fig, "figS5_geometry")
    return {"caseA_gram_condition_si": port.condition_number()}


def build():
    vals = {}
    fig_s1_algebra()
    vals.update(fig_s2_ingredients())
    fig_s3_structural()
    peaks = fig_s4_credit_peak()
    for k, v in peaks.items():
        vals["credit_peak_at_60yr_" + k.replace(" ", "_").replace("-", "_")] = v
    vals.update(fig_s5_geometry())
    write_values(vals)
    return vals


if __name__ == "__main__":
    v = build()
    for k in sorted(v):
        print("  {} = {:.4g}".format(k, v[k]) if isinstance(v[k], float)
              else "  {} = {}".format(k, v[k]))
