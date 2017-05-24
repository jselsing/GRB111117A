#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')
import numpy as np
from scipy import stats
import matplotlib as mpl

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7.281, 4.5]
   }
mpl.rcParams.update(params)

def propagate_uncertainties(func, parameters, parameter_errors, samples):
    """
    For a given function, evaluates the function and computes the 1 \sigma errors.
    """

    for i, k in enumerate(parameter_errors):
        parameter_errors[i] = k * (np.random.randn(samples))

    for i, k in enumerate(parameters):
        parameters[i] = k + parameter_errors[i]

    func_eval = func (*parameters)
    # print(func_eval)
    # print('Discarded iterations: ', len(func_eval[np.isnan(func_eval)]))
    if len(func_eval[np.isnan(func_eval)])/samples  > 0.3:
        print('High number of dicards. Potentially biased result.')
    return np.mean(func_eval[~np.isnan(func_eval)]), np.std(func_eval[~np.isnan(func_eval)])

def fraction(x, y):
    return np.log10(x / y)


def main():
    """
    # Script to produce N_H vs. z
    """
    # Read in SDSS line-fluxes
    sdss = fits.open('/Users/jselsing/Work/spectral_libraries/sdss_catalogs/portsmouth_emlinekin_full-DR12.fits')
    # mask = np.logical_and((sdss[1].data.field('Z') >= 0.5),(sdss[1].data.field('Z') <= 2.0))


    mask = (sdss[1].data.field('Z') <= 0.1)

    Hb_sdss = sdss[1].data.field('FLUX')[:, 15]
    OIII_sdss = sdss[1].data.field('FLUX')[:, 17]

    Ha_sdss = sdss[1].data.field('FLUX')[:, 24]
    NII_sdss = sdss[1].data.field('FLUX')[:, 25]

    x = np.log10(NII_sdss[mask]/Ha_sdss[mask])
    y = np.log10(OIII_sdss[mask]/Hb_sdss[mask])


    # Plot values from Arcodia et al. 2016
    g = sns.JointGrid(x=x, y=y, xlim=(-2, 1), ylim=(-1.5, 1.5), space=0)
    color = sns.color_palette()[0]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = sns.blend_palette(colors, as_cmap=True)

    g = g.plot_joint(pl.hexbin, cmap=cmap, bins='log', extent=[-2, 1, -1.5, 1.5], alpha=0.3, edgecolors="NAAone")

    color = sns.color_palette()[1]


    nii = 4.5e-18
    niie = 9.9e-18
    ha = 1.1e-16
    hae = 1.2e-17

    niiha, niihae = propagate_uncertainties(fraction, [nii, ha], [niie, hae], 1000000)

    hb = 1.3e-17
    hbe = 3.5e-18
    oiii = 4.3e-17
    oiiie = 9.9e-18

    hboiii, hboiiie = propagate_uncertainties(fraction, [oiii, hb], [oiiie, hbe], 1000000)

    g.x = np.log10(3*niie / ha)
    g.y = np.log10(oiii / (3*hbe))
    g = g.plot_joint(pl.errorbar, yerr=0.2, xerr=0.2, xuplims=True, uplims=True, lw=1.5, color=sns.color_palette()[2], label="GRB111117A")


    g.x = niiha
    g.y = hboiii
    g = g.plot_joint(pl.scatter, facecolors='none', marker = "*", s = 150, lw=1.5, color=sns.color_palette()[2], label="GRB111117A")
    ax = pl.gca()
    from matplotlib.patches import Ellipse
    for j in np.arange(1, 4):
        ax.add_artist(Ellipse((g.x, g.y), j*niihae, j*hboiiie, fill=False, linestyle='dashed', lw = 2, alpha = 1.0/(2.0*j) ))




    ax = pl.gca()
    # ax.axhline(0.5, color="black", linestyle="dashed", alpha=0.5)
    # ax.annotate(r"$\beta_{OX} = 0.5$", (19.6, 0.45))
    g.set_axis_labels(r"log([N II]$\lambda$6584/H$\alpha$)", r"log([O III]$\lambda$5007/H$\beta$)")
    pl.tight_layout()


    # Save figure for tex
    pl.legend()
    pl.savefig("../figures/BPT.pdf", dpi="figure")
    pl.show()

if __name__ == '__main__':
    main()