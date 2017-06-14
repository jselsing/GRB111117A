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
   'axes.labelsize': 16,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'text.usetex': False,
   'figure.figsize': [7.281, 4.5]
   }
mpl.rcParams.update(params)
from astropy.cosmology import Planck15 as cosmo



def iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    h = 2 * iqr(a) / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


def filt_nan(input_array, fill_value=np.nan):
    """
    Small functino to remove values in array, not convertible to float. Return the filtered array
    """
    holder = []
    for ii in input_array:
        try:
            holder.append(float(ii))
        except ValueError:
            holder.append(fill_value)
    return np.array(holder)


def main():
    """
    # Script to produce Eiso vs. z
    """

    Eiso_111117A = 3.38 * (cosmo.luminosity_distance(2.211)**2/cosmo.luminosity_distance(1.3)**2)*(2.3 / 3.211)*1e51


    # Read in long GRBs from Turpin et al. 2015, https://arxiv.org/abs/1503.02760
    burst_table = pd.read_csv("../data/Comparison sample - Long Eiso.csv")
    name, z_long, Eiso_long = burst_table["GRB"].values, burst_table["z"].values, burst_table["Eiso"].values

    Eiso_long = np.log10(10*filt_nan(Eiso_long, fill_value=np.nan)*1e51)
    z_long = filt_nan(z_long, fill_value=np.nan)

    # Read in short GRBs from D'Avanzo et al. 2014
    burst_table = pd.read_csv("../data/Comparison sample - Short.csv")
    name, z_short, Eiso_short = burst_table["GRB"].values, burst_table["z"].values, burst_table["Eiso"].values

    # Remove GRB090426
    idx = np.where(name == "090426")[0]
    name, z_short, Eiso_short = np.delete(name, idx), np.delete(z_short, idx), np.delete(Eiso_short, idx)

    Eiso_short = np.log10(filt_nan(Eiso_short, fill_value=np.nan)*1e51)




    # Plot values from Turpin et al. 2015
    g = sns.JointGrid(x=z_long, y=Eiso_long, xlim=(0, 4), ylim=(49, 55), space=0)
    color = sns.color_palette()[2]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = sns.blend_palette(colors, as_cmap=True)
    g = g.plot_marginals(sns.distplot, hist=False, color=color)
    g = g.plot_joint(sns.kdeplot, cmap=cmap, label="Turpin et al. 2015", zorder=1)

    color = sns.color_palette()[1]
    g.x = np.array([2.211])
    g.y = np.array([np.log10(Eiso_111117A)])
    g = g.plot_joint(pl.scatter, facecolors='none', marker = "*", s = 150, lw=1.5, color=sns.color_palette()[1], label="GRB111117A")
    g = g.plot_marginals(sns.rugplot, color=color, height = 1.0, lw = 4.0)

    # Plot values D'Avanzo et al. 2014
    color = sns.color_palette()[0]
    g.x = z_short
    g.y = Eiso_short
    g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0})
    g = g.plot_joint(pl.scatter, color=color, label="D'Avanzo et al. 2014")


    g.x = -1
    g.y = 1
    g = g.plot_joint(pl.plot, color=sns.color_palette()[2], label="Turpin et al. 2015")





    ax = pl.gca()
    # ax.axhline(0.5, color="black", linestyle="dashed", alpha=0.5)
    # ax.annotate(r"$\beta_{OX} = 0.5$", (19.6, 0.45))
    g.set_axis_labels(r"z", r"log(E$_{iso}$) [erg]")
    pl.tight_layout()



    # Save figure for tex
    pl.legend(loc=4)
    pl.savefig("../figures/Eiso_z.pdf", dpi="figure")
    pl.show()

if __name__ == '__main__':
    main()