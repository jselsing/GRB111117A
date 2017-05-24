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
    # Script to produce N_H vs. z
    """
    # Read in long GRBs
    burst_table = pd.read_csv("../data/Comparison sample - Long NH.csv")
    name, z_long, NH_long = burst_table["GRB"].values, burst_table["z"].values, burst_table["Nnew(z)"].values

    NH_long_dec = np.log10(1e21*filt_nan(NH_long, fill_value=np.nan))
    z_long_dec = filt_nan(z_long, fill_value=np.nan)
    print(len(z_long_dec), len(z_long_dec[np.isnan(z_long_dec)]))

    for ii, kk in enumerate(z_long):
        print(kk, NH_long[ii], NH_long_dec[ii])


    # Get lower limits
    NH_long_lowlim = np.log10(1e21*np.array([(kk, float(ii[1:])) for kk, ii in enumerate(NH_long[np.isnan(NH_long_dec)]) if ii[0] == ">"]))
    z_long_lowlim, NH_long_lowlim = z_long[np.isnan(NH_long_dec)][((10**NH_long_lowlim[:, 0])/1e21).astype("int")], NH_long_lowlim[:, 1]
    z_long_lowlim = filt_nan(z_long_lowlim, fill_value=np.nan)
    print(z_long_lowlim)


    # Get upper limits
    NH_long_uplim = np.array([(kk, float(ii[1:])) for kk, ii in enumerate(NH_long[np.isnan(NH_long_dec)]) if ii[0] == "<"])
    z_long_uplim, NH_long_uplim = z_long[np.isnan(NH_long_dec)][NH_long_uplim[:, 0].astype("int")], np.log10(1e21*NH_long_uplim[:, 1])


    # Read in short GRBs
    burst_table = pd.read_csv("../data/Comparison sample - Short.csv")
    name, z_short, NH_short = burst_table["GRB"].values, burst_table["z"].values, burst_table["NH(z)"].values

    NH_short_dec = np.log10(1e21*filt_nan(NH_short, fill_value=np.nan))
    NH_short_lim = np.log10(1e21*np.array([float(ii[1:]) for ii in NH_short[np.isnan(NH_short_dec)]]))

    z_short_dec = filt_nan(z_short, fill_value=np.nan)
    z_short_lim = z_short[np.isnan(NH_short_dec)]

    # Plot values from Arcodia et al. 2016
    g = sns.JointGrid(x=z_long_dec, y=NH_long_dec, xlim=(0, 4), ylim = (20.5, 23.5),  space=0)
    color = sns.color_palette()[2]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = sns.blend_palette(colors, as_cmap=True)
    g = g.plot_marginals(sns.distplot, hist=False, color=color)
    g = g.plot_joint(sns.kdeplot, cmap=cmap, label="Arcodia et al. 2016", zorder=1)
    # g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0})
    # g = g.plot_joint(pl.scatter, color=color, label="Arcodia et al. 2016")
    g.x = z_long_uplim
    g.y = NH_long_uplim
    g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0, "alpha": 0.5})
    g = g.plot_joint(pl.scatter, color=color, marker=u'$\u2193$', s=100, lw=1.0)

    g.x = z_long_lowlim
    g.y = NH_long_lowlim
    g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0, "alpha": 0.5})
    g = g.plot_joint(pl.scatter, color=color, marker=u'$\u2191$', s=100, lw=1.0)




    color = sns.color_palette()[1]
    g.x = np.array([2.211])
    g.y = np.array([np.log10(2.4e22)])
    # g.y = np.array([np.log10(2.4e22+2.4e22)])
    gyupperr = np.log10(2.4e22)/np.log10(2.4e22) # np.log10(2.4e22) - np.log10(1.6e22)
    gylowerr = np.log10(1.6e22)/np.log10(2.4e22) #  np.log10(4.8e22) - np.log10(2.4e22)
    g = g.plot_joint(pl.scatter, marker = "*", s = 150, color=sns.color_palette()[1], label="GRB111117A")

    g = g.plot_marginals(sns.rugplot, color=color, height = 1.0, lw = 4.0)
    g.x = np.array([2.211, 2.211])
    g.y = np.array([np.log10(2.4e22 - 1.6e22), np.log10(2.4e22 + 2.4e22)])
    g = g.plot_joint(pl.plot, color=sns.color_palette()[1])

    # Plot values D'Avanzo et al. 2014
    color = sns.color_palette()[0]
    g.x = z_short_dec
    g.y = NH_short_dec
    print(g.y)
    g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0})
    g = g.plot_joint(pl.scatter, color=color, label="D'Avanzo et al. 2014")
    g.x = z_short_lim
    g.y = NH_short_lim
    g = g.plot_marginals(sns.distplot, hist=False, rug=True, kde=False, color=color, rug_kws={"height": 0.7, "lw": 2.0, "alpha": 0.5})
    g = g.plot_joint(pl.scatter, color=color, marker=u'$\u2193$', s=100, lw=1.0)

    g.x = -1
    g.y = 1
    g = g.plot_joint(pl.plot, color=sns.color_palette()[2], label="Arcodia et al. 2016")





    ax = pl.gca()
    # ax.axhline(0.5, color="black", linestyle="dashed", alpha=0.5)
    # ax.annotate(r"$\beta_{OX} = 0.5$", (19.6, 0.45))
    g.set_axis_labels(r"z", r"log(N$_H$) [cm$^{-2}$]")
    pl.tight_layout()



    # Save figure for tex
    pl.legend(loc=1)
    pl.savefig("../figures/NH_z.pdf", dpi="figure")
    pl.show()

if __name__ == '__main__':
    main()