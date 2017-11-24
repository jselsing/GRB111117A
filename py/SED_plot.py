#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import astropy.units as u
import astropy.constants as c

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
import lineid_plot

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
params = {
   'axes.labelsize': 24,
   'font.size': 24,
   'legend.fontsize': 24,
   'xtick.labelsize': 24,
   'ytick.labelsize': 24,
   'text.usetex': False,
   'figure.figsize': [16, 16/1.618]
   }
mpl.rcParams.update(params)

def main():
    # Small script to plot SED, photometry and spectrum

    # Load data
    spectrum = np.genfromtxt("../data/spectroscopy/stitched_spectrum_bin5.dat")
    wl_spec = spectrum[:, 0] #* u.AA
    flux_spec = spectrum[:, 1] #* (u.erg/ u.s / u.cm**2 / u.AA)
    error_spec = spectrum[:, 2] #* (u.erg/(u.s * u.cm**2 * u.AA))
    f = interp1d(wl_spec, flux_spec, fill_value = np.nan)
    g = interp1d(wl_spec, error_spec, fill_value = np.nan)
    x = np.arange(min(wl_spec), max(wl_spec), np.median(np.diff(wl_spec)))# * u.AA
    flux_spec = f(x) * (u.erg/(u.s * u.cm**2 * u.AA))
    error_spec = g(x) * (u.erg/(u.s * u.cm**2 * u.AA))
    wl_spec = x * u.AA



    SED = np.genfromtxt("../data/SED.dat")
    wl_SED = SED[:, 0]
    AB_SED = SED[:, 1]
    phot = np.genfromtxt("../data/photometry.dat")
    mag = phot[:, 0]
    magmask = (mag != -99)
    mag = phot[:, 0][magmask]
    magerr = phot[:, 1][magmask]
    wl_phot = phot[:, 2][magmask]
    wl_plot_wid = phot[:, 3][magmask]/2.35

    AB = flux_spec.to(u.ABmag, u.spectral_density(wl_spec))
    flux_jan = flux_spec.to(u.Jy, u.spectral_density(wl_spec))*1e6
    err_jan = error_spec.to(u.Jy, u.spectral_density(wl_spec))*1e6




    ax1 = pl.gca()


    ax2 = ax1.twinx()
    # pl.errorbar(wl_spec, AB, yerr=[AB - AB_err_lo, AB_err_hi - AB], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3, label="X-shooter spectrum", zorder=1)
    ax1.errorbar(wl_phot, mag, xerr=wl_plot_wid, yerr=magerr, fmt=".", capsize=0, ms=20, color = "#4C72B0", label = "Photometric points", zorder=3, mec="black", mew = 1)
    # pl.errorbar(wl_spec, AB, fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3)
    ax1.plot(0, 0, color="black", linestyle="steps-mid", label="X-shooter spectrum", zorder=1, alpha=0.3)
    ax1.plot(0, 0, label="X-shooter error spectrum", linestyle="dashed", lw=1, zorder=1)
    ax1.plot(wl_SED, AB_SED, linestyle="steps-mid", lw=2.0, color = "#C44E52", label = "SED fit", zorder=2)
    # pl.axhline(0, linestyle="dashed", color = "black", lw = 0.4)

    ax1.legend(loc=2)
    # scale = np.median(AB[~np.isnan(AB)])

    # ax1 = pl.gca()
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax1.set_ylabel(r'Brightness [AB$_{mag}$]')
    ax1.set_ylim(26.1, 17.5)

    # ax2 = ax1.twinx()
    ax2.plot(wl_spec[::1], medfilt(flux_jan[::1], 1), color="black", linestyle="steps-mid", lw=1, alpha=0.3, label="X-shooter spectrum", zorder=1)
    ax2.plot(wl_spec[::1], medfilt(err_jan[::1], 31), linestyle="dashed", lw=1, zorder=1)
    mn, mx = 26.1, 17.5#ax2.get_ylim()
    print(mn)
    ax2.set_ylim(10**((23.9 - mn)/2.5), 10**((23.9 - mx)/2.5))
    ax2.semilogy()
    ax2.set_ylabel(r'F$_\nu$ [$\mu$Jy]')

    # pl.ylim(-1 * scale, 10 * scale)

    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()


    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #Overplot lines
    linelist = np.array([1216, 3728.09, 4862.68, 4960.30, 5008.24, 6564.61]) * (1 + 2.210)
    linenames = [r"$\mathrm{Ly\alpha}$", r"$\mathrm{[OII]}$", r"$\mathrm{H\beta}$", r"$\mathrm{[OIII]}$", r"$\mathrm{[OIII]}$", r"$\mathrm{H\alpha}$"]
    val = []
    for p in range(len(linelist)):
        xcoord = linelist[p]
        mask = (wl_SED> xcoord - 1) & (wl_SED< xcoord + 1)
        y_val = np.mean(AB_SED[mask])
        val.append(y_val - 0.5)
    val[3] -= 1
    lineid_plot.plot_line_ids(wl_SED, AB_SED, linelist, linenames, arrow_tip=val, ax=ax1, max_iter=0)
    # for i in ax1.lines:
    #     if '$' in i.get_label():
    #         i.set_alpha(0.3)
    a = ax1.findobj(mpl.text.Annotation)
    for i in a:
        if '$' in i.get_label():
            i.set_size(16)

    pl.semilogx()
    pl.xlim(3000, 24000)

    pos = [4000, 6000, 20000]
    ax1.set_xticks(pos, pos)
    # ax2.set_xticks(pos, pos)
    ax1.xaxis.set_minor_formatter(FormatStrFormatter('%i'))
    ax2.xaxis.set_minor_formatter(FormatStrFormatter('%i'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%i'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%i'))

    pl.tight_layout()
    pl.savefig("../figures/SEDspecphot.pdf")
    pl.show()


if __name__ == '__main__':
    main()