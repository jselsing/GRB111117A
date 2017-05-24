#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')
import numpy as np
from scipy import stats, interpolate
import matplotlib as mpl
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants as c

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7.281, 4.5]
   }

def convert_air_to_vacuum(air_wave) :
    # convert air to vacuum. Based onhttp://idlastro.gsfc.nasa.gov/ftp/pro/astro/airtovac.pro
    # taken from https://github.com/desihub/specex/blob/master/python/specex_air_to_vacuum.py

    sigma2 = (1e4/air_wave)**2
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) +  1.67917e-3/( 57.362 - sigma2)
    vacuum_wave = air_wave*fact

    # comparison with http://www.sdss.org/dr7/products/spectra/vacwavelength.html
    # where : AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    # air_wave=numpy.array([4861.363,4958.911,5006.843,6548.05,6562.801,6583.45,6716.44,6730.82])
    # expected_vacuum_wave=numpy.array([4862.721,4960.295,5008.239,6549.86,6564.614,6585.27,6718.29,6732.68])
    return vacuum_wave






def main():
    """
    # Script to get lineflux
    """
    # Get extraction
    data = np.genfromtxt("../data/NIRext.dat", dtype=None)
    # data = np.genfromtxt("../data/NIROB4skysubstdext.dat", dtype=None)
    # print(np.std([18, 31, 38, 20]))
    # exit()
    wl = data[:, 1]
    flux = data[:, 2]
    error = data[:, 3]
    error[error > 1e-15] = np.median(error)
    # Load synthetic sky
    sky_model = fits.open("../data/NIRskytable.fits")
    wl_sky = 10*(sky_model[1].data.field('lam')) # In nm
    # Convolve to observed grid
    f = interpolate.interp1d(wl_sky, convolve(sky_model[1].data.field('flux'), Gaussian1DKernel(stddev=3)), bounds_error=False, fill_value=np.nan)
    # f = interpolate.interp1d(wl_sky, sky_model[1].data.field('flux'), bounds_error=False, fill_value=np.nan)
    flux_sky = f(wl)
    # flux[(wl > 21055) & (wl < 21068)] = np.nan
    # flux[(wl > 21098) & (wl < 21102)] = np.nan
    # flux[(wl > 21109) & (wl < 21113)] = np.nan

    m = (flux_sky < 10000) & ~np.isnan(flux)
    g = interpolate.interp1d(wl[m], flux[m], bounds_error=False, fill_value=np.nan)
    flux = g(wl)


    mask = (wl > convert_air_to_vacuum(15613) - 12) & (wl < convert_air_to_vacuum(15613) + 12)
    # mask = (wl > 21020) & (wl < 21150)
    F_hb = np.trapz(flux[mask])
    print("Total %0.1e" %F_hb)
    F_hb_err = np.sqrt(np.trapz((error**2.)[mask]))
    print("Total %0.1e +- %0.1e" % (F_hb, F_hb_err))


    # flux[flux_sky > 10000] = np.nan
    pl.plot(wl, flux, label = "Spectrum")
    pl.plot(wl[mask], flux[mask], label = "Integration limits")
    pl.plot(wl, flux_sky*1e-22, label = "Sky spectrum")
    pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3, label=r"f$_{H\beta}$ = %0.1e +- %0.1e" % (F_hb, F_hb_err))
    pl.xlim(15550, 15650)
    pl.ylim(-0.5e-17, 3e-17)
    # pl.show()

    # Save figure for tex
    pl.legend()
    pl.savefig("../figures/hb_flux.pdf", dpi="figure")
    pl.show()

if __name__ == '__main__':
    main()