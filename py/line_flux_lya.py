#!/usr/bin/env python
# -*- coding: utf-8 -*-




# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')


from astropy.io import fits
import pandas as pd
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
    data = np.genfromtxt("../data/spectroscopy/UVBext_lya.dat", dtype=None)

    wl = data[:, 1]
    wl_mask = (wl > 3850) & (wl < 3960)

    flux = data[:, 2]
    error = data[:, 3]

    wl, flux, error = wl[wl_mask], flux[wl_mask], error[wl_mask]

    error[error > 1e-15] = np.median(error)

    mask = (wl > convert_air_to_vacuum(3907) - 12) & (wl < convert_air_to_vacuum(3907) + 12)

    continuum_fit = np.polynomial.chebyshev.chebfit(wl[~mask], flux[~mask], deg=2)
    continuum = np.polynomial.chebyshev.chebval(wl, continuum_fit)

    pl.plot(wl, continuum - continuum)

    flux = flux - continuum

    F_lya = np.trapz(flux[mask], x=wl[mask])
    print("Total %0.1e" %F_lya)

    F_lya_err = np.sqrt(np.trapz((error**2.)[mask], x=wl[mask]))


    dL = cosmo.luminosity_distance(2.211).to(u.cm).value
    L_lya = F_lya * 4 * np.pi * dL**2
    L_lya_err = F_lya_err * 4 * np.pi * dL**2
    print(L_lya)
    SFR = 9.1e-43*L_lya / 1.64
    SFR_err = 9.1e-43*L_lya_err / 1.64
    print(SFR, SFR_err)


    # flux[flux_sky > 10000] = np.nan
    pl.plot(wl, flux, label = "Spectrum")
    pl.plot(wl[mask], flux[mask], label = "Integration limits")

    pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3, label=r"f$_{[L\alpha]}$ = %0.1e +- %0.1e" % (F_lya, F_lya_err))
    pl.errorbar(1, 1, yerr=1, fmt=".k", capsize=0, elinewidth=0.5, ms=3, label="SFR = " +str(np.around(SFR, 0)) + " +-" + str(np.around(SFR_err, 0)))
    pl.xlim(3850, 3960)
    pl.ylim(-0.5e-17, 1e-17)


    # Save figure for tex
    pl.legend()
    pl.savefig("../figures/lya_flux.pdf", dpi="figure")
    pl.show()

if __name__ == '__main__':
    main()