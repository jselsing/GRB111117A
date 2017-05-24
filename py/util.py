#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from astropy.stats import sigma_clip
from scipy import signal
import matplotlib.pyplot as pl
from scipy.special import wofz, erf

__all__ = ["correct_for_dust", "bin_image", "avg", "convert_air_to_vacuum", "convert_vacuum_to_air", "bin_spectrum"]




def avg(flux, error, mask=None, axis=2, weight=False, weight_map=None):

    """Calculate the weighted average with errors
    ----------
    flux : array-like
        Values to take average of
    error : array-like
        Errors associated with values, assumed to be standard deviations.
    mask : array-like
        Array of bools, where true means a masked value.
    axis : int, default 0
        axis argument passed to numpy

    Returns
    -------
    average, error : tuple

    Notes
    -----
    """
    try:
        if not mask:
            mask = np.zeros_like(flux).astype("bool")
    except:
        pass
        # print("All values are masked... Returning nan")
        # if np.sum(mask.astype("int")) == 0:
        #     return np.nan, np.nan, np.nan


    # Normalize to avoid numerical issues in flux-calibrated data
    norm = abs(np.median(flux[flux > 0]))
    if norm == np.nan or norm == np.inf or norm == 0:
        print("Nomalization factor in avg has got a bad value. It's "+str(norm)+" ... Replacing with 1")

    flux_func = flux.copy() / norm
    error_func = error.copy() / norm

    # Calculate average based on supplied weight map
    if weight_map is not None:
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        average = np.sum(weight_map * flux_func, axis = axis)
        variance = np.sum(weight_map ** 2 * error_func ** 2.0, axis = axis)

    # Inverse variance weighted average
    elif weight:
        ma_flux_func = np.ma.array(flux_func, mask=mask)
        ma_error_func = np.ma.array(error_func, mask=mask)
        w = 1.0 / (ma_error_func ** 2.0)
        average = np.ma.sum(ma_flux_func * w, axis = axis) / np.ma.sum(w, axis = axis)
        variance = 1. / np.ma.sum(w, axis = axis)
        if not isinstance(average, float):
            # average[average.mask] = np.nan
            average = average.data
            # variance[variance.mask] = np.nan
            variance = variance.data

    # Normal average
    elif not weight:
        # Number of pixels in the mean
        n = np.sum((~mask).astype("int"), axis = axis)
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        # mean
        average = (1 / n) * np.sum(flux_func, axis = axis)
        # probagate errors
        variance = (1 / n**2) * np.sum(error_func ** 2.0, axis = axis)

    mask = (np.sum((~mask).astype("int"), axis = axis) == 0).astype("int")
    return (average * norm, np.sqrt(variance)*norm, mask)


def correct_for_dust(wavelength, ra, dec):
    """Query IRSA dust map for E(B-V) value and returns reddening array
    ----------
    wavelength : numpy array-like
        Wavelength values for which to return reddening
    ra : float
        Right Ascencion in degrees
    dec : float
        Declination in degrees

    Returns
    -------
    reddening : numpy array

    Notes
    -----
    For info on the dust maps, see http://irsa.ipac.caltech.edu/applications/DUST/
    """

    from astroquery.irsa_dust import IrsaDust
    import astropy.coordinates as coord
    import astropy.units as u
    C = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
    dust_image = IrsaDust.get_images(C, radius=2 *u.deg, image_type='ebv', timeout=60)[0]
    ebv = np.mean(dust_image[0].data[40:42, 40:42])
    r_v = 3.1
    av =  r_v * ebv
    from specutils.extinction import reddening
    return reddening(wavelength* u.angstrom, av, r_v=r_v, model='ccm89'), ebv


def bin_spectrum(wl, flux, error, mask, binh, weight=False):

    """Bin low S/N 1D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    print("Binning image by a factor: "+str(binh))
    if binh == 1:
        return wl, flux, error, mask

    # Outsize
    size = flux.shape[0]
    outsize = int(np.round(size/binh))

    # Containers
    wl_out = np.zeros((outsize))
    res = np.zeros((outsize))
    reserr = np.zeros((outsize))
    resbp = np.zeros((outsize))

    for ii in np.arange(0, size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = int((ii + binh)/binh) - 1
        # Construct weighted average and weighted std along binning axis
        res[h_index], reserr[h_index], resbp[h_index] = avg(flux[ii:ii + binh], error[ii:ii + binh], mask = mask[ii:ii + binh], axis=0, weight=weight)
        wl_out[h_index] = np.median(wl[ii:ii + binh], axis=0)

    return wl_out[1:-1], res[1:-1], reserr[1:-1], resbp[1:-1]


def bin_image(flux, error, mask, binh, weight=False):

    """Bin low S/N 2D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    print("Binning image by a factor: "+str(binh))
    if binh == 1:
        return flux, error

    # Outsize
    v_size, h_size = flux.shape
    outsizeh = int(h_size/binh)

    # Containers
    res = np.zeros((v_size, outsizeh))
    reserr = np.zeros((v_size, outsizeh))

    flux_tmp = flux.copy()
    for ii in np.arange(0, h_size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = int((ii + binh)/binh - 1)

        # Sigma clip before binning to remove noisy pixels with bad error estimate.
        clip_mask = sigma_clip(flux[:, ii:ii + binh], axis=1)

        # Combine masks
        mask_comb = mask[:, ii:ii + binh].astype("bool") | clip_mask.mask

        # Construct weighted average and weighted std along binning axis
        res[:, h_index], reserr[:, h_index], __ = avg(flux_tmp[:, ii:ii + binh], error[:, ii:ii + binh], mask=mask_comb, axis=1, weight=weight)

    return res, reserr


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


def convert_vacuum_to_air(vac_wave) :
    # convert vacuum to air
    # taken from http://www.sdss.org/dr7/products/spectra/vacwavelength.html

    air_wave = vac_wave / (1.0 + 2.735182e-4 + 131.4182 / vac_wave**2 + 2.76249e8 / vac_wave**4)
    return air_wave
