#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Imports
import numpy as np

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
from scipy import interpolate

def main():

    # GRB111117A L_B
    MB_star = -21.83 # From Marchesini et al. 2007
    L_B_star = 10 ** (-1*(-22 - MB_star)/2.5)

    #L_B for GRB111117A
    print("L_B:")
    print(L_B_star)
    print()
    # L_B for a galaxy, 1 mag fainter
    dM = 0.5
    print("L_B_limit:")
    print(L_B_star*10**(-0.4*(dM)))
    print()

    # LB from Berger 2014
    L_B = np.array([0.1, 1.0, 0.3, np.nan, np.nan, 0.1, 0.1, 0.6, np.nan, 0.1, 1.4, 1.9, 1.2, np.nan, np.nan, np.nan, np.nan, np.nan, 0.3, 1.0, np.nan, 0.5, np.nan, np.nan, 0.6, 0.8, 1.0, 5.0, np.nan, np.nan, 1.6, 0.6, 0.9, 0.4, 1.0, 1.2, 1.0, 0.2, 1.3])
    #Exclude nans
    L_B_nonan = L_B[~np.isnan(L_B)]
    # Length of list
    ln = len(L_B)
    # Nr with redshifts
    lnredshifts = len(L_B_nonan)
    # Brigter than GRB111117A
    lnbrigter = len(L_B[L_B > L_B_star])
    #Redshift completeness
    print("Redshift completeness:")
    print(lnredshifts/ln)
    print()
    #Brighter than:
    print("Brighter than:")
    print(1 - lnbrigter/lnredshifts)
    print()
    # print(len(L_B[L_B > 1.17]))
    # print(np.sum((~np.isnan(L_B)).astype("int")))
    # print(1 - 7/26)


    MB_star = -21.83#f(2.211)
    L_B_star = 10 ** (-1*(-22 - MB_star)/2.5)


    lnunobs = len(L_B[L_B > L_B_star*10**(-0.4*(dM))])
    # Fraction unobservable
    print("Fraction unobservable:")
    print(lnredshifts - lnunobs, lnredshifts)
    print(1 - lnunobs/lnredshifts)


    # print(np.percentile(L_B[~np.isnan(L_B)], [14, 50, 85]))

if __name__ == '__main__':
    main()




