#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Imports
import numpy as np

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
from scipy import interpolate
import pandas as pd

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
    burst_table = pd.read_csv("../data/Berger 2014 - Table 2.csv")
    name, L_B = burst_table["GRB"].values, burst_table["LB"].values
    #Flux complete from D'Avanzo 2014 #130515A not in Berger sample, 090426c, 111117Ac excluded
    flux_complete = ["051221A", "060313", "061201", "070714B", "080123", "080503", "080905A", "090426c", "090510", "090515", "100117A", "100625A", "101219A", "111117Ac", "130515A", "130603B"]

    # Filter for names
    complete_L_B = [ii for ii, kk in enumerate(name) if kk in flux_complete]
    print(len(complete_L_B))
    complete = False
    if complete == True:
        L_B = L_B[complete_L_B]
    # print(np.nanmean(L_B))
    # exit()
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
    print(lnredshifts, ln)
    print(lnredshifts/ln)
    print()
    #Brighter than:
    print("Brighter than:")
    print(lnbrigter, lnredshifts)
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




