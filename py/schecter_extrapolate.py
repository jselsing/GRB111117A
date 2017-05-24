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

    # z = [0.30, 0.50, 0.70, 0.90, 1.10, 1.30]
    # z_arr = np.arange(0, 3, 0.01)
    # MB = [-20.36, -20.72, -21.15, -21.21, -21.38, -21.86]
    # MB_up = [0.13, 0.05, 0.07, 0.00, 0.04, 0.07]
    # MB_low = [0.11, 0.07, 0.07, 0.03, 0.05, 0.08]
    # f = interpolate.interp1d(z, MB, kind="slinear", fill_value="extrapolate")
    # pl.errorbar(z, MB, yerr=[MB_up, MB_low])
    # pl.plot(z_arr, f(z_arr))
    # print(f(2.211))
    # pl.show()

    # LB from Berger 2014
    L_B = np.array([0.1, 1.0, 0.3, np.nan, np.nan, 0.1, 0.1, 0.6, np.nan, 0.1, 1.4, 1.9, 1.2, np.nan, np.nan, np.nan, np.nan, np.nan, 0.3, 1.0, np.nan, 0.5, np.nan, np.nan, 0.6, 0.8, 1.0, 5.0, np.nan, np.nan, 1.6, 0.6, 0.9, 0.4, 1.0, 1.2, 1.0, 0.2, 1.3])
    print(len(L_B))
    print(26/39)
    print(len(L_B[L_B > 1.17]))
    print(np.sum((~np.isnan(L_B)).astype("int")))
    print(1 - 7/26)
    L_B_nonan = L_B[~np.isnan(L_B)]

    MB_star = -21.83#f(2.211)
    L_B_star = 10 ** (-1*(-22 - MB_star)/2.5)
    # seaborn.distplot(L_B_nonan, rug=True)
    # pl.show()
    print(L_B_star)
    print(L_B_star*10**(-0.4))
    print(len(L_B[L_B > L_B_star*10**(-0.4)]))
    print(1 - 18/26)
    print(np.percentile(L_B[~np.isnan(L_B)], [14, 50, 85]))

if __name__ == '__main__':
    main()




