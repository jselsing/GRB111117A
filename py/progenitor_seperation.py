#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Imports
from astropy import units as u
from astropy import constants as c
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')
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

def calc_sep(age, m1, m2):
    """
    Calculation of progenitor seperation.
    age in Myr, progenitor masses in Msun
    returns seperation in units of Rsun
    """
    # Convert to seconds
    t0 = (age*1e6 * u.yr).to(u.s)
    # Convert to kg
    M_ns1 = (m1 * u.Msun).to(u.kg)
    M_ns2 = (m2 * u.Msun).to(u.kg)

    # Total system mass
    M = M_ns1 + M_ns2
    # Reduced mass
    mu = M_ns1*M_ns2/M

    # Calculate initial progenitor seperation from eq 29 from Postnov and Yungelson 2014
    a0 = ((t0 * ((c.G**3)/(c.c**5)) * ((256 * M**2 * mu)/5))**(1/4))

    return a0.value/u.Rsun.to(u.m) * u.Rsun


def main():

    # Calculate initial progenitor seperation from eq 29 from Postnov and Yungelson 2014
    # Time since big bang
    t_constraint = 2970 #Myr
    # t_constraint = 1000 #Myr
    # Neutron star masses
    m1 = 1.4 # Msun
    m2 = 1.4 # Msun
    # Get seperation
    ratio = np.arange(0.5, 2, 0.01)
    masses = np.arange(0.9, 2.1, 0.01)
    xx, yy = np.meshgrid(masses, masses[::-1])
    a0 = calc_sep(t_constraint, xx, yy)

    color = sns.color_palette()[2]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = sns.blend_palette(colors, as_cmap=True)
    fig, ax = pl.subplots()
    im = ax.imshow(a0, extent=[min(masses), max(masses), min(masses), max(masses)], cmap="viridis_r")
    cbar = pl.colorbar(im)
    cbar.set_label(r'a$_0$ [R$_\odot$]')


    from matplotlib.patches import Ellipse
    ax.scatter(1.33, 1.33, s=10, color = "white")
    ax.add_artist(Ellipse((1.33, 1.33), 1.33 - 1.21, 1.43 - 1.33, fill=False, linestyle='dashed', lw = 1, alpha = 1.0, color = "white"))
    ax.add_artist(Ellipse((1.33, 1.33), 1.33 - 1.10, 1.55 - 1.33, fill=False, linestyle='dashed', lw = 1, alpha = 0.7, color = "white"))



    ax.set_xlabel(r"M$_{\mathrm{NS}}$ [M$_\odot$]")
    ax.set_ylabel(r"M$_{\mathrm{NS}}$ [M$_\odot$]")
    pl.tight_layout()
    pl.savefig("../figures/prog_sep.pdf")
    pl.show()

    a01 = calc_sep(t_constraint, 1.33, 1.33)
    print(a01)
    # 1-sigma
    a0 = calc_sep(t_constraint, 1.21, 1.21)
    print(a01 - a0)
    a0 = calc_sep(t_constraint, 1.43, 1.43)
    print(a0 - a01)
    # 2-sigma
    a0 = calc_sep(t_constraint, 1.10, 1.10)
    print(a01 - a0)
    a0 = calc_sep(t_constraint, 1.55, 1.55)
    print(a0 - a01)
if __name__ == '__main__':
    main()




