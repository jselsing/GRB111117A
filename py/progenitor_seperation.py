#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Imports
from astropy import units as u
from astropy import constants as c


def calc_sep(age, m1, m2):
    """
    Small calculation of progenitor seperation.
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
    # t_constraint = 500 #Myr
    # Neutron star masses
    m1 = 1.4 # Msun
    m2 = 1.4 # Msun
    # Get seperation
    a0 = calc_sep(t_constraint, m1, m2)

    print(a0)


if __name__ == '__main__':
    main()




