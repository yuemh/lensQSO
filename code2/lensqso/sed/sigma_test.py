import os, sys

sys.path.append('/Users/minghao/Research/Projects/lensQSO/code2')

import numpy as np
import matplotlib.pyplot as plt
import random

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as c
from astropy.io import fits
from astropy.table import Table, vstack, hstack

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import mylib.spectrum.spec_measurement as spec
from mylib.spectrum.spec_measurement import Spectrum

def MStar_sigma(mstar, re, n):
    '''
    Bezanson et al. 2011
    '''
    Kv = 73.32 / (10.465 + (n-0.94)**2) + 0.954
    Ks = 0.557 * Kv

    Ms = np.power(10, mstar)*c.M_sun
    re_phys = re * u.kpc
    sigma = np.sqrt(c.G * Ms / Ks / re_phys)
    sigma_num = sigma.to('km/s').value

def FP_sigma():
    dummy = 1

def dispersion_function(tbl):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    zlist = np.arange(0.3, 1.5, 0.3)
    vlist = np.arange(2.0, 3.0, 0.1)

    sigmafunc = []

    area_factor = (11./60./180.*np.pi)**2/4/np.pi*10

    for zmin in zlist:
        zmax = zmin + 0.3
        comov_vol = cosmo.comoving_volume(zmax)\
                - cosmo.comoving_volume(zmin)

        comov_vol = comov_vol.value
#        sigmafunc_thisz = []
        for vmin in vlist:
            vmax = vmin + 0.1

#            print(zmin, vmin)
            tblselect = tbl[(tbl['SIGMA']>10**vmin*2)\
                            &(tbl['SIGMA']<10**vmax*2)\
                           &(tbl['redshift']>zmin)\
                           &(tbl['redshift']<zmax)]

            sigmafunc.append(len(tblselect)/(comov_vol*area_factor)/0.1)

    sigmafunc = np.log10(sigmafunc).reshape(len(zlist), len(vlist))
    print(sigmafunc)
    return sigmafunc

def main():
    tbl = Table.read('./data/JAGUAR_galaxy_phot.fits')
    dispersion_function(tbl)

if __name__=='__main__':
    main()

