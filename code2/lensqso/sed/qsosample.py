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

dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

DESg = spec.read_filter('DECam', 'DECam_g', index=[0, 1])
DESr = spec.read_filter('DECam', 'DECam_r', index=[0, 2])
DESi = spec.read_filter('DECam', 'DECam_i', index=[0, 3])
DESz = spec.read_filter('DECam', 'DECam_z', index=[0, 4])
DESY = spec.read_filter('DECam', 'DECam_Y', index=[0, 5])

PSg = spec.read_filter('Pan-Starrs', 'PSg')
PSr = spec.read_filter('Pan-Starrs', 'PSr')
PSi = spec.read_filter('Pan-Starrs', 'PSi')
PSz = spec.read_filter('Pan-Starrs', 'PSz')
PSy = spec.read_filter('Pan-Starrs', 'PSy')

VHSJ = spec.read_filter('VHS', 'VHS_J')
#VHSJ.set_vega_AB(0.916)
VHSH = spec.read_filter('VHS', 'VHS_H')
#VHSH.set_vega_AB(1.366)
VHSK = spec.read_filter('VHS', 'VHS_Ks')
#VHSK.set_vega_AB(1.827)

UHSJ = spec.read_filter('UHS', 'UHS_J')
#UHSJ.set_vega_AB(0.938)
UHSH = spec.read_filter('UHS', 'UHS_H')
#UHSH.set_vega_AB(1.379)
UHSK = spec.read_filter('UHS', 'UHS_K')
#UHSK.set_vega_AB(1.900)

W1 = spec.read_filter('WISE', 'W1', uflag=2)
#W1.set_vega_AB(2.699)
W2 = spec.read_filter('WISE', 'W2', uflag=2)
#W2.set_vega_AB(3.339)

filters = [PSg, PSr, PSi, PSz, PSy,\
           DESg, DESr, DESi, DESz, DESY,\
           VHSJ, VHSH, VHSK, UHSJ, UHSH, UHSK,\
           W1, W2]

def save_phots(qsoinfofile, qsospecfile, qsophotfile):
    qsoinfo = Table.read(qsoinfofile)

    print(len(qsoinfo))

    # wave, spec
    qsohdulist = fits.open(qsospecfile)

    qsoheader = qsohdulist[0].header
    print(qsospecfile)
    qsoflux_all = qsohdulist[0].data
    qsowave = np.exp(qsoheader['CRVAL1'] + np.arange(qsoheader['NAXIS1'])\
                     * qsoheader['CD1_1'])

    allinfo = []
    allphot = []

    for qidx in range(len(qsoinfo)):
        q_wavelength = qsowave
        q_flux = qsoflux_all[qidx]

        qso_spec = Spectrum(wavelength=q_wavelength, value=q_flux * 1e-17)
        thisphot = np.array([qso_spec.magnitude(filt)\
                             for filt in filters], dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0

        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                'DESg', 'DESr', 'DESi', 'DESz', 'DESy',\
                'VHSJ', 'VHSH', 'VHSK', 'UHSJ', 'UHSH', 'UHSK', 'W1', 'W2'])

    tbl.write(qsophotfile, overwrite=True)

def main():
    qsoinfo = './data/simqso_z4.5_6.5_10000deg2.fits'
    qsospec = './data/simqso_z4.5_6.5_10000deg2_spectra.fits'
    qsophot = './data/simqso_z4.5_6.5_10000deg2_phot.fits'

    save_phots(qsoinfo, qsospec, qsophot)

if __name__=='__main__':
    main()
