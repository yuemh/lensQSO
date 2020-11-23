import os, sys

import numpy as np
import matplotlib.pyplot as plt
import random

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as c
from astropy.io import fits
from astropy.table import Table, vstack, hstack

from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk,buildQsoSpectrum,save_spectra,load_spectra,restore_qso_grid
from simqso.sqmodels import BOSS_DR9_PLEpivot,get_BossDr9_model_vars
from simqso import sqmodels, hiforest

from matplotlib import font_manager, rcParams
fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
prop = font_manager.FontProperties(fname=fontpath)
rcParams['font.family'] = prop.get_name()

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import mylib.spectrum.spec_measurement as spec
from mylib.spectrum.spec_measurement import Spectrum

import lensqso.sed.lens_sed as sed

dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

def theta_E(sigma, zl, zs, cosmo=defaultcosmo):
    '''
    Parameters:
        sigma: the velocity dispersion of the deflector, in km s-1
        zl: the redshift of the deflector (lens)
        zs: the redshift of the source
        cosmo: the cosmology model (astropy.cosmology.FlatLambdaCDM class)

    Returns:
        the Einstein radius, in arcsecond
    '''
    Ds = cosmo.angular_diameter_distance(zs)
    Dl = cosmo.angular_diameter_distance(zl)
    Dls = cosmo.angular_diameter_distance(zl, zs)

    theta = Dls / Ds * 2 * np.pi * (sigma/3e5)** 2 * 206265
    return theta

def randomized_galaxy_catalog(tbl, width=0.05, nrand=10):

    cols=['ID', 'mStar', 'redshift', 'Re_circ', 'sersic_n', 'SIGMA']

    tbl_list = []

    for index in range(len(tbl)):
        redshift = tbl['redshift'][index]
        d_redshift = (1+redshift) * width
        randomized_redshift = []
        while 1:
            randz = np.random.normal(redshift, d_redshift)
            if randz>0.1:
                randomized_redshift.append(randz)
                if len(randomized_redshift)==nrand:
                    break
        row_thisgal = tbl[cols][index]
        tbl_thisgal = vstack([row_thisgal] * nrand)
        tbl_thisgal['orig_index'] = [index] * nrand
        tbl_thisgal['new_redshift'] = randomized_redshift

        tbl_list.append(tbl_thisgal)

    alltbl = vstack(tbl_list)
    alltbl.write(dir_data + \
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.randz.fits',\
            overwrite=True)

def bad_ids():
    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_sigmafile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'
    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data

    jaguar_info = Table.read(jaguar_info_sigmafile)
    mask = (jaguar_info['redshift']<2)&(jaguar_info['SIGMA']>150)

    jaguar_info_masked = jaguar_info[mask]
    jaguar_flux_masked = jaguar_flux[mask]

    wave = jaguar_wave

    bad_id = []

    for index in range(len(jaguar_info_masked)):
        ID = jaguar_info_masked['ID'][index]
        flux = jaguar_flux[jaguar_info['ID']==ID]
        redshift = jaguar_info_masked['redshift'][index]

        print('ID: %d, redshift: %.2f'%(ID, redshift))
        plt.plot(jaguar_wave, flux[0])
        plt.xlim([0, 40000])
        plt.show()
        flag = input('Good (G) / Bad (B)?')
        if flag in 'Bb':
            bad_id.append(ID)

        np.savetxt('./bad_id_s150_z2.txt', np.transpose(bad_id), fmt='%d')

def modify_sample(randz_file, output):
    tbl = Table.read(randz_file)
    Dl_old = defaultcosmo.luminosity_distance(tbl['redshift'])
    Dl_new = defaultcosmo.luminosity_distance(tbl['new_redshift'])

    factor = np.array((Dl_old / Dl_new)**2, dtype=float)

    tbl['factor'] = factor
    tbl.write(output, overwrite=True)

def test():
    tbl = Table.read(dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.randz2.fits')
    plt.hist(tbl['new_redshift'], bins=np.arange(0, 2, 0.05))
    plt.show()

def main():
    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_sigmafile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'
    jaguar_info_randzfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.randz.fits'
    jaguar_info_randzfile2 = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.randz2.fits'

    tbl = Table.read(jaguar_info_sigmafile)
    tbl = tbl[(tbl['redshift']<2)&(tbl['SIGMA']>150)]
    print(len(tbl))
    print(tbl['ID'])
#    randomized_galaxy_catalog(tbl, width=0.05)
#    bad_ids()
#    modify_sample(jaguar_info_randzfile, jaguar_info_randzfile2)
#    test()

if __name__=='__main__':
    main()

