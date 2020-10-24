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

max_ID_SF_dir = {}

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

    return sigma_num

def add_sigma_JAGUAR(jaguar_info_masterfile, output):
    jaguar_info = Table.read(jaguar_info_masterfile)

    sigma_list = []
    selected_index_list = []
    for index in range(len(jaguar_info)):
        z_lens = jaguar_info['redshift'][index]

        sigma = MStar_sigma(jaguar_info['mStar'][index],\
                            jaguar_info['Re_circ'][index],\
                            jaguar_info['sersic_n'][index])
        sigma_list.append(sigma)

    jaguar_info['SIGMA'] = np.array(sigma_list)
    jaguar_info.write(output)

def concat_table(SFfile, Qfile, mastfile):
    SFtbl = Table.read(SFfile)
    Qtbl = Table.read(Qfile)

    columns = ['ID', 'RA', 'DEC', 'redshift', 'mStar', 'SIGMA']
    SFtbl = SFtbl[columns][:]
    Qtbl = Qtbl[columns][:]

    masttbl = vstack([SFtbl, Qtbl])

    masttbl = masttbl[(masttbl['SIGMA']>100)\
                      &(masttbl['redshift']<2)]
    masttbl.write(mastfile, overwrite=True)
    print(len(masttbl))

def find_specfile(realization, redshift, galID):
    dir_info = './data/r%d/info/'%realization
    dir_spec = './data/r%d/spec/'%realization

    info_SF = dir_info+'/JADES_SF_mock_r%d_v1.2_sigma.fits.gz'%realization
    info_Q = dir_info+'/JADES_Q_mock_r%d_v1.2_sigma.fits.gz'%realization
    info_master = dir_info+'/JADES_master_mock_r%d_v1.2_sigma.fits.gz'%realization

    # SF or Q?
    try:
        max_SF_ID = max_ID_SF_dir[realization]
    except KeyError:
        tbl_info_SF = Table.read(info_SF)
        max_SF_ID = len(tbl_info_SF)
        max_ID_SF_dir[realization] = max_SF_ID

    if galID>max_SF_ID:
        galtype = 'Q'
    elif galID<=max_SF_ID:
        galtype='SF'
    else:
        raise IndexError('Galaxy ID must be an integer.')

    # z range
    if galtype=='Q':
        zrange_str = ''
    elif galtype=='SF' and redshift<1:
        zrange_str = '_z_0p2_1'
    elif galtype=='SF' and (redshift>1 and redshift<1.5):
        zrange_str = '_z_1_1p5'
    elif galtype=='SF' and (redshift>1.5 and redshift<2):
        zrange_str = '_z_1p5_2'
    else:
        raise ValueError('Redshift out of range')

    # the jaguar filename in the form of:
    specfile = 'JADES_%s_mock_r%d_v1.2_spec_5A_30um%s.fits'%(galtype, realization, zrange_str)

    return specfile

def readspec(realization, redshift, galID):
    specfile = find_specfile(realization, redshift, galID)
    specdir = os.path.abspath('./data/r%d/spec'%realization)

    jaguar_data_hdulist = fits.open(specdir + '/' + specfile)

    # get the spec
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data
    jaguar_info = jaguar_data_hdulist[3].data

    # get the index of the galaxy
    gindex = np.where(jaguar_info['ID']==galID)

    # get the flux of the galaxy
    gflux = jaguar_flux[gindex] / (1 + redshift)
    gwave = jaguar_wave * (1 + redshift)

    return (gflux[0], gwave)

def save_phots(realization, photfile):
    dir_info = './data/r%d/info/'%realization
    dir_spec = './data/r%d/spec/'%realization

    info_SF = dir_info+'/JADES_SF_mock_r%d_v1.2_sigma.fits.gz'%realization
    info_Q = dir_info+'/JADES_Q_mock_r%d_v1.2_sigma.fits.gz'%realization
    info_master = dir_info+'/JADES_master_mock_r%d_v1.2_sigma.fits.gz'%realization

    if not os.path.exists(info_master):
        concat_table(info_SF, info_Q, info_master)

    tbl_master = Table.read(info_master)

    # get photometric points

    allinfo = []
    allphot = []

    print(len(tbl_master))
    for index in range(len(tbl_master)):
        print(index)
        # read galaxy spec
        ID = tbl_master['ID'][index]
        redshift = tbl_master['redshift'][index]

        flux, wave = readspec(realization, redshift, ID)
        gal_spec = Spectrum(wavelength=wave, value=flux)
        thisphot = np.array([gal_spec.magnitude(filt)\
                             for filt in filters], dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0

        print(thisphot)
        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                'DESg', 'DESr', 'DESi', 'DESz', 'DESy',\
                'VHSJ', 'VHSH', 'VHSK', 'UHSJ', 'UHSH', 'UHSK', 'W1', 'W2'])

    tbl.write(photfile, overwrite=True)

def combine_all_reals():
    tbllist = []

    for r in range(1,11, 1):
        print(r)
        dir_info = './data/r%d/info/'%r

        file_info = dir_info + '/JADES_master_mock_r%d_v1.2_sigma.fits.gz'%r
        file_phot = dir_info + '/phot%d.fits'%r

        tbl_info = Table.read(file_info)
        tbl_phot = Table.read(file_phot)

        tbl_thisr = hstack([tbl_info, tbl_phot])
        tbl_thisr['realization'] = r
        tbllist.append(tbl_thisr)

    tbl = vstack(tbllist)
    tbl.write('./data/JAGUAR_galaxy_phot.fits')

def main():
    realization = 10
    dir_info = './data/r%d/info/'%realization
    dir_spec = './data/r%d/spec/'%realization

#    add_sigma_JAGUAR(\
#            dir_info+'/JADES_SF_mock_r%d_v1.2.fits.gz'%realization,\
#            dir_info+'/JADES_SF_mock_r%d_v1.2_sigma.fits.gz'%realization)
#    add_sigma_JAGUAR(\
#            dir_info+'/JADES_Q_mock_r%d_v1.2.fits.gz'%realization,\
#            dir_info+'/JADES_Q_mock_r%d_v1.2_sigma.fits.gz'%realization)

#    concat_table(\
#        dir_info+'/JADES_SF_mock_r%d_v1.2_sigma.fits.gz'%realization,\
#        dir_info+'/JADES_Q_mock_r%d_v1.2_sigma.fits.gz'%realization,\
#        dir_info+'/JADES_master_mock_r%d_v1.2_sigma.fits.gz'%realization)

#    mastfile = dir_info+'/JADES_master_mock_r1_v1.2_sigma.fits.gz'
#    index = 100

#    tbl = Table.read(mastfile)
#    ID = tbl['ID'][index]
#    redshift = tbl['redshift'][index]

#    flux, wave = readspec(realization, redshift, ID)

#    plt.plot(wave, flux)
#    plt.show()

#    save_phots(realization, dir_info + '/phot%d.fits'%realization)
#    combine_all_reals()

if __name__=='__main__':
    main()
