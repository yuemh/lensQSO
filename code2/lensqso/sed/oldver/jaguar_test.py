import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as c
from astropy.io import fits

import mylib.spectrum.spec_measurement as spec


defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)
dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

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


def FP_sigma(mag, re, redshift):
    # Fundamental plane; use Hyde et al. (2009) relation from SDSS
    # after re-arranging
    # (1-5b)log10(Re/kpc) + 10*b*log10(1+z) + 5*b*log10(D_A/kpc) - 2.5*b*log10(2*pi) = a*log10(sigma/km s-1) + b*mag + c
    # from mag we derive sigma
    # from sigma and z we derive the cross section

    a = 1.434
    b = 0.315
    c = 0.39 - a * 2.19 - b * 19.53

    term1 = (1-5*b)*np.log10(re)
    term2 = 10 * b * np.log10(1 + redshift)
    term3 = 5 * b * np.log10(defaultcosmo.angular_diameter_distance(\
                                        redshift).to(u.kpc).value / 206265)
    term4 = - 2.5 * b * np.log10(2 * np.pi)
    term5 = - b * mag

    return (term1 + term2 + term3 + term4 + term5 - c) / a


def main():
    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'

    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_info_hdulist = fits.open(jaguar_info_masterfile)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data


    tbl = Table.read(jaguar_info_masterfile)

    mask = tbl['redshift']<2
    print(len(tbl))

    SDSSr = spec.read_filter('SDSS', 'SDSSr')
    rmags = []
    for index in range(len(tbl)):
        zgal = tbl['redshift'][index]
        wavegal = (1 + zgal) * jaguar_wave
        fluxgal = jaguar_flux[index] / (1+zgal)

        spec_gal = spec.Spectrum(wavegal, fluxgal)

        SDSSr.update(SDSSr.wavelength * (1+zgal), SDSSr.value, SDSSr.units)
        rmag = spec_gal.magnitude(SDSSr)
        SDSSr.update(SDSSr.wavelength / (1+zgal), SDSSr.value, SDSSr.units)

        rmags.append(rmag)
        if index%1000==0:
            print(index, rmag, zgal)

    rmags = np.array(rmags)

    mag = -2.5 * np.log10(tbl['HST_F606W_fnu'] / 3631e9)
#    mag=rmag
    redshift = tbl['redshift']
    sersic_n = tbl['sersic_n']
    mstar = tbl['mStar']
    re = tbl['Re_circ']

    sigma1 = MStar_sigma(mstar, re, sersic_n)
    sigma2 = 10**FP_sigma(rmags, re, redshift)

    sigma1 = tbl['SIGMA']
#    plt.hist(sigma1, bins=np.arange(0, 500, 10))
    plt.plot(sigma1[mask], sigma2[mask], '.')
#    plt.plot(mag, rmags, '.')
    plt.show()

if __name__=='__main__':
    main()
