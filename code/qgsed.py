from __init__ import *
import numpy as np
import os
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
import pandas as pd
import matplotlib.pyplot as plt
import mylib.spec_measurement as spec

from lenspop import sim_a_lens
import surveys

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def FP_sigma(mag, re, redshift, kcorr, kpcsep):
    # surface brightness

    mu = mag + 2.5 * np.log10 (2 * np.pi * re**2) - 10 * np.log10(1+redshift) \
            - kcorr
    print('mu', mu)

    logRe = np.log10(re / kpcsep / 0.7)

    a, b, c = 1.52, -0.78, -8.895

    logsigma = (logRe + b*mu/2.5 - c)/a

    return float(10**logsigma)


def read_template(key=-1):

    template_df = pd.read_csv(\
                dir_data + '/GalaxySED/elliptical.csv')

    if key<0 or key>(len(template_df)-1):
        key = np.random.randint(0, len(template_df)-1)

    template_name = str(template_df['Name'][key])
    template_file = dir_data + '/GalaxySED/%s_spec.dat'%(template_name)
    template_data = np.loadtxt(template_file)
    wave, flux, _, _ = template_data.T

    spec_gal = spec.Spectrum(wave, flux, mode='ABS')
    return spec_gal


def read_template_qso(filename=dir_data+'/quasarComposite.txt'):
    wave, Fv, _, _ = np.loadtxt(filename).T
    Flambda = Fv / wave / wave

    wave = wave * 10000
    qsospec = spec.Spectrum(wave, Flambda, mode='ABS')
    return qsospec


def galaxy_paras(Mabs, re_phys, redshift, tempkey, ellip, pa, filt,\
                 x_center=0, y_center=0, n=4, r_core=1e-3, \
                 std_filt=spec.read_filter('SDSS','SDSSi')):

    tempspec = read_template(tempkey)
    tempspec.normalize(std_filt, Mabs)

    mapp = tempspec.magnitude(filt)\
            + float(defaultcosmo.distmod(redshift)/u.mag)

    tempspec.to_obs(redshift)
    mapp_kcorr = tempspec.magnitude(filt)
    kpcsep = float(u.kpc / defaultcosmo.angular_diameter_distance(redshift))\
            * 206265.
    re = re_phys * kpcsep

    sigma = FP_sigma(mapp, re, redshift, kcorr=0, kpcsep=kpcsep)

    lensmassinfo = [{'Type':'sie', 'sigma':sigma, 'ellipticity':ellip,\
                     'pa':pa, 'x_center':x_center, 'y_center':y_center,\
                     'r_core':r_core}]
    lenslightinfo = [{'Type':'sersic', 'Mag':mapp_kcorr, 're':re, 'n':n,\
                      'ellipticity':ellip, 'pa':pa, \
                      'x_center':x_center, 'y_center':y_center}]

    return [lensmassinfo, lenslightinfo]


def quasar_paras(Mabs, redshift, filt, std_wavelength,\
                x_center, y_center):
    tempspec = read_template_qso()
    # normalize
    flux_stdwave = tempspec.getvalue(std_wavelength)
    Fnu = flux_stdwave * tempspec.units[1] / const.c\
            * (std_wavelength * u.Angstrom)**2
    Fnu_obj = 3631 * u.Jy * 10 ** (-0.4*Mabs)
    scale = float(Fnu_obj / Fnu)
    tempspec.value *= scale

    tempspec.to_obs(redshift)
    mapp_kcorr = tempspec.magnitude(filt)
    print(mapp_kcorr)

    sourceinfo = [{'Type':'point', 'Mag':mapp_kcorr,\
                   'x_center':x_center, 'y_center':y_center}]

    return sourceinfo


def main():
    redshift = 0.5
    angu_dist = float(defaultcosmo.angular_diameter_distance(redshift) / u.kpc)
    kpcsep = 1./angu_dist * 206265.
    distmod = defaultcosmo.distmod(redshift) / u.mag
    mag = -23 + distmod
    print('mag', mag)
    re = 5.* kpcsep
    print('re', re)

    filt = spec.read_filter('Pan-Starrs', 'PSi')

    kcorr = 0

    print(np.log10(FP_sigma(mag, re, redshift, kcorr, kpcsep)))

    lensmassinfo, lensinfo = galaxy_paras(-22, 5, 0.5, 2, 0.3, 30, filt)
    sourceinfo = quasar_paras(-26, 4.5, filt, 1350, 0.4, 0.3)

    PSsurvey = surveys.PanStarrsSurvey()

    sim_a_lens(4.5, sourceinfo, 0.5, lensinfo, lensmassinfo,\
               PSsurvey, 'PSi')


if __name__=='__main__':
    main()
