import os, sys

import numpy as np
import matplotlib.pyplot as plt
import random

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as c
from astropy.io import fits

from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk,buildQsoSpectrum,save_spectra,load_spectra,restore_qso_grid
from simqso.sqmodels import BOSS_DR9_PLEpivot,get_BossDr9_model_vars
from simqso import sqmodels, hiforest

#from matplotlib import font_manager, rcParams
#fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
#prop = font_manager.FontProperties(fname=fontpath)
#rcParams['font.family'] = prop.get_name()

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)
#np.random.seed(1)

import mylib.spectrum.spec_measurement as spec
from mylib.spectrum.spec_measurement import Spectrum

dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

class SED(Spectrum):
    def __init__(self, wavelength, value,\
                units=[u.Angstrom,u.erg/u.s/u.cm/u.cm/u.Angstrom],\
                mode='OBS'):
        super().__init__(wavelength, value, units=units, mode=mode)

#    @property
    def lfl_filter(self, filt, unit=u.erg/u.s/u.cm/u.cm):
        mag_AB = self.magnitude(filt)
        wavelength = filt.central_wavelength
        frequency = c.c / wavelength

        fv = 3631 * 10 ** (-0.4*mag_AB) * u.Jy
        vfv = frequency * fv
        lfl = vfv.to(unit).value

        return lfl

class multiSED(SED):
    def __init__(self, wavelength=np.arange(1000, 100000, 10), specs=[]):

        value = np.zeros(len(wavelength))
        super().__init__(wavelength, value,\
                units=[u.Angstrom,u.erg/u.s/u.cm/u.cm/u.Angstrom],\
                mode='OBS')
        for spec in specs:
            self.add_spec(spec)

    def add_spec(self, spec):
        if not spec.mode=='OBS':
            raise ValueError(\
"The new spectrum should be an observed-mode spectrum (self.mode=='OBS')")

        else:
            addvalue = (spec.getvalue(self.wavelength)\
                    * spec.units[1]).to(self.units[1]).value
            newvalue = self.value + addvalue

            self.update(wavelength=self.wavelength,\
                        value=newvalue, units=self.units)

def random_quasarsample_ian(redshift, Mabs, **kwargs):

    try:
        _ = len(redshift)
    except TypeError:
        redshift = [redshift]

    try:
        _ = len(Mabs)
    except TypeError:
        Mabs = [Mabs]

    obswave = sqbase.fixed_R_dispersion(500, 6e4, 3000)

    # cover 3000A to 5um at R=500
    forestModel = sqmodels.forestModels['McGreer+2013']
    forestSpec = hiforest.IGMTransmissionGrid(obswave, forestModel,\
                                              1, zmax=redshift[0])
    forestVar = HIAbsorptionVar(forestSpec)

    # Basic setting 

    M = AbsMagVar(FixedSampler(Mabs), restWave=1450)
    z = RedshiftVar(FixedSampler(redshift))
    qsos = QsoSimPoints([M, z], cosmo=defaultcosmo, units='luminosity')

    # use the canonical values for power law continuum slopes in FUV/NUV,
    # with breakpoint at 1215A
    contVar = BrokenPowerLawContinuumVar(\
            [GaussianSampler(-1.5,0.3), GaussianSampler(-0.5,0.3)], [1215.])

    # add two dust components as in Lyu+Rieke 2017,
    # but reduce the hot dust flux by a factor of 2
    subDustVar = DustBlackbodyVar([ConstSampler(0.05),ConstSampler(1800.)],\
                                  name='sublimdust')
    subDustVar.set_associated_var(contVar)
    hotDustVar = DustBlackbodyVar([ConstSampler(0.1),ConstSampler(880.)],\
                                  name='hotdust')
    hotDustVar.set_associated_var(contVar)

    # generate lines using the Baldwin Effect emission line model from BOSS DR9
    emLineVar = generateBEffEmissionLines(qsos.absMag)

    # the default iron template from Vestergaard & Wilkes 2001
    # was modified to fit BOSS spectra
    fescales = [(0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),\
                (1868,2140,1.0),(2140,3500,1.0)]
    feVar = FeTemplateVar(VW01FeTemplateGrid(qsos.z, obswave, scales=fescales))

    # Now add the features to the QSO grid
    qsos.addVars([contVar, subDustVar, hotDustVar, emLineVar,\
                  feVar, forestVar])

    # ready to generate spectra
    _, obsflux = buildSpectraBulk(obswave, qsos, saveSpectra=True)

    return [obsflux[0], obswave, redshift]

def add_sigma_JAGUAR(jaguar_file, jaguar_info_masterfile, output):
    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data

    jaguar_info = Table.read(jaguar_info_masterfile)

    sigma_list = []
    selected_index_list = []
    for index in range(len(jaguar_info)):

        flux = jaguar_flux[index]
        wave = jaguar_wave
        z_lens = jaguar_info['redshift'][index]

        sigma = MStar_sigma(jaguar_info['mStar'][index],\
                            jaguar_info['Re_circ'][index],\
                            jaguar_info['sersic_n'][index])
        sigma_list.append(sigma)

    jaguar_info['SIGMA'] = np.array(sigma_list)
    jaguar_info.write(output)

def galaxysample_JAGUAR(jaguar_file, jaguar_info_sigmafile,\
                        sigma_lim=100, z_lim=5,\
                        random=True, seed=-1):
    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data

    jaguar_info = Table.read(jaguar_info_sigmafile)

    mask_gal = (jaguar_info['redshift']<z_lim)\
            &(jaguar_info['SIGMA']>sigma_lim)

    allidx = np.arange(len(jaguar_info))
    masked_idx = allidx[mask_gal]

    if random:

        print(len(masked_idx))

        if seed>0:
            random.seed(seed)
        random_selected_index = np.random.choice(masked_idx)
        redshift = jaguar_info['redshift'][random_selected_index]
        sigma = jaguar_info['SIGMA'][random_selected_index]

        return [jaguar_flux[random_selected_index]/(1+redshift),\
                jaguar_wave*(1+redshift), \
                redshift, sigma]

    else:
        redshift = jaguar_info['redshift'][masked_idx]
        sigma = jaguar_info['SIGMA'][masked_idx]
        flux = [jaguar_flux[masked_idx[idx]]/(1+redshift[idx])\
               for idx in range(len(masked_idx))]
        wave = [jaguar_wave * (1+z) for z in redshift]

        return [flux, wave, redshift, sigma]


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

def save_qso_spec_ian():
    '''
    wave = sqbase.fixed_R_dispersion(1000,20e4,1000)

    # just make up a few random redshifts between z=2 and z=3, then assign apparent mags according 
    # to the BOSS DR9 QLF
    nqso = 100
    np.random.seed(12345)
    zin = 4.5 + np.random.rand(nqso)
    kcorr = sqbase.ContinuumKCorr('DECam-r',1450,effWaveBand='SDSS-r')
    qsos = generateQlfPoints(BOSS_DR9_PLEpivot(cosmo=defaultcosmo),
                                                  (17,22),(4.5, 5.5),
                                                  kcorr=kcorr,zin=zin,
                                                  qlfseed=12345,gridseed=67890)

    # add the fiducial quasar SED model from BOSS DR9
    # need to set forestseed if the forest transmission sightlines are to be reproducible
    sedVars = get_BossDr9_model_vars(qsos,wave,0,forestseed=192837465,verbose=1)
    qsos.addVars(sedVars)

    qsos.loadPhotoMap([('DECam','DECaLS'),('WISE','AllWISE')])

    _,spectra = buildSpectraBulk(wave,qsos,saveSpectra=True,maxIter=3,verbose=10)

    qsos.write('../codedata/quasarinfo.fits',extname='quickspec',overwrite=True)
    save_spectra(wave,spectra,'../codedata/quasarspec.fits')
    '''

    fluxes = []
    Mabs_list = []
    redshift_list = []

    for Mabs in np.arange(-29, -24, 0.5):
        for redshift in np.arange(4.5, 5.5, 0.1):
            spec, wave, redshift = random_quasarsample_ian(redshift, Mabs)

            Mabs_list.append(Mabs)
            redshift_list.append(redshift)
            fluxes.append(spec)

    ### Saving table ###

    prim_hdu = fits.PrimaryHDU()

    Mabs_col = fits.Column(name='Mabs', array=Mabs_list, format='D')
    redshift_col = fits.Column(name='redshift', array=redshift_list, format='D')

    info_hdu = fits.BinTableHDU.from_columns([Mabs_col, redshift_col])
    info_hdu.header['w0'] = 500
    info_hdu.header['w1'] = 60000
    info_hdu.header['R'] = 3000

    data_hdu = fits.ImageHDU(data=np.array(fluxes))

    hdulist = fits.HDUList([prim_hdu, info_hdu, data_hdu])

    hdulist.writeto('../codedata/quasar.fits', overwrite=True)

def read_quasar_specs(filename, plot=False):
    hdulist = fits.open(filename)

    header = hdulist[1].header
    qsoinfo = hdulist[1].data
    qsospec = hdulist[2].data

    w0 = header['w0']
    w1 = header['w1']
    R = header['R']
    wave = sqbase.fixed_R_dispersion(w0, w1, R)

    if plot:
        for index in range(len(qsospec)):
            redshift = qsoinfo['redshift'][index]
            plt.plot(wave, qsospec[index])
            plt.plot([1216*(1+redshift)]*2, [0, 1e-15], 'k-')
            plt.xlim(3400, 1e4)
            plt.show()
            plt.close('all')

    return [wave, qsospec]

def save_gal_spec_jaguar():

    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'
    jaguar_info_sigmafile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'

    sigma_list = []
    redshift_list = []

    wave_list = []
    flux_list = []

    flag = True

    while flag:

        gal_flux = random_galaxysample_JAGUAR(jaguar_file, \
                    jaguar_info_sigmafile, sigma_lim=250)
        g_flux, g_wavelength, g_redshift, g_sigma = gal_flux

        if g_redshift in redshift_list:
            continue

        plt.plot(g_wavelength, g_flux, 'b-', label = 'LRG')

        plt.title(r'$z_l=%.2f, \sigma=%.2f$'%(g_redshift, g_sigma))
        plt.xlabel(r'Wavelength $(\AA)$')
        plt.ylabel(r'Flux $(erg/s/cm^2/\AA)$')
        plt.legend()
        plt.show()

        goodflag = input('Include this galaxy?')
        if goodflag in 'Yy':
            sigma_list.append(g_sigma)
            redshift_list.append(g_redshift)
            wave_list.append(g_wavelength)
            flux_list.append(g_flux)

        contflag = input('Continue?')
        if not contflag in 'Yy':
            flag = False

    ### Saving table

    print(len(sigma_list), len(redshift_list), len(wave_list), len(flux_list))

    prim_hdu = fits.PrimaryHDU()

    sigma_col = fits.Column(name='sigma', array=sigma_list, format='D')
    redshift_col = fits.Column(name='redshift', array=redshift_list, format='D')

    info_hdu = fits.BinTableHDU.from_columns([sigma_col, redshift_col])

    wave_hdu = fits.ImageHDU(data=np.array(wave_list))
    flux_hdu = fits.ImageHDU(data=np.array(flux_list))

    hdulist = fits.HDUList([prim_hdu, info_hdu, wave_hdu, flux_hdu])

    hdulist.writeto('../codedata/galaxy.fits', overwrite=True)

def random_magnification():
    x = np.random.rand(1)
    y = np.random.rand(1)

    r = np.sqrt(x**2+y**2)
    return 2/r

def test():

    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'
    jaguar_info_sigmafile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'

#    add_sigma_JAGUAR(jaguar_file, jaguar_info_masterfile,\
#                    jaguar_info_sigmafile)

#    '''
    PSg = spec.read_filter('Pan-Starrs', 'PSg')
    PSr = spec.read_filter('Pan-Starrs', 'PSr')
    PSi = spec.read_filter('Pan-Starrs', 'PSi')
    PSz = spec.read_filter('Pan-Starrs', 'PSz')
    PSy = spec.read_filter('Pan-Starrs', 'PSy')

    W1 = spec.read_filter('WISE', 'W1', uflag=2)
    W2 = spec.read_filter('WISE', 'W2', uflag=2)

    flag=True
    while flag:
        lfl_list = []
        q_lfl_list = []
        g_lfl_list = []

        l_list = []

        qso_flux = random_quasarsample_ian(5, -26)
        q_flux, q_wavelength, q_redshift = qso_flux

        gal_flux = galaxysample_JAGUAR(jaguar_file, \
                    jaguar_info_sigmafile, sigma_lim=200,\
                    z_lim=1, random=1)
        g_flux, g_wavelength, g_redshift, g_sigma = gal_flux

        qso_spec = SED(wavelength=q_wavelength, value=q_flux)
        gal_spec = SED(wavelength=g_wavelength, value=g_flux)

        sdss_like_wave = np.arange(3000, 10000, 3)
        qspec_sdss = qso_spec.getvalue(sdss_like_wave)
        gspec_sdss = gal_spec.getvalue(sdss_like_wave)

        aspec_sdss = qspec_sdss + gspec_sdss
        aspec = SED(wavelength=sdss_like_wave, value=aspec_sdss)

        gal_spec.to_abs(redshift=g_redshift)

        gabsmags = [gal_spec.magnitude(filt) for filt in [PSg, PSr, PSi, PSz, PSy]]
        print(gabsmags)

        gal_spec.to_obs(redshift=g_redshift)
        gmags = [gal_spec.magnitude(filt) for filt in [PSg, PSr, PSi, PSz, PSy]]
        print(gmags)
        qmags = [qso_spec.magnitude(filt) for filt in [PSg, PSr, PSi, PSz, PSy]]
        print(qmags)
        amags = [aspec.magnitude(filt) for filt in [PSg, PSr, PSi, PSz, PSy]]
        print(amags)

        plt.plot(sdss_like_wave, aspec_sdss, 'k-', label='QSO + LRG')
        plt.plot(sdss_like_wave, qspec_sdss, 'r-', alpha=0.3, label = 'QSO')
        plt.plot(sdss_like_wave, gspec_sdss, 'b-', alpha=0.3, label = 'LRG')

        plt.title(r'$z_l=%.2f, \sigma=%.2f$'%(g_redshift, g_sigma))
        plt.xlabel(r'Wavelength $(\AA)$')
        plt.ylabel(r'Flux $(erg/s/cm^2/\AA)$')
        plt.legend()
        plt.show()

        '''
        for filt in [PSg, PSr, PSi, PSz, PSy, W1, W2]:
            q_lfl = qso_spec.lfl_filter(filt)
            g_lfl = gal_spec.lfl_filter(filt)

            q_lfl_list.append(q_lfl)
            g_lfl_list.append(g_lfl)
            lfl_list.append(q_lfl+g_lfl)
            l_list.append(filt.central_wavelength.to(u.um).value)


        plt.plot(l_list, lfl_list, 'o')
        plt.plot(l_list, g_lfl_list, 'b^')
        plt.plot(gal_spec.wavelength/1e4, gal_spec.value*gal_spec.wavelength,\
                 color='b', lw=1, alpha=0.3)

        plt.plot(l_list, q_lfl_list, 'rs')
        plt.plot(qso_spec.wavelength/1e4, qso_spec.value*qso_spec.wavelength,\
                 color='r', lw=1, alpha=0.3)
        plt.title(r'$z_l=%.2f, \sigma=%.2f$'%(g_redshift, g_sigma))


        plt.xlim([0.4, 6])
#        plt.xscale('log')
        plt.show()
#        '''

def main():
#    save_qso_spec_ian()
#    read_quasar_specs('../codedata/quasar.fits', plot=True)
    test()

if __name__=='__main__':
    main()
