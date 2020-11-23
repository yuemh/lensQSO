import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as c

from sklearn.ensemble import RandomForestClassifier

import mylib.spectrum.spec_measurement as spec
from lensqso.sed.lens_sed import galaxysample_JAGUAR, read_quasar_specs, SED,\
        multiSED

dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

jaguar_file = dir_data +\
        '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
jaguar_info_masterfile = dir_data +\
        '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'
jaguar_info_sigmafile = dir_data +\
        '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'
jaguar_info_randzfile = dir_data +\
        '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.randz2.fits'

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

### use the following filters ###

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

VHSJ = spec.read_filter('VHS', 'VHS_J', uflag=1)
VHSJ.set_vega_AB(0.916)
VHSH = spec.read_filter('VHS', 'VHS_H', uflag=1)
VHSH.set_vega_AB(1.366)
VHSK = spec.read_filter('VHS', 'VHS_Ks', uflag=1)
VHSK.set_vega_AB(1.827)

UHSJ = spec.read_filter('UHS', 'UHS_J', uflag=1)
UHSJ.set_vega_AB(0.938)
UHSH = spec.read_filter('UHS', 'UHS_H', uflag=1)
UHSH.set_vega_AB(1.379)
UHSK = spec.read_filter('UHS', 'UHS_K', uflag=1)
UHSK.set_vega_AB(1.900)

W1 = spec.read_filter('WISE', 'W1', uflag=2)
W1.set_vega_AB(2.699)
W2 = spec.read_filter('WISE', 'W2', uflag=2)
W2.set_vega_AB(3.339)

filters = [PSg, PSr, PSi, PSz, PSy,\
           DESg, DESr, DESi, DESz, DESY,\
           VHSJ, VHSH, VHSK, UHSJ, UHSH, UHSK,\
           W1, W2]
maglim = np.array([22.8, 22.7, 22.6, 21.8, 20.8,\
                   23.0, 22.6, 22.2, 21.5, 20.6,\
                   19.0, 18.0, 17.5, 19.0, 18.0, 17.5,\
                   18.0, 17.0])
filtwave = np.array([filt.central_wavelength.to(u.Angstrom).value for filt in filters])

# utilities

def random_magnification():
    while True:
        x = np.random.rand(1)
        y = np.random.rand(1)

        r = np.sqrt(x**2+y**2)
        if r<1:
            break

    return 2/r

def add_noise(mag, maglim, maglim_nsig=5, additional_msig=0.02):
    flux = 10**(-0.4 * mag)
    back_fluxerr = 10**(-0.4 * maglim) / maglim_nsig
    add_fluxerr = additional_msig * flux

    fluxerr = np.sqrt(back_fluxerr**2 + add_fluxerr**2)

    newflux = flux + np.random.normal(0, fluxerr)

    newmag = -2.5 * np.log10(newflux)
    newmagerr = -2.5 / np.log(10) * fluxerr / newflux

    return (newmag, newmagerr)

def combine_mags(mags, magerrs):
    fluxes = 10 ** (-0.4 * np.array(mags))
    fluxerrs = fluxes * 0.4 * np.log(10) * np.array(magerrs)

    aflux = np.sum(fluxes, axis=0)
    afluxerr = np.sqrt(np.sum(fluxerrs**2, axis=0))

    mag = -2.5 * np.log10(aflux)
    magerr = 2.5 / np.log(10) / aflux * afluxerr

    return (mag, magerr)

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
    Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)

    theta = Dls / Ds * 2 * np.pi * (sigma/3e5)** 2 * 206265
    return theta

def generate_lensqso_info(qsoinfofile, galinfofile, output,\
                           bad_ids_file='./bad_id_s150_z2.txt'):

    # read quasar information
    qsoinfo = Table.read(qsoinfofile)
    print(len(qsoinfo))

    # read galaxy info
    galinfo = Table.read(galinfofile)
    bad_ids = np.loadtxt(bad_ids_file)

#    galinfo_orig = Table.read(jaguar_info_sigmafile)

    print(len(galinfo))
    galinfo = galinfo[~np.isin(galinfo['ID'], bad_ids)]
    print(len(galinfo))

    # generate galaxy-quasar pairs

    N_qso = 5000
    N_gal = 1000
    info_of_all_lenses = []

    rand_qso_idx = np.random.choice(range(len(qsoinfo)), N_qso)
#    print(qsoinfo)
#    return 0

    for index in rand_qso_idx:
        #pick up a quasar
        zs_all = np.array([qsoinfo['z'][index]] * len(galinfo))

        # then random draw 100 galaxies and summarize their info
        zl_all = galinfo['redshift']
        sigma_all = galinfo['SIGMA']
        theta_E_thisq = theta_E(sigma_all, zl_all, zs_all)

        weight = theta_E_thisq ** 2
        prob = weight / np.sum(weight)

        info_of_gals = []
        mulist = []
        for niter in range(N_gal):
            gindex = np.random.choice(range(len(galinfo)), p=prob)

            info_thisgal = galinfo['ID', 'redshift', 'SIGMA'][gindex]
            info_of_gals.append(info_thisgal)

            mulist.append(random_magnification())

        info_of_gals = vstack(info_of_gals)
        info_of_qso = vstack([qsoinfo['z', 'absMag', 'appMag'][index]]*N_gal)
        info_of_qso['QID'] = [index] * len(info_of_qso)
        info_of_qso['mu'] = mulist

        info_of_lens = hstack([info_of_qso, info_of_gals])

        info_of_all_lenses.append(info_of_lens)

    info_of_all_lenses = vstack(info_of_all_lenses)
    info_of_all_lenses.write(output, overwrite=True)

def read_galspec(galinfofile, galspecfile, gid, oldz, newz, factor):
    jaguar_info = Table.read(galinfofile)

    # read gal spectra
    jaguar_data_hdulist = fits.open(galspecfile)

    # get the spec
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data
#    jaguar_info = jaguar_data_hdulist[3].data
    # get the index of the galaxy
    gindex = np.where(jaguar_info['ID']==gid)

    print(jaguar_info['redshift', 'ID', 'SIGMA'][gindex])

    # get the flux of the galaxy
    gflux = jaguar_flux[gindex] / (1 + newz) * factor
    gwave = jaguar_wave * (1 + newz)
    spec = SED(wavelength=gwave, value=gflux[0])

    return spec

def qso_photometry(qsoinfofile, qsospecfile, qsophotfile):
    # get quasar info
    qsoinfo = Table.read(qsoinfofile)

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

        qso_spec = SED(wavelength=q_wavelength, value=q_flux * 1e-17)
        thisphot = np.array([qso_spec.magnitude(filt) - filt.vega_AB\
                             for filt in filters], dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0

        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                'DESg', 'DESr', 'DESi', 'DESz', 'DESy',\
                'VHSJ', 'VHSH', 'VHSK', 'UHSJ', 'UHSH', 'UHSK', 'W1', 'W2'])

    tbl.write(qsophotfile, overwrite=True)

def gal_photometry(galinfofile, galspecfile, galphotfile):
    # read galaxy info
    galinfo = Table.read(galinfofile)
    # read gal spectra
    jaguar_data_hdulist = fits.open(galspecfile)

    # get the spec
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data
    jaguar_info = jaguar_data_hdulist[3].data

    # get photometric points

    allinfo = []
    allphot = []

    for index in range(len(galinfo)):

        # read galaxy spec
        galid = galinfo['ID'][index]
        oldz = galinfo['redshift'][index]
        newz = galinfo['new_redshift'][index]
        factor = galinfo['factor'][index]

        g_wavelength = jaguar_wave * (1+newz)
        g_flux = jaguar_flux[jaguar_info['ID']==galid][0] / (1+newz) * factor

        gal_spec = SED(wavelength=g_wavelength, value=g_flux)
        thisphot = np.array([gal_spec.magnitude(filt) - filt.vega_AB\
                             for filt in filters], dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0

        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                'DESg', 'DESr', 'DESi', 'DESz', 'DESy',\
                'VHSJ', 'VHSH', 'VHSK', 'UHSJ', 'UHSH', 'UHSK', 'W1', 'W2'])

    tbl.write(galphotfile, overwrite=True)

def generate_lensqso_phot(lensinfofile, qsoinfofile, galinfofile,\
                          qsophotfile, galphotfile, lensphotfile):
    lensinfo = Table.read(lensinfofile)
    qsoinfo = Table.read(qsoinfofile)
    galinfo = Table.read(galinfofile)
    qsophot = Table.read(qsophotfile)
    galphot = Table.read(galphotfile)

    allphot = []
    allphoterr = []
    qsoidlist = []
    galidlist = []

    print(qsophot.colnames)
    print(galphot.colnames)

    for index in range(len(lensinfo)):

        # read quasar phot
        qsoidx = lensinfo['QID'][index]
        mu = lensinfo['mu'][index]
        qsomags = [qsophot[col][qsoidx] for col in qsophot.colnames]
        qsomags = np.array(qsomags) - 2.5 * np.log10(mu)

        # read galaxy spec
        galid = lensinfo['ID'][index]
        galmask = (galinfo['ID']==galid)
        galmags = [galphot[col][galmask][0] for col in galphot.colnames]
        galmags = np.array(galmags)

        qobsmag, qobsmagerr = add_noise(qsomags, maglim=maglim)
        gobsmag, gobsmagerr = add_noise(galmags, maglim=maglim)

        qobsmag[np.isnan(qobsmag)] = 99.0
        qobsmagerr[np.isnan(qobsmag)] = 1.0
        gobsmag[np.isnan(gobsmag)] = 99.0
        gobsmagerr[np.isnan(gobsmag)] = 1.0

        mags = np.array([qobsmag, gobsmag])
        magerrs = np.array([qobsmagerr, gobsmagerr])
        mags0 = np.array([qsomags, galmags])
        magerrs0 = np.array([[0]*len(qsomags), [0]*len(galmags)])

        newmag, newmagerr = combine_mags(mags, magerrs)
        newmag0, newmagerr0 = combine_mags(mags0, magerrs0)

        badmask = (np.abs(newmag)>50)|(np.abs(newmagerr)>1.)
        newmag[badmask] = 99.0
        newmagerr[badmask] = 1.0

        allphot.append(newmag)
        allphoterr.append(newmagerr)
        qsoidlist.append(qsoidx)
        galidlist.append(galid)

    allphot = np.array(allphot)
    allphoterr = np.array(allphoterr)

    print(allphot.shape)

    tbl = Table({'PSg': allphot[:,0],\
                 'PSg_err': allphoterr[:,0],\
                 'PSr': allphot[:,1],\
                 'PSr_err': allphoterr[:,1],\
                 'PSi': allphot[:,2],\
                 'PSi_err': allphoterr[:,2],\
                 'PSz': allphot[:,3],\
                 'PSz_err': allphoterr[:,3],\
                 'PSy': allphot[:,4],\
                 'PSy_err': allphoterr[:,4],\
                 'DESg': allphot[:,5],\
                 'DESg_err': allphoterr[:,5],\
                 'DESr': allphot[:,6],\
                 'DESr_err': allphoterr[:,6],\
                 'DESi': allphot[:,7],\
                 'DESi_err': allphoterr[:,7],\
                 'DESz': allphot[:,8],\
                 'DESz_err': allphoterr[:,8],\
                 'DESy': allphot[:,9],\
                 'DESy_err': allphoterr[:,9],\
                 'VHSJ': allphot[:,10],\
                 'VHSJ_err': allphoterr[:,10],\
                 'VHSH': allphot[:,11],\
                 'VHSH_err': allphoterr[:,11],\
                 'VHSK': allphot[:,12],\
                 'VHSK_err': allphoterr[:,12],\
                 'UHSJ': allphot[:,13],\
                 'UHSJ_err': allphoterr[:,13],\
                 'UHSH': allphot[:,14],\
                 'UHSH_err': allphoterr[:,14],\
                 'UHSK': allphot[:,15],\
                 'UHSK_err': allphoterr[:,15],\
                 'W1': allphot[:,16],\
                 'W1_err': allphoterr[:,16],\
                 'W2': allphot[:,17],\
                 'W2_err': allphoterr[:,17]})

    tbl.write(lensphotfile, overwrite=True)


def main():

    qsoinfofile = '../../../data/simimg/QuasarSED/sim_highz_qso_10000deg2.fits'
    qsospecfile = '../../../data/simimg/QuasarSED/sim_highz_qso_10000deg2_spectra.fits'
    qsophotfile = '../codedata/qsophot_highz_master.fits'

    galinfofile = jaguar_info_randzfile
    galspecfile = jaguar_file
    galphotfile = '../codedata/galphot_highz_master.fits'

    lensinfofile = '../codedata/lensinfo_highz_master.fits'
    lensphotfile = '../codedata/lensphot_highz_master.fits'

#    generate_lensqso_info(qsoinfofile, jaguar_info_randzfile, lensinfofile)

#    qso_photometry(qsoinfofile, qsospecfile, qsophotfile)
#    gal_photometry(jaguar_info_randzfile, jaguar_file, galphotfile)
    generate_lensqso_phot(lensinfofile, qsoinfofile, jaguar_info_randzfile,\
                         qsophotfile, galphotfile, lensphotfile)

if __name__=='__main__':
    main()

