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
VHSJ = spec.read_filter('VHS', 'VHS_J', uflag=1)

PSg = spec.read_filter('Pan-Starrs', 'PSg')
PSr = spec.read_filter('Pan-Starrs', 'PSr')
PSi = spec.read_filter('Pan-Starrs', 'PSi')
PSz = spec.read_filter('Pan-Starrs', 'PSz')
PSy = spec.read_filter('Pan-Starrs', 'PSy')

W1 = spec.read_filter('WISE', 'W1', uflag=2)
W2 = spec.read_filter('WISE', 'W2', uflag=2)

filters = [PSg, PSr, PSi, PSz, PSy, W1, W2]
maglim = np.array([22.8, 22.7, 22.6, 21.8, 20.8, 18.1+2.699, 17.0+3.339])
filtwave = np.array([filt.central_wavelength.to(u.Angstrom).value for filt in filters])
vega_zp = np.array([0, 0, 0, 0, 0, 2.699, 3.339])

def random_magnification():
    while True:
        x = np.random.rand(1)
        y = np.random.rand(1)

        r = np.sqrt(x**2+y**2)
        if r<1:
            break

    return 2/r

def add_noise(mag, maglim, maglim_nsig=5, additional_msig=0.02):
#    print(type(mag))
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

def weight_LRG(mags, zl, zs):
    # Fundamental plane; use Hyde et al. (2009) relation from SDSS
    # after re-arranging
    # (1-5b)log10(Re/kpc) + 10*b*log10(1+z) + 5*b*log10(D_A/kpc) - 2.5*b*log10(2*pi) = a*log10(sigma/km s-1) + b*mag
    # from mag we derive sigma
    # from sigma and z we derive the cross section

    a = 1.434
    b = 0.315

def generate_lensqso_sample_simple():
    qsophot = Table.read('../codedata/qsophot.fits')
    galphot = Table.read('../codedata/galphot.fits')

    maglim = np.array([22.0, 22.2, 22.2, 21.3, 20.5, 17.1+2.699, 15.7+3.339])

    allphot = []

    for qidx in range(len(qsophot)):
        for gidx in range(len(galphot)):
            mu = random_magnification()

            qmags = np.array([qsophot[colname][qidx] for colname in qsophot.colnames]) - 2.5 * np.log10(mu)
            gmags = np.array([galphot[colname][gidx] for colname in galphot.colnames])

            qobsmag, qobsmagerr = add_noise(qmags, maglim=maglim)
            gobsmag, gobsmagerr = add_noise(gmags, maglim=maglim)

            qobsmag[np.isnan(qobsmag)] = 99.0
            qobsmagerr[np.isnan(qobsmag)] = 1.0
            gobsmag[np.isnan(gobsmag)] = 99.0
            gobsmagerr[np.isnan(gobsmag)] = 1.0

            mags = np.array([qobsmag, gobsmag])
            magerrs = np.array([qobsmagerr, gobsmagerr])

            newmag, newmagerr = combine_mags(mags, magerrs)
            allphot.append(np.concatenate([newmag, newmagerr]))

    tbl = Table(rows=allphot,\
                names=['u', 'g', 'r', 'i', 'z', 'w1', 'w2',\
                      'ue', 'ge', 're', 'ie', 'ze', 'w1e', 'w2e'])
    tbl.write('../codedata/trainingsample/lensqso.fits', overwrite=True)

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

    N_qso = 1000
    N_gal = 200
    info_of_all_lenses = []

    rand_qso_idx = np.random.choice(range(len(qsoinfo)), N_qso, replace=False)

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

            info_thisgal = galinfo[galinfo.colnames][gindex]
            info_of_gals.append(info_thisgal)

            mulist.append(random_magnification())

        info_of_gals = vstack(info_of_gals)
        info_of_qso = vstack([qsoinfo[qsoinfo.colnames][index]]*N_gal)
        info_of_qso['qsoindex'] = [index] * len(info_of_qso)
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

def generate_lensqso_phot(lensinfofile, qsoinfofile, galinfofile,\
                          qsophotfile, galphotfile):
    lensinfo = Table.read(lensinfofile)
    qsoinfo = Table.read(qsoinfofile)
    galinfo = Table.read(galinfofile)
    qsophot = Table.read(qsophotfile)
    galphot = Table.read(galphotfile)

    allphot = []

    for index in range(len(lensinfo)):

        # read quasar phot
        qsoidx = lensinfo['qsoindex'][index]
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

#        print(qsomags)
#        print(galmags)
#        print(newmag0)

        thisphot = np.array([qsomags, galmags, qobsmag, gobsmag,\
                            qobsmagerr, gobsmagerr,
                            newmag0, newmag, newmagerr])
        allphot.append(thisphot)
#        break
#        input('Press enter')
    allphot = np.array(allphot)
    print(allphot.shape)
    tbl = Table({'qsomag0': allphot[:, 0, :], 'galmag0': allphot[:, 1, :],\
                 'qsomag': allphot[:, 2, :], 'galmag': allphot[:, 3, :],\
                 'qsomagerr': allphot[:, 4, :], 'galmagerr': allphot[:, 5, :],\
                 'mag0': allphot[:, 6, :], 'mag': allphot[:, 7, :],\
                 'magerr':allphot[:, 8, :]})
    tbl.write('../codedata/trainingsample/lensqso_highz.fits', overwrite=True)

def qso_photometry(qsoinfofile, qsospecfile):
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
        thisphot = np.array([qso_spec.magnitude(filt) for filt in filters], dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0

        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy', 'W1', 'W2'])
    tbl.write('../codedata/qsophot_PSWISE.fits', overwrite=True)

def gal_photometry(galinfofile, galspecfile):
    # read galaxy info
    galinfo = Table.read(galinfofile)
    # read gal spectra
    jaguar_data_hdulist = fits.open(galspecfile)

    # get the spec
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data
    jaguar_info = jaguar_data_hdulist[3].data

    # get photometric points

    filters = [PSg, PSr, PSi, PSz, PSy, W1, W2]

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
        thisphot = np.array([gal_spec.magnitude(filt) for filt in filters],\
                            dtype=float)
        thisphot[np.isinf(thisphot)] = 99.0
#        print(oldz, newz, factor, galinfo['SIGMA'][index])
#        print(thisphot)
#        fl = 3631 * 10 ** (-0.4 * thisphot) * 1e-23 * (3e10) / (filtwave/1e8)**2 * 1e-8

#        plt.plot(g_wavelength, g_flux)
#        plt.plot(filtwave, fl, '^')
#        plt.show()

        allphot.append(thisphot)

    tbl = Table(rows=allphot, names=['PSg', 'PSr', 'PSi', 'PSz', 'PSy', 'W1', 'W2'])
    tbl.write('../codedata/galphot_PSWISE.fits', overwrite=True)

def generate_galstar_sample(galinfofile, galphotfile, starphotfile,\
                           lensinfofile):

    galinfo = Table.read(galinfofile)
    galphot = Table.read(galphotfile)

    lensinfo = Table.read(lensinfofile)

    starphot = Table.read(starphotfile)

    starphot = starphot[(starphot['magerr_aper_8_r']<0.3)\
                       &(starphot['magerr_aper_8_i']<0.3)\
                       &(starphot['magerr_aper_8_z']<0.3)\
                       &(starphot['magerr_aper_8_y']<0.3)\
                        &(starphot['flags_r']<4)\
                       &(starphot['flags_i']<4)\
                       &(starphot['flags_z']<4)\
                       &(starphot['flags_y']<4)]

    starmag_keys = ['mag_aper_8_g_dered', 'mag_aper_8_r_dered', 'mag_aper_8_i_dered',\
                    'mag_aper_8_z_dered', 'mag_aper_8_y_dered', 'jAperMag4',\
                    'w1mpro', 'w2mpro']
    starmagerr_keys = ['magerr_aper_8_g', 'magerr_aper_8_r',\
                       'magerr_aper_8_i', 'magerr_aper_8_z',\
                       'magerr_aper_8_y', 'jAperMag4Err',\
                       'w1sigmpro', 'w2sigmpro']

    allphot = []

    for index in range(len(lensinfo)):
        # read galaxy phot
        galid = lensinfo['ID'][index]
        galmask = (galinfo['ID']==galid)
        galmags = [galphot[col][galmask][0] for col in galphot.colnames]
        galmags = np.array(galmags)

        gobsmag, gobsmagerr = add_noise(galmags, maglim=maglim)
        gobsmag[np.isnan(gobsmag)] = 99.0
        gobsmagerr[np.isnan(gobsmag)] = 1.0

        sidx = np.random.choice(len(starphot))
        sobsmag = np.array([starphot[key][sidx] for key in starmag_keys], dtype=float)
        sobsmagerr = np.array([starphot[key][sidx] for key in starmagerr_keys], dtype=float)

        sobsmag = sobsmag + vega_zp
        sobsmag[np.isnan(sobsmag)] = 99.0
        sobsmagerr[np.isnan(sobsmag)] = 1.0

        mags = np.array([sobsmag, gobsmag])
        magerrs = np.array([sobsmagerr, gobsmagerr])

        newmag, newmagerr = combine_mags(mags, magerrs)

        badmask = (np.abs(newmag)>50)|(np.abs(newmagerr)>1.)
        newmag[badmask] = 99.0
        newmagerr[badmask] = 1.0

        thisphot = np.array([sobsmag, galmags, sobsmag, gobsmag,\
                            sobsmagerr, gobsmagerr,
                            newmag, newmag, newmagerr])
        allphot.append(thisphot)

    allphot = np.array(allphot)
    print(allphot.shape)
    tbl = Table({'starmag0': allphot[:, 0, :], 'galmag0': allphot[:, 1, :],\
                 'starmag': allphot[:, 2, :], 'galmag': allphot[:, 3, :],\
                 'starmagerr': allphot[:, 4, :], 'galmagerr': allphot[:, 5, :],\
                 'mag0': allphot[:, 6, :], 'mag': allphot[:, 7, :],\
                 'magerr':allphot[:, 8, :]})

    tbl.write('../codedata/trainingsample/galstar.fits', overwrite=True)

def generate_starstar_sample(starphotfile, niter=1000000):

    starphot = Table.read(starphotfile)
    starphot = starphot[(starphot['magerr_aper_8_r']<0.3)\
                       &(starphot['magerr_aper_8_i']<0.3)\
                       &(starphot['magerr_aper_8_z']<0.3)\
                       &(starphot['magerr_aper_8_y']<0.3)\
                       &(starphot['flags_r']<4)\
                       &(starphot['flags_i']<4)\
                       &(starphot['flags_z']<4)\
                       &(starphot['flags_y']<4)]

    starmag_keys = ['mag_aper_8_g_dered', 'mag_aper_8_r_dered', 'mag_aper_8_i_dered',\
                    'mag_aper_8_z_dered', 'mag_aper_8_y_dered', 'jAperMag4',\
                    'w1mpro', 'w2mpro']
    starmagerr_keys = ['magerr_aper_8_g', 'magerr_aper_8_r',\
                       'magerr_aper_8_i', 'magerr_aper_8_z',\
                       'magerr_aper_8_y', 'jAperMag4Err',\
                       'w1sigmpro', 'w2sigmpro']

    allphot = []

    for index in range(niter):
        sidx1 = np.random.choice(len(starphot))
        sobsmag1 = np.array([starphot[key][sidx1] for key in starmag_keys], dtype=float)
        sobsmagerr1 = np.array([starphot[key][sidx1] for key in starmagerr_keys], dtype=float)

        sobsmag1 = sobsmag1 + vega_zp
        sobsmag1[np.isnan(sobsmag1)] = 99.0
        sobsmagerr1[np.isnan(sobsmag1)] = 1.0


        sidx2 = np.random.choice(len(starphot))
        sobsmag2 = np.array([starphot[key][sidx2] for key in starmag_keys], dtype=float)
        sobsmagerr2 = np.array([starphot[key][sidx2] for key in starmagerr_keys], dtype=float)

        sobsmag2 = sobsmag2 + vega_zp
        sobsmag2[np.isnan(sobsmag2)] = 99.0
        sobsmagerr2[np.isnan(sobsmag2)] = 1.0

        mags = np.array([sobsmag1, sobsmag2])
        magerrs = np.array([sobsmagerr1, sobsmagerr2])

        newmag, newmagerr = combine_mags(mags, magerrs)

        badmask = (np.abs(newmag)>50)|(np.abs(newmagerr)>1.)
        newmag[badmask] = 99.0
        newmagerr[badmask] = 1.0

        thisphot = np.array([sobsmag1, sobsmag2,\
                            sobsmagerr1, sobsmagerr2,
                            newmag, newmagerr])
        allphot.append(thisphot)

    allphot = np.array(allphot)
    print(allphot.shape)
    tbl = Table({'starmag1': allphot[:, 0, :], 'starmag2': allphot[:, 1, :],\
                 'starmagerr1': allphot[:, 2, :], 'starmagerr2': allphot[:, 3, :],\
                 'mag': allphot[:, 4, :], 'magerr':allphot[:, 5, :]})

    tbl.write('../codedata/trainingsample/starstar.fits', overwrite=True)

def generate_starphot_sample(starphotfile):
    starphot = Table.read(starphotfile)
    print(len(starphot))
    starphot = starphot[(starphot['magerr_aper_8_r']<0.3)\
                       &(starphot['magerr_aper_8_i']<0.3)\
                       &(starphot['magerr_aper_8_z']<0.3)\
                       &(starphot['magerr_aper_8_y']<0.3)\
                       &(starphot['flags_r']<4)\
                       &(starphot['flags_i']<4)\
                       &(starphot['flags_z']<4)\
                       &(starphot['flags_y']<4)]
    print(len(starphot))

    plt.plot(starphot['JAPERMAG3'], starphot['JAPERMAG3ERR'], '.')
    plt.show()

    starmag_keys = ['mag_aper_8_g_dered', 'mag_aper_8_r_dered', 'mag_aper_8_i_dered',\
                    'mag_aper_8_z_dered', 'mag_aper_8_y_dered', 'jAperMag4',\
                    'w1mpro', 'w2mpro']
    starmagerr_keys = ['magerr_aper_8_g', 'magerr_aper_8_r',\
                       'magerr_aper_8_i', 'magerr_aper_8_z',\
                       'magerr_aper_8_y', 'jAperMag4Err',\
                       'w1sigmpro', 'w2sigmpro']

    allphot = []

    for sidx1 in range(len(starphot)):
        sobsmag1 = np.array([starphot[key][sidx1] for key in starmag_keys], dtype=float)
        sobsmagerr1 = np.array([starphot[key][sidx1] for key in starmagerr_keys], dtype=float)

        sobsmag1 = sobsmag1 + vega_zp
        sobsmag1[np.isnan(sobsmag1)] = 99.0
        sobsmagerr1[np.isnan(sobsmag1)] = 1.0

        newmag = sobsmag1
        newmagerr = sobsmagerr1

        badmask = (np.abs(newmag)>50)|(np.abs(newmagerr)>1.)
        newmag[badmask] = 99.0
        newmagerr[badmask] = 1.0

        thisphot = np.array([newmag, newmagerr])
        allphot.append(thisphot)

    allphot = np.array(allphot)
    print(allphot.shape)
    tbl = Table({'mag': allphot[:, 0, :], 'magerr':allphot[:, 1, :]})

    tbl.write('../codedata/trainingsample/star.fits', overwrite=True)

def train():
    # phot tables
    lensqso = Table.read('../codedata/trainingsample/lensqso.fits')
    galstar = Table.read('../codedata/trainingsample/galstar.fits')
    starstar = Table.read('../codedata/trainingsample/starstar.fits')

    # define a good training sample
    lensqso = lensqso[(lensqso['magerr'][:, 6]<0.1)\
                     &(lensqso['magerr'][:, 5]<0.2)\
                     &(lensqso['magerr'][:, 4]<0.1)\
                     &(lensqso['magerr'][:, 3]<0.1)\
                     &(lensqso['magerr'][:, 2]<0.1)\
                     &(lensqso['magerr'][:, 1]<0.2)]

    galstar = galstar[(galstar['magerr'][:, 6]<0.1)\
                     &(galstar['magerr'][:, 5]<0.2)\
                     &(galstar['magerr'][:, 4]<0.1)\
                     &(galstar['magerr'][:, 3]<0.1)\
                     &(galstar['magerr'][:, 2]<0.1)\
                     &(galstar['magerr'][:, 1]<0.2)]

    starstar = starstar[(starstar['magerr'][:, 6]<0.1)\
                     &(starstar['magerr'][:, 5]<0.2)
                     &(starstar['magerr'][:, 4]<0.1)
                     &(starstar['magerr'][:, 3]<0.1)
                     &(starstar['magerr'][:, 2]<0.1)
                     &(starstar['magerr'][:, 1]<0.2)]

    # define training sample and cross-validation sample
    np.random.seed(12345)
    nlensqso = int(len(lensqso) / 2)
    ngalstar = int(len(galstar) / 2)
    nstarstar = int(len(starstar) / 2)

#    print(nlensqso, ngalstar, nstarstar)

    idx_lensqso = np.random.choice(range(len(lensqso)), nlensqso, replace=False)
    mask_lensqso_tr = np.isin(range(len(lensqso)), idx_lensqso)
    idx_galstar = np.random.choice(range(len(galstar)), ngalstar, replace=False)
    mask_galstar_tr = np.isin(range(len(galstar)), idx_galstar)
    idx_starstar = np.random.choice(range(len(starstar)), nstarstar, replace=False)
    mask_starstar_tr = np.isin(range(len(starstar)), idx_starstar)

    lensqso_train = lensqso[lensqso.colnames][mask_lensqso_tr]
    lensqso_test = lensqso[lensqso.colnames][~mask_lensqso_tr]

    galstar_train = galstar[galstar.colnames][mask_galstar_tr]
    galstar_test = galstar[galstar.colnames][~mask_galstar_tr]

    starstar_train = starstar[starstar.colnames][mask_starstar_tr]
    starstar_test = starstar[starstar.colnames][~mask_starstar_tr]

    # features
    master_train = vstack([lensqso_train['mag'][:], galstar_train['mag'][:],\
                     starstar_train['mag'][:]])
    imag_train = master_train['mag'][:, 2]
    gr_train = master_train['mag'][:, 0] - master_train['mag'][:, 1]
    ri_train = master_train['mag'][:, 1] - master_train['mag'][:, 2]
    iz_train = master_train['mag'][:, 2] - master_train['mag'][:, 3]
    zy_train = master_train['mag'][:, 3] - master_train['mag'][:, 4]
    yJ_train = master_train['mag'][:, 4] - master_train['mag'][:, 5]
    JW1_train = master_train['mag'][:, 5] - master_train['mag'][:, 6]
    W1W2_train = master_train['mag'][:, 6] - master_train['mag'][:, 7]

    master_test = vstack([lensqso_test['mag'][:], galstar_test['mag'][:],\
                     starstar_test['mag'][:]])
#    master_test = vstack([lensqso_test['mag'][:]])
#    print(len(lensqso_train), len(lensqso_test))
    imag_test = master_test['mag'][:, 2]
    gr_test = master_test['mag'][:, 0] - master_test['mag'][:, 1]
    ri_test = master_test['mag'][:, 1] - master_test['mag'][:, 2]
    iz_test = master_test['mag'][:, 2] - master_test['mag'][:, 3]
    zy_test = master_test['mag'][:, 3] - master_test['mag'][:, 4]
    yJ_test = master_test['mag'][:, 4] - master_test['mag'][:, 5]
    JW1_test = master_test['mag'][:, 5] - master_test['mag'][:, 6]
    W1W2_test = master_test['mag'][:, 6] - master_test['mag'][:, 7]

    X = np.transpose([imag_train, gr_train, ri_train, iz_train,\
                      zy_train, yJ_train, JW1_train])
    X_test = np.transpose([imag_test, gr_test, ri_test, iz_test,\
                      zy_test, yJ_test, JW1_test])

    # classes
    y = [0] * len(lensqso_train)\
        + [1] * len(galstar_train)\
        + [2] * len(starstar_train)
    y = np.array(y)

    # random forest
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2)

    # fit
    clf.fit(X, y)

    # predict
    prob = clf.predict_proba(X_test)
    print(prob.shape)

    plt.hist(prob[:len(lensqso_test), 0], label='lens', alpha=0.5, bins=np.arange(0, 1.02, 0.02))
    plt.hist(prob[len(lensqso_test): len(lensqso_test)+len(galstar_test), 0], label='galstar', alpha=0.5, bins=np.arange(0, 1.02, 0.02))
    plt.hist(prob[len(lensqso_test)+len(galstar_test):, 0] ,label='starstar', alpha=0.5, bins=np.arange(0, 1.02, 0.02))

    plt.legend()
    plt.show()


    # real data

    real = Table.read('../codedata/realsample/DES_test_v3.fits')

    imag_real = real['mag'][:, 2]
    gr_real = real['mag'][:, 0] - real['mag'][:, 1]
    ri_real = real['mag'][:, 1] - real['mag'][:, 2]
    iz_real = real['mag'][:, 2] - real['mag'][:, 3]
    zy_real = real['mag'][:, 3] - real['mag'][:, 4]
    yJ_real = real['mag'][:, 4] - real['mag'][:, 5]
    JW1_real = real['mag'][:, 5] - real['mag'][:, 6]
    W1W2_real = real['mag'][:, 6] - real['mag'][:, 7]

    X_real = np.transpose([imag_real, gr_real, ri_real, iz_real,\
                      zy_real, yJ_real, JW1_real])

    prob_real = clf.predict_proba(X_real)

    plt.hist(prob_real[:, 0], bins=np.arange(0, 1.02, 0.02), alpha=0.3, label='lens')
    plt.hist(prob_real[:, 1], bins=np.arange(0, 1.02, 0.02), alpha=0.3, label='galstar')
    plt.hist(prob_real[:, 2], bins=np.arange(0, 1.02, 0.02), alpha=0.3, label='starstar')
    plt.show()
    return clf

def main():
#    qso_photometry()
#    gal_photometry()

#    generate_lensqso_sample()
#    generate_lrgstar_sample()

#    generate_starstar_sample()
#    train()\
    qsoinfofile = '../../../data/simimg/QuasarSED/sim_highz_qso_v2.fits'
    qsospecfile = '../../../data/simimg/QuasarSED/sim_highz_qso_v2_spectra.fits'
    lensinfofile = '../codedata/lensinfo_highz.fits'

    qsophotfile = '../codedata/qsophot_highz_PSWISE.fits'
    galphotfile = '../codedata/galphot_highz_PSWISE.fits'

#    DESstarfile = '../codedata/download_catalog/DES_v5_star.fits'

#    generate_lensqso_info(qsoinfofile, jaguar_info_randzfile, lensinfofile)
#    generate_lensqso_spec(qsospecfile, jaguar_file, lensinfofile,\
#                         qsoinfofile, jaguar_info_sigmafile)
#    qso_photometry(qsoinfofile, qsospecfile)
#    gal_photometry(jaguar_info_randzfile, jaguar_file)
    generate_lensqso_phot(lensinfofile, qsoinfofile, jaguar_info_randzfile,\
                         qsophotfile, galphotfile)
#    generate_galstar_sample(jaguar_info_randzfile, galphotfile, DESstarfile, lensinfofile)
#    generate_starstar_sample(DESstarfile)
#    train()
#    generate_starphot_sample(DESstarfile)

if __name__=='__main__':
    main()
