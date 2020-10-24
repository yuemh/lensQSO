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
#from lensqso.sed.lens_sed import galaxysample_JAGUAR, read_quasar_specs, SED,\
#        multiSED
defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

maglim = np.array([22.8, 22.7, 22.6, 21.8, 20.8,\
                   23.0, 22.6, 22.2, 21.5, 20.6,\
                   19.0+0.916, 18.0+1.366, 17.5+1.827,\
                   19.0+0.938, 18.0+1.379, 17.5+1.900,\
                   18.0+2.699, 17.0+3.339])


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

def generate_lensqso_info(qsoinfofile, galinfofile, output):

    # read quasar information
    qsoinfo = Table.read(qsoinfofile)

    # read galaxy info
    galinfo = Table.read(galinfofile)

    # some selection?
    # no. Do it with Einstein radius

    # generate galaxy-quasar pairs
    N_qso = 1000
    N_gal = 1000
    info_of_all_lenses = []

    rand_qso_idx = np.random.choice(range(len(qsoinfo)), N_qso)

    for index in rand_qso_idx:
        print(index)
        #pick up a quasar
        zs_all = np.array([qsoinfo['z'][index]] * len(galinfo))

        # then random draw 100 galaxies and summarize their info
        zl_all = galinfo['redshift']
        sigma_all = galinfo['SIGMA']
        theta_E_thisq = theta_E(sigma_all, zl_all, zs_all)

        weight = theta_E_thisq ** 2
        mask = (theta_E_thisq<0.25)|(theta_E_thisq>3)|(sigma_all>1000)
        weight[mask] = 0
        prob = weight / np.sum(weight)

#        plt.plot(zl_all, sigma_all, '.')
#        plt.plot(zl_all[~mask], sigma_all[~mask], '.')

#        plt.show()
        info_of_gals = []
        mulist = []
        for niter in range(N_gal):
            gindex = np.random.choice(range(len(galinfo)), p=prob)

            info_thisgal =\
                galinfo['ID', 'redshift', 'SIGMA', 'realization'][gindex]
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
        galre = lensinfo['realization'][index]
        galmask = (galinfo['ID']==galid)&(galinfo['realization']==galre)
        galmags = [galphot[col][galmask][0] for col in qsophot.colnames]
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
    generate_lensqso_info(\
            './data/simqso_z4.5_6.5_10000deg2.fits',\
            './data/JAGUAR_galaxy_phot.fits',\
            './data/lensqso_info_test.fits')

#    generate_lensqso_phot(\
#            './data/lensqso_info.fits',\
#            './data/simqso_z4.5_6.5_10000deg2.fits',\
#            './data/JAGUAR_galaxy_phot.fits',\
#            './data/simqso_z4.5_6.5_10000deg2_phot.fits',\
#            './data/JAGUAR_galaxy_phot.fits',\
#            './data/lensqso_phot.fits')

if __name__=='__main__':
    main()

