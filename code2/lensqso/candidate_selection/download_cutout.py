import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
import qsosurvey as qs
import qsosurvey.obs as obs
from urllib2 import HTTPError

dir_root = os.path.abspath(os.getcwd()+'/../../')
dir_data = dir_root + '/data'
dir_catalog = dir_data + '/catalog/catalog_download'

def download_cutouts_ps1(catalog_file, ra_col='raStack', dec_col='decStack',\
                    name_col='', outdir='./', size=5):
    info = Table.read(catalog_file)
    ra = info[ra_col]
    dec = info[dec_col]

    if len(name_col):
        name = info[name_col]

    else:
        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        name = ['J' + str(onename) for onename in name]

    for index in range(len(ra)):
        rai, deci = ra[index], dec[index]
        namei = name[index]

        print('Downloading object %s (number %d)'%(namei, index))

        outdir_i = outdir + '/' + namei
        if not os.path.exists(outdir_i):
            os.system('mkdir -p ' + outdir_i)

        qs.image.ps1_cutout(rai, deci, datapath=outdir_i,\
                    filters = ['g','r','i','z','y'],size=size)

        pos = SkyCoord(ra=rai, dec=deci, unit=[u.deg, u.deg])


def download_cutouts_sdss(catalog_file, ra_col='raStack', dec_col='decStack',\
                    name_col='', outdir='./', size=5):

    info = Table.read(catalog_file)
    ra = info[ra_col]
    dec = info[dec_col]

    if len(name_col):
        name = info[name_col]

    else:
        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        name = ['J' + str(onename) for onename in name]

    for index in range(len(ra)):
        rai, deci = ra[index], dec[index]

        namei = name[index]
        outdir_i = outdir + '/' + namei
        if not os.path.exists(outdir_i):
            os.system('mkdir -p ' + outdir_i)

        pos = SkyCoord(ra=rai, dec=deci, unit='deg')

        try:
#            dat = SkyView().get_images(position=pos, \
#                    survey=['SDSSu', 'SDSSg', 'SDSSr', 'SDSSi', 'SDSSz'], \
#                    radius=5*u.arcsecond)
            dat = SkyView().get_images(position=pos, \
                    survey='SDSSu', radius=5*u.arcsecond)

            if len(dat)>0:
                dat[0].writeto(outdir_i + '/' + namei + '.sdssu.fits', clobber=True)
#                dat[1].writeto(outdir_i + '/' + namei + '.sdssg.fits', clobber=True)
#                dat[2].writeto(outdir_i + '/' + namei + '.sdssr.fits', clobber=True)
#                dat[3].writeto(outdir_i + '/' + namei + '.sdssi.fits', clobber=True)
#                dat[4].writeto(outdir_i + '/' + namei + '.sdssz.fits', clobber=True)

        except HTTPError:
            pass

def download_cutouts_legacy(catalog_file, ra_col='raStack', dec_col='decStack',\
                            name_col='', outdir='./'):
    info = Table.read(catalog_file)
    ra = info[ra_col]
    dec = info[dec_col]

    if len(name_col):
        name = info[name_col]

    else:
        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        name = ['J' + str(onename) for onename in name]

    for index in range(len(ra)):
        rai, deci = ra[index], dec[index]

        namei = name[index]
        outdir_i = outdir + '/' + namei
        if not os.path.exists(outdir_i):
            os.system('mkdir -p ' + outdir_i)

        if deci<30:
            qs.image.dels_cutout(rai, deci, datapath=outdir_i,\
                                 release='decals-dr7')
        else:
            qs.image.dels_cutout(rai, deci, datapath=outdir_i,\
                                 release='mzls+bass-dr6')

def main():
    catalog = dir_catalog +\
            '/2019_LDSS3/PS_WISE_GAIA_Stationary_Morphriz_simbad.fits'
    outdir = dir_data +\
            '/catalog/cutouts/PS_WISE_GAIA_stationary/Morphriz/sdss'

    if not os.path.exists(outdir):
        os.system('mkdir -p '+outdir)
    download_cutouts_sdss(catalog, outdir=outdir,\
                         ra_col='raStack', dec_col='decStack')

if __name__=='__main__':
    main()

