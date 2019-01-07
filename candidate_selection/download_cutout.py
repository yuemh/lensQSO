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

def download_cutouts_ps1(catalog_file, ra_col='raStack', dec_col='decStack',\
                    outdir='./', size=5):
    info = Table.read(catalog_file)
    ra = info[ra_col]
    dec = info[dec_col]
    namecol = info['JName']
    print(len(info))

    for index in range(len(ra)):
        rai, deci = ra[index], dec[index]

        pname = namecol[index]
        outdir_i = outdir + '/' + pname
        if not os.path.exists(outdir_i):
            os.system('mkdir ' + outdir_i)

#        os.system('rm ' + outdir_i + '/*')
        qs.image.ps1_cutout(rai, deci, datapath=outdir_i,\
                    filters = ['g','r','i','z','y'],size=size)

        pos = SkyCoord(ra=rai, dec=deci, unit=[u.deg, u.deg])
        try:
            dat = SkyView().get_images(position=pos, survey='SDSSu', radius=5*u.arcsecond)
        except HTTPError:
            pass
        if len(dat)>0:
            dat[0].writeto(outdir_i + '/' + pname + '.sdssu.fits', clobber=True)

#        break

def main():
    catalog = dir_data + '/catalog/PS1_UKIDSS_WISE_GAIA_test.fits'
    outdir = dir_data + '/catalog/cutouts'

    download_cutouts_ps1(catalog, outdir=outdir)

if __name__=='__main__':
    main()

