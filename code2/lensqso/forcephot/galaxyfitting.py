import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from scipy.fftpack import fft2, ifft2
from scipy.optimize import fsolve
from scipy.ndimage import shift

from photutils import DAOStarFinder

from mylib.utils import galfit
from mylib.utils.cutout import PS1_cutout_downloader

#from lensqso.forcephot.psf_extract import PSFimage, PSFmodel

def cut_image(full_rms_image, radius, pixscale, out_rms_image,\
                 **kwargs):
    hdu = fits.open(full_rms_image)[0]
    wcs = WCS(hdu.header)

    if 'position' in kwargs.keys():
        position = kwargs['position']

    else:
        shape = hdu.data.shape
        position = (int(shape[0]/2), int(shape[1]/2))

    size = radius / pixscale * 2

    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    hdu.writeto(out_rms_image, overwrite=True)


class OneImage(object):
    def __init__(self, imagefilename, hduindex=0, pixelscale=None,\
                **kwargs):
        self.imagename = imagefilename

        hdulist = fits.open(self.imagename)
        header = hdulist[hduindex].header
        data = hdulist[hduindex].data

        self.exptime = header['EXPTIME']

        self.shape = data.shape
        self.xc = (self.shape[0]-1) / 2
        self.yc = (self.shape[1]-1) / 2

        if 'pixelscale' in kwargs.keys():
            self.pixscale=pixelscale
        else:
            try:
                self.pixscale =\
                    np.sqrt(header['CD1_1']**2 + header['CD1_2']**2) * 3600
            except KeyError:
                self.pixscale = 0.25

    def set_psf(self, psffilename):
        self.psfname = psffilename

    def set_galfit_basic(self, **kwargs):
        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        if 'zp' in kwargs.keys():
            zp = kwargs['zp']
        else:
            zp = 25.0

        self.zp = zp

        if 'fitted' in kwargs.keys():
            fitted = kwargs['fitted']
        else:
            fitted = imagename[:-5] + '.fit'

        self.fitted = fitted

        if 'rmsimage' in kwargs.keys():
            self.rmsimage = kwargs['rmsimage']
        else:
            self.rmsimage = self.imagename[:-5] + '.rms.fits'

        if 'radius' in kwargs.keys() and kwargs['radius']>0:
            radius_pix = kwargs['radius'] / self.pixscale
            print('radius pix', radius_pix)
            region = [int(self.xc-radius_pix+1), int(self.xc+radius_pix+1),\
                     int(self.yc-radius_pix+1), int(self.yc+radius_pix+1)]
        else:
            region = [1, 1+self.shape[0], 1, 1+self.shape[1]]
        self.region = region

        # set galfit variable

        gf = galfit.galfit(imagename, fitted, direct+'/')
        gf.noise = os.path.basename(self.rmsimage)
        gf.setpsf(os.path.basename(self.psfname))
        gf.setplatescale(self.pixscale)
        gf.setzeropoint(self.zp)
        gf.setboxsize(100)
        gf.psfovrsamp = 1

        # set fitting region #
        gf.setimageregion((region[0], region[1], region[2], region[3]))

        return gf

    def fit_sersic(self, **kwargs):

        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        # previous working directory

        cwd = os.getcwd()

        # then move to the directory of the image

        os.chdir(direct)

        if 'radius' in kwargs.keys() and kwargs['radius']>0:
            radius_pix = kwargs['radius'] / self.pixscale
            gf = self.set_galfit_basic(**kwargs)
        else:
            radius = -1
            gf = self.set_galfit_basic(radius=radius, **kwargs)

        if 'bkg' in kwargs.keys():
            bkg = kwargs['bkg']
        else:
            bkg = 0

        # set constraints #
        os.system('rm %s/constrains'%direct)

        f=open(direct+'/constrains','w')
        string=\
'''
1    re    %.1f  to  %.1f
1    n     %.1f  to  %.1f
1    ar    %.1f  to  %.1f
'''%(1, 10, 0.5, 5, 0.5, 1)
        f.write(string)
        f.close()

        gf.setconstraints(direct+'/constrains')

        # add components: one Sersic, one sky.

        image = fits.open(self.imagename)[0].data
        xc, yc = image.shape[0]/2+3, image.shape[1]/2

#        print(self.region)
        cutout_image = image[self.region[0]:self.region[1],\
                             self.region[2]:self.region[3]]

        #print(np.sum(cutout_image))

        mag_init = -2.5 * np.log10(np.sum(cutout_image))\
                + self.zp + 2.5 * np.log10(self.exptime)

        sersic_init = {'Re': 5, 'n': 4, 'axis_ratio': 1, 'pa': 0}
        sersic_freelist = [True, True, True, True, True, True]

        gf.addobject(galfit.galfit_sersic(\
                xc, yc, mag_init, **sersic_init,\
                freelist=sersic_freelist))
        gf.addobject(galfit.galfit_sky(bkg, freelist=[False]))

        # fit
        gf.fit()
        result = galfit.read_fit_results(gf.output)

        # cleaning
        os.system('rm ' + direct + '/galfit.*')
        os.system('rm ' + direct + '/fit.log')
        os.chdir(cwd)
        del gf

        return result

    def fit_sersic_psf(self, sersic_params, psf_params_list, **kwargs):

        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        # previous working directory

        cwd = os.getcwd()

        # then move to the directory of the image

        os.chdir(direct)

        if 'radius' in kwargs.keys() and kwargs['radius']>0:
            radius_pix = kwargs['radius'] / self.pixscale
            gf = self.set_galfit_basic(**kwargs)
        else:
            radius = -1
            gf = self.set_galfit_basic(radius=radius, **kwargs)

        if 'bkg' in kwargs.keys():
            bkg = kwargs['bkg']
        else:
            bkg = 0

        # set constraints #
        # fixed galaxy parameters.
        # PSFs: can only vary x and y by less than 1 pixel
        os.system('rm %s/constrains'%direct)

        f = open(direct+'/constrains','w')
        string = ''

        for index in range(len(psf_params_list)):
            n_comp = index + 2
            xc = psf_params_list[index]['x']
            yc = psf_params_list[index]['y']

            string += ' %d x %f to %f\n'%(n_comp, xc-1, xc+1)
            string += ' %d y %f to %f\n'%(n_comp, yc-1, yc+1)

        f.write(string)
        f.close()

        gf.setconstraints(direct+'/constrains')

        # add components: one Sersic, several PSFsone sky.

        # Sersic

        sersic_input = {'x':sersic_params['position'][0],\
                        'y':sersic_params['position'][1],\
                        'mag':sersic_params['mag'],\
                        'Re': sersic_params['Re'],\
                        'n': sersic_params['n'],\
                        'axis_ratio': sersic_params['axis_ratio'],\
                        'pa': sersic_params['pa']}

        sersic_freelist = [False, True, False, False, False, False]

        gf.addobject(galfit.galfit_sersic(**sersic_input,\
                freelist=sersic_freelist))

        # PSF

        for index in range(len(psf_params_list)):
            gf.addobject(galfit.galfit_psf(**psf_params_list[index]))

        # sky

        gf.addobject(galfit.galfit_sky(bkg, freelist=[False]))

        # fit
        gf.fit()

        # read result
        result = galfit.read_fit_results(gf.output)

        return result

    def reconstruct_galaxy(self, sersic_params, output_name):

        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        # previous working directory

        cwd = os.getcwd()

        # then move to the directory of the image

        os.chdir(direct)

        gf = self.set_galfit_basic()

        # add components: one Sersic, one sky.

        sersic_input = {'x':sersic_params['position'][0],\
                        'y':sersic_params['position'][1],\
                        'mag':sersic_params['mag'],\
                        'Re': sersic_params['Re'],\
                        'n': sersic_params['n'],\
                        'axis_ratio': sersic_params['axis_ratio'],\
                        'pa': sersic_params['pa']}
        sersic_freelist = [False, False, False, False, False, False]

        gf.addobject(galfit.galfit_sersic(**sersic_input,\
                freelist=sersic_freelist))
        gf.addobject(galfit.galfit_sky(0, freelist=[False]))

        # fit
        gf.fit()

        # cleaning
        os.system('rm ' + direct + '/galfit.*')
        os.system('rm ' + direct + '/fit.log')

        os.chdir(cwd)
        os.system('cp %s %s'%(gf.output, output_name))

        del gf

        return output_name

    def subtract_galaxy(self, galimg, rmsimg):
        direct = os.path.dirname(os.path.realpath(self.imagename))

        image_hdu = fits.open(galimg)
#        image_hdu[0].writeto('./test.fits')

        rawimage = image_hdu[1].data
        galimage = image_hdu[2].data
        rms = fits.open(rmsimg)[0].data

        f_init = np.max(rawimage) / np.max(galimage)
        factor = f_init

        while True:
            res = rawimage -  galimage * factor
            if np.min(res) < np.min(rawimage) - np.min(rms)*0.2:
                factor = factor * 0.999
            else:
                break

        flux = np.sum(galimage) * factor

        image_hdu[1].data = res
        image_hdu[1].writeto(direct+'/subtracted.fits', overwrite=True)

        return {'residual': res, 'galaxy_flux': flux}

    def findpeaks(self, resimage, rmsimage):
        std = np.median(rmsimage)

        daofind = DAOStarFinder(fwhm=4.0, threshold=5.*std)
        sources = daofind(resimage)

        sourcelist = self._convert_daophot_to_galfit(sources)
        return sourcelist

    def _convert_daophot_to_galfit(self, sources):
        psf = fits.open(self.psfname)[0].data

        sourcelist = []

        for index in range(len(sources)):
            xsource = sources['xcentroid'][index] + 1
            ysource = sources['ycentroid'][index] + 1
            peak = sources['peak'][index]

            flux = peak / np.max(psf) * np.sum(psf)
            mag = -2.5 * np.log10(flux/self.exptime) + 25

            sourcelist.append({'x': xsource,\
                               'y': ysource,\
                               'mag': mag})

        return sourcelist

def ImageSet(object):
    def __init__(self, imagelist):
        self.imagelist = imagelist

def run_psf(psfimage):
    psfimage.run_sextractor()
    psfimage.run_psfex()

def download_images(ra, dec, direct, size=300):
    PS1_cutout_downloader.download_images(ra, dec, filters='grizy',\
                                          outdir=direct, size=size)

def do_one_object(ra, dec, direct, download=False, do_psf=True,\
                  radius=3, pixscale=0.25):
    pool = mp.Pool(mp.cpu_count())

    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    coordstr = coord.to_string('hmsdms', sep='', precision=2)
    objname = 'J' + coordstr.replace(' ', '')

    # download the images

    if download:
        download_images(ra, dec, direct, size=300)

    # generate PSF


    # cut small images to fit, then generate the PSF

    if do_psf:
        psfimages = [PSFimage(direct + '/' + objname + '.%s.fits'%band,\
            direct + '/' + objname + '.%s.cat.fits'%band) for band in 'grizy']

        pool.map(run_psf, psfimages)

        for band in 'grizy':
            cut_image(direct + '/' + objname + '.%s.fits'%band,\
                      radius, pixscale,\
                      direct + '/galfit' + objname[1:] + '.%s.fits'%band)
            cut_image(direct + '/' + objname + '.%s.rms.fits'%band,\
                      radius, pixscale,\
                      direct + '/galfit' + objname[1:] + '.%s.rms.fits'%band)

            psfmodel = PSFmodel(direct + '/' + objname +\
                                '.%s.cat.psf.fits'%band)
            psfmodel.save_psf(-1, -1, direct + '/%spsf.fits'%band)

    # fit g-band image

    gimage = OneImage(direct + '/galfit' + objname[1:] + '.g.fits',\
                    pixelscale=pixscale)

    gimage.set_psf(os.path.abspath(direct + '/gpsf.fits'))
    sersicparam = gimage.fit_sersic()

    # work on z-band image
    zimage = OneImage(direct + '/galfit' + objname[1:] + '.z.fits',\
            pixelscale=0.25)

    zimage.set_psf(os.path.abspath(direct + '/zpsf.fits'))
    zimage.reconstruct_galaxy(sersicparam['sersic1'], direct + '/zgalaxy.fits')

    zimage.subtract_galaxy(direct + '/zgalaxy.fits',\
                rmsimg=direct + '/' + objname + '.z.rms.fits')

    residual = fits.open(direct + '/subtracted.fits')[1].data
    rmsimage = fits.open(direct + '/galfit' + objname[1:] +\
                         '.z.rms.fits')[0].data

    sources = zimage.findpeaks(residual, rmsimage)

    zimage.fit_sersic_psf(sersicparam['sersic1'], sources, radius=3)

    # work on r-band image
    rimage = OneImage(direct + '/galfit' + objname[1:] + '.r.fits',\
            pixelscale=0.25)

    rimage.set_psf(os.path.abspath(direct + '/rpsf.fits'))

def psf_strength(image, psf, pixscale=0.25):
    fftimage = fft2(image)
    fftpsf = fft2(psf)

    fftimage_amp = np.abs(fftimage)

    xsize, ysize = fftimage.shape
    xcoord = np.arange(0, xsize, 1)
    ycoord = np.arange(0, xsize, 1)

    xv, yv = np.meshgrid(xcoord, ycoord)
    dist2 = (xv - xsize/2)**2 + (yv - ysize/2)**2
    mask = (dist>(2/pixscale))

def test():

    rawimage = fits.open('./test2/galfit145211.50+422429.59.z.fits')[0].data
    galimage = fits.open('./test2/galfit145211.50+422429.59.r.fits')[0].data
    rms = fits.open('./test2/galfit145211.50+422429.59.i.rms.fits')[0].data

    f_init = np.nanmax(rawimage) / np.nanmax(galimage)

    factor = f_init

    while True:
        res = rawimage -  galimage * factor
        if np.min(res) < np.min(rawimage) - np.median(rms):
            factor = factor * 0.99
        else:
            break
    res = rawimage - np.rot90(rawimage, 2)

    flux = np.sum(galimage) * factor
    print(factor)

    daofind = DAOStarFinder(fwhm=5.0, threshold=5.*np.median(rms))
    sources = daofind(res)
    print(sources)

    phdu = fits.PrimaryHDU(data=res)
    phdu.writeto('./test2/subtracted.fits', overwrite=True)

def FWHM_to_sigma(k):
    f_to_solve = lambda x: k * x + x ** 1.67 - 1
    result = fsolve(f_to_solve, 1)

    factor = 1/np.sqrt(2 * result)/2
    return factor

def construct_PSF(FWHM_A, FWHM_B, theta, k, size, pixscale):
    # FWHM_A: the psfMajorFWHM value of PS catalog, arcsec
    # FWHM_B: the psfMinorFWHM value of PS catalog, arcsec
    # theta: the psfTheta value of PS catalog, degree
    # k: the psfCore value of PS catalog, dimensionless

    # FWHM to sigma factor
    factor = FWHM_to_sigma(k)
    sigmaA = factor * FWHM_A
    sigmaB = factor * FWHM_B

    # arcsec to pixel
    sigmaA = sigmaA / pixscale
    sigmaB = sigmaB / pixscale

    theta_rad = theta * np.pi / 180.
    sigma_xx_m2 = sigmaA**-2 * np.cos(theta_rad)**2 \
            + sigmaB**-2 * np.sin(theta_rad)**2
    sigma_yy_m2 = sigmaB**-2 * np.cos(theta_rad)**2 \
            + sigmaA**-2 * np.sin(theta_rad)**2
    sigma_xy = 0.5 * np.sin(2*theta_rad) * (sigmaB**-2 - sigmaA**-2)

    z_xy = lambda x, y: x**2*sigma_xx_m2/2 + y**2*sigma_yy_m2/2 + \
            sigma_xy*x*y

    psf_zk = lambda z, k: 1/(1+k*z+z**1.67)

    ### generate coordinate grid ###
    xlist = np.arange(0, size, 1) - (size-1)/2
    ylist = np.arange(0, size, 1) - (size-1)/2

    X, Y = np.meshgrid(xlist, ylist)

    Z = z_xy(X, Y)
    PSF = psf_zk(Z, k)

    return PSF

def generate_fft_mask(size, radius):
    xlist = np.arange(0, size, 1)
    ylist = np.arange(0, size, 1)

    X, Y = np.meshgrid(xlist, ylist)

    dist1 = X**2 + Y**2
    dist2 = (X-size+1)**2 + Y**2
    dist3 = (X-size+1)**2 + (Y-size+1)**2
    dist4 = X**2 + (Y-size+1)**2

    good = (dist1<radius**2)|(dist2<radius**2)|(dist3<radius**2)|(dist4<radius**2)
    return good

def main():
    '''
    time0 = time.time()

    ra, dec = 146.52017, 18.59453

    pool = mp.Pool(mp.cpu_count())

    # step 1: download images
    if 1:
        PS1_cutout_downloader.download_images(ra, dec, filters='grizy',\
                                              outdir='./test', size=300)

    # step 2: generate the PSF model
    if 1:
        psfimages = [PSFimage('./test/J094604.84+183540.31.%s.fits'%band,\
            './test/J094604.84+183540.31.%s.cat.fits'%band) for band in 'grizy']

        pool.map(run_psf, psfimages)

        for band in 'grizy':
            cut_image('./test/J094604.84+183540.31.%s.fits'%band, 3, 0.25,\
                          './test/galfit094604.84+183540.31.%s.fits'%band)
            cut_image('./test/J094604.84+183540.31.%s.rms.fits'%band, 3, 0.25,\
                          './test/galfit094604.84+183540.31.%s.rms.fits'%band)

    # step 3: read PSF model

    if 1:
        psfmodels = PSFmodel('./test/J094604.84+183540.31.g.cat.psf.fits')
        psfmodels.save_psf(-1, -1, './test/gpsf.fits')

    # step 4: fit a galaxy to the g-band image

    if 1:
        gimage = OneImage(\
                    os.path.abspath('./test/galfit094604.84+183540.31.g.fits'),\
                    pixelscale=0.25)

        gimage.set_psf(os.path.abspath('./test/gpsf.fits'))
        sersicparam = gimage.fit_sersic(radius=3)

    # step 5: generate z band PSFmodel
    if 0:
        zpsfimage = PSFimage('./test/J094604.84+183540.31.z.fits',\
                            './test/J094604.84+183540.31.z.cat.fits')
        zpsfimage.run_sextractor()
        zpsfimage.run_psfex()


    # step 6: re-construct the galaxy image in z band
    if 1:
        zpsfmodel = PSFmodel('./test/J094604.84+183540.31.z.cat.psf.fits')
        zpsfmodel.save_psf(-1, -1, './test/zpsf.fits')

        zimage = OneImage(\
                    os.path.abspath('./test/galfit094604.84+183540.31.z.fits'),\
                    pixelscale=0.25)

        zimage.set_psf(os.path.abspath('./test/zpsf.fits'))

        zimage.reconstruct_galaxy(sersicparam['sersic1'], './test/zgalaxy.fits')

    # step 7: subtract the galaxy from the image
    if 1:
        zimage.subtract_galaxy('./test/zgalaxy.fits', rmsimg='./test/J094604.84+183540.31.z.rms.fits')

    # step 8: find peaks in the residual image
    if 1:
        residual = fits.open('./test/subtracted.fits')[1].data
        rmsimage = fits.open('./test/galfit094604.84+183540.31.z.rms.fits')[0].data

        sources = zimage.findpeaks(residual, rmsimage)
        print(sources)

    # step 9: do another fit
    if 1:
        zimage.fit_sersic_psf(sersicparam['sersic1'], sources, radius=3)

    time1 = time.time()
    print(time1 - time0)
    '''
#    ra, dec = 223.04792, 42.40822
#    PS1_cutout_downloader.download_images(ra=ra, dec=dec, filters='gry', format='jpg', color=True)

#    do_one_object(ra, dec, os.path.abspath('./test2'), download=False, do_psf=False)
#    iimage = OneImage('./test2/galfit145211.50+422429.59.i.fits',\
#            pixelscale=0.25)
#    iimage.subtract_galaxy('./test2/galfit145211.50+422429.59.r.fits',
#                           './test2/galfit145211.50+422429.59.i.rms.fits')
#    test()

    psf = construct_PSF(1.0673500299453735,\
                        1.028439998626709,\
                        -7.811769962310791,\
                        0.4476209878921509, size=24, pixscale=0.25)
    fftpsf = fft2(psf)

    image = np.array(fits.open('./test/galfit094604.84+183540.31.z.fits')[0].data, dtype=float)
    fftimg = fft2(image)

    fft_deconv_img = fftimg / fftpsf
    mask = (np.abs(fft_deconv_img)>1e10)|(np.isnan(fft_deconv_img))
    fft_deconv_img[mask] = 0

    mask2 = generate_fft_mask(24, 5)
    fft_deconv_img[~mask2] = 0

    deconv_img = ifft2(fft_deconv_img)
    deconv_img = np.roll(deconv_img, 12, axis=0)
    deconv_img = np.roll(deconv_img, 12, axis=1)

    plt.imshow(np.abs(deconv_img))
    plt.show()


if __name__ == '__main__':
    main()
