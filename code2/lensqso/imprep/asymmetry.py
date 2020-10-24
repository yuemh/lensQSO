import os, sys
import numpy as np

### let's try astropy Cutout2D first; if it does interpolation, it is good.

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table

from scipy import ndimage
from scipy.interpolate import interp2d

from mylib.utils.cutout import PS1_cutout_downloader

class ImageObject(object):
    def __init__(self, filename, extension):
        self.data = fits.open(filename)[extension].data
        self.center = ndimage.measurements.center_of_mass(self.data)

    def cut_center(self, x, y, size):
        cutout = Cutout2D(self.data, [x,y], size)
        return cutout

    def bkg_sigma(self):
        masked = sigma_clip(self.data)
        std = np.std(masked)
        return std

    def resample_recenter(self):
        ### new grid ###
        x = np.arange(self.data.shape[1])
        y = np.arange(self.data.shape[0])
#        X, Y = np.meshgrid(x, y)

        imgcenter_x = (self.data.shape[1]-1)/2
        imgcenter_y = (self.data.shape[0]-1)/2

        print(imgcenter_x, imgcenter_y)

        newx = x + self.center[1] - imgcenter_x
        newy = y + self.center[0] - imgcenter_y

        func = interp2d(x, y, self.data, kind='linear')
        newdata = func(newx, newy)

#        print(ndimage.measurements.center_of_mass(newdata))

        return newdata

def sigma_mask(data, sigma, n_sigma=2):
    return (data > sigma * n_sigma)

def asymmetry(data, mask, weight=1):
    data2 = np.rot90(data, 2)
    diff = data2 - data

    absdiff = np.abs(diff)

    flux = np.sum(data[mask])
    fluxdiff = np.sum(absdiff[mask])

    asym = fluxdiff / flux

    return asym

def test():
    ### coordinates ###
    ra, dec = 223.04792, 42.40822

    cutout_dir = os.path.abspath('./test')
    if not os.path.exists(cutout_dir):
        os.system('mkdir %s'%cutout_dir)

#    PS1_cutout_downloader.download_images(ra=ra, dec=dec, filters='z', size=6)

    img = ImageObject('./test/J145211.50+422429.59.z.fits',0)
    sigma = img.bkg_sigma()
    data = img.resample_recenter()
    print(img.center)

    hdu = fits.PrimaryHDU(data=data)
    hdu.writeto('./test/testcut.fits', overwrite=True)

    mask = sigma_mask(data, sigma, 2)
    print(asymmetry(data, mask))

def test_comparison():
    tbl = Table.read('./MyTable_zhenliyi.fit')

    indice = np.random.choice(range(len(tbl)), 1000)

    z_asym = []

    for index in indice:
        ra = tbl[index]['raStack']
        dec = tbl[index]['decStack']
#        PS1_cutout_downloader.download_images(ra=ra, dec=dec, filters='z', size=6,\
#                                             outdir='./test_comparison')

        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        coordstr = coord.to_string('hmsdms', sep='', precision=2)
        filename = './test_comparison/J' + coordstr + '.z.fits'

        if not os.path.exists(filename):
            continue

        img = ImageObject(filename, 0)
        sigma = img.bkg_sigma()
        data = img.resample_recenter()

#        hdu = fits.PrimaryHDU(data=data)
#        hdu.writeto('./test/testcut.fits', overwrite=True)

        mask = sigma_mask(data, sigma, 2)
        A = asymmetry(data, mask)
        z_asym.append(A)


def main():
    test()

if __name__=='__main__':
    main()


