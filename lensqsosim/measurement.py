import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, interpolate
from scipy.optimize import minimize
from lenspop import *
from surveys import *


def aperture_mask(image, xc, yc, aper_radius):
    xlist = np.arange(image.shape[1])
    ylist = np.arange(image.shape[0])

    XX, YY = np.meshgrid(xlist, ylist)
    mask = ((XX - xc)**2 + (YY-yc)**2 < aper_radius**2)

    return mask

def psf_residual(psf_paras, image_paras):

    xc, yc, amp = psf_paras
    image, noise, psf, mask = image_paras
    psf = np.array(psf)
    xsize_img, ysize_img = image.shape
    xsize_psf, ysize_psf = psf.shape
    xc_img, yc_img = np.array(image.shape) / 2
    xc_psf, yc_psf = np.array(psf.shape) / 2

    newgrid_x = np.linspace(xc - xsize_psf/2.0, xc + xsize_psf/2.0, xsize_psf)
    newgrid_y = np.linspace(yc - ysize_psf/2.0, yc + ysize_psf/2.0, ysize_psf)

    f = interpolate.interp2d(newgrid_x, newgrid_y, psf)
    newpsf = f(np.arange(xsize_img), np.arange(ysize_img))

    res = image - newpsf*amp
    chi2 = (res / noise)**2

    return np.sum(chi2[mask])

class LensImage(object):
    def __init__(self, image_array, noise_array, bkg=-1):
        if bkg>0:
            self.image = image_array.copy() - bkg
        else:
            self.image = image_array.copy() - np.median(image_array)
        self.noise = noise_array

    def aperflux(self, aper_radius):
        xc, yc = self.center
        mask = aperture_mask(self.image, xc, yc, aper_radius)

        flux = np.sum(self.image[mask])
        sigma = np.sqrt(np.sum(self.noise[mask]**2))
        return (flux, sigma)

    def apermag(self, aper, zp):
        flux = self.aperflux(aper)
        return -2.5 * np.log10(flux) + zp

    def interpolate(self, center):
        xc, yc = center
        xlist = np.arange(self.image.shape[1])
        ylist = np.arange(self.image.shape[0])

        Y, X = self.image.shape
        xc0, yc0 = X/2., Y/2.

        newx = xlist + xc - xc0
        newy = ylist + yc - yc0

        newimgfunc = interpolate.interp2d(xlist, ylist, self.image)
        newimage = newimgfunc(newx, newy)

        return newimage

    def psffit(self, psf, aper_radius=-1):
        xc, yc = self.center
        Y, X = self.image.shape
        if aper_radius>0:
            mask = aperture_mask(self.image, xc, yc, aper_radius)
        else:
            mask = np.full(self.image.shape, True)

        init = [xc, yc, 1]
        bounds = [[0, X],[0, Y],[0, None]]
        result = minimize(psf_residual, x0=init,\
                         args=[self.image, self.noise, psf, mask],\
                         bounds=bounds, method='SLSQP')

        return result

    def asymmetry(self, aper_radius):
        newimage = self.interpolate(self.center)
        newimage_rot = np.rot90(newimage, 2)
        plt.imshow(self.image, origin='lower')
        plt.show()
        plt.imshow(newimage, origin='lower')
        plt.show()
        plt.imshow(newimage_rot, origin='lower')
        plt.show()
        plt.imshow(newimage_rot-newimage, origin='lower')
        plt.show()

        xc, yc = np.array(newimage.shape) / 2
        mask = aperture_mask(newimage, xc, yc, aper_radius)

        diff = newimage - newimage_rot

        asymmetry = np.sum(np.abs(diff[mask]))\
                /np.sum(newimage[mask])
        return asymmetry

    @property
    def center(self):
#        print(self.image)
        center = ndimage.measurements.center_of_mass(self.image)
        center = np.flip(center, 0)
        return center


def main():
#    PSsurvey = PanStarrsSurvey()
    image = fits.open('./noisymock.fits')[4].data
    sigma = fits.open('./noisymock.fits')[5].data
    psf = fits.open('./lens_psf.fits')[0].data

    imageobj = LensImage(image, sigma, bkg=9416.93)
#    print(imageobj.center)
#    print(imageobj.aperflux(10))
#    print(imageobj.asymmetry(10))

    xc, yc = imageobj.center

    print(imageobj.psffit(psf))

if __name__=='__main__':
    main()
