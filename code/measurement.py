import numpy as np
from scipy import ndimage, interpolate
from lenspop import *
from surveys import *


def aperture_mask(image, xc, yc, aper_radius):
    xlist = np.arange(image.shape[1])
    ylist = np.arange(image.shape[0])

    XX, YY = np.meshgrid(xlist, ylist)
    mask = ((XX - xc)**2 + (YY-yc)**2 < aper_radius**2)

    return mask

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

    def interp_to_center(self):
        xc, yc = self.center
        xlist = np.arange(self.image.shape[1])
        ylist = np.arange(self.image.shape[0])

        newx = xlist + xc - int(xc)
        newy = ylist + yc - int(yc)

        newimgfunc = interpolate.interp2d(xlist, ylist, self.image)
        newimage = newimgfunc(newx, newy)

        return newimage

    def psfflux(self):
        do_something = 1

    def asymmetry(self, aper_radius):
        newimage = self.interp_to_center()
        newimage_rot = np.rot90(newimage, 2)

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
        return center


def main():
#    PSsurvey = PanStarrsSurvey()
    image = fits.open('./noisymock.fits')[4].data
    sigma = fits.open('./noisymock.fits')[5].data

    imageobj = LensImage(image, sigma, bkg=9416.93)
    print(imageobj.center)
    print(imageobj.aperflux(10))
    print(imageobj.asymmetry(10))

if __name__=='__main__':
    main()
