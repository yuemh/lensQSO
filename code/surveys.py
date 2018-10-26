import numpy as np
import mylib.spec_measurement as spec
import astropy.units as u
import astropy.constants as const
from errors import *

def mag_to_counts(mag, Filter, Aeff, texp):
    '''
    Parameters
    ---------_
    Filter: mylib.spec_measurement.Filter

    Aeff: float
        Effective area in m^2

    texp: float
        Exposure time in seconds
    '''

    zp = 3631 * u.Jy
    flux = 10 ** (-0.4 * mag) * zp
    counts_unfiltered = float(flux * Aeff * texp * u.m**2 * u.s / const.h)

    logwave = np.log10(Filter.wavelength)
    logint_filt = np.trapz(Filter.value, logwave)

    return float(counts_unfiltered * logint_filt)

def skybright_to_background(skybright, Filter, Aeff, texp, pixscale):
    mag_per_pix = skybright + 2.5 * np.log10(1/pixcale**2)
    skycount = mag_to_counts(mag_per_pix)
    return float(skycount)

def maglim_to_background(maglim, Filter, Aeff, texp, FWHM, pixscale):
    N_pix = np.pi * (FWHM/2)**2 / pixscale**2
    counts = mag_to_counts(maglim, Filter, Aeff, texp)
    sigma = counts / 5
    bkg = sigma ** 2 / N_pix

    return float(bkg)

class _survey(object):
    def __init__(self):
        # get the filters and depth of the survey #
        self.getinfo()

    def getbackground(self, filt):
        if not filt in self.filters:
            raise UnknownFilterError('Survey %s does not have filter %s'\
                                    %(self.name, filt))
        maglim = self.maglim[filt]
        FWHM = self.FWHM[filt]
        texp = self.texp[filt] * self.visits[filt]

        filtobj = spec.read_filter(self.filter_key, filt)

        bkg = maglim_to_background(maglim, filtobj, self.Aeff,\
                                   texp, FWHM, self.pixscale)
        return bkg

    def getcounts(self, filt, mag):
        if not filt in self.filters:
            raise UnknownFilterError('Survey %s does not have filter %s'\
                                    %(self.name, filt))
        filtobj = spec.read_filter(self.filter_key, filt)
        counts = mag_to_counts(mag, filtobj, self.Aeff, self.texp[filt])
        return counts

    def noisyimg(self, filt, image):
        RN = self.readnoise
        nexp = self.visits[filt]

        noisyimg = np.random.poisson(image)\
                + np.random.normal(scale=RN*np.sqrt(nexp))
        return noisyimg


class PanStarrsSurvey(_survey):
    def getinfo(self):
        self.name='Pan-Starrs'
        self.Aeff = 2.54
        self.pixscale = 0.25
        self.readnoise = 5
        self.filter_key = 'Pan-Starrs'
        self.filters = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']
        self.maglim = {'PSg':23.3, 'PSr':23.2, 'PSi':23.1,\
                       'PSz':22.3, 'PSy':21.3}
        self.FWHM = {'PSg':1.31, 'PSr':1.19, 'PSi':1.11,\
                     'PSz':1.07, 'PSy':1.02}
        self.texp = {'PSg':43, 'PSr':40, 'PSi':45,\
                       'PSz':30, 'PSy':30}
        self.visits = {'PSg':10, 'PSr':12, 'PSi':17,\
                       'PSz':11, 'PSy':12}


class WISESurvey(_survey):
    def getinfo(self):
        self.name = 'WISE'


class TwoMASSSurvey(_survey):
    def getinfo(self):
        self.name = '2MASS'


class VISTASurvey(_survey):
    def getinfo(self):
        self.name = 'VISTA'
