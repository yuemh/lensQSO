import numpy as np
from astropy.io import fits
import surveys
import sim_image_glafic_ver2 as sim
import os
from __init__ import *

def luminosity_func():
    do_something = 1

def sim_a_lens(lenssystem, survey, filt, psf_file,\
               prefix='lens', imgsize=10, output='mock.fits',\
               noisy=True, noisy_output='noisymock.fits', clean=True):

    background = survey.getbackground(filt)
    pixscale = survey.pixscale
    zp = survey.zeropoint[filt]

    lenssystem.sim_image(imgsize, pixscale, psf_file, output, sky=background,\
                        zeropoint=zp)

    hdulist = fits.open(output)
    rawimg = hdulist[0].data
    noisyimg, noise = survey.noisyimg(filt, rawimg)
    noisyhdu = fits.ImageHDU(noisyimg, name='NOISY')
    hdulist.append(noisyhdu)
    sigmahdu = fits.ImageHDU(noise, name='SIGMA')
    hdulist.append(sigmahdu)

    hdulist.writeto(noisy_output, overwrite=True)

    if clean:
        os.system('rm ' + output)

def main():

    sourceinfo = [{'Type':'point', 'x_center':0.3, 'y_center':0.4}]
    lensinfo = [{'Type':'sersic', 'x_center':0, 'y_center':0,\
                 'ellipticity':0.3, 'pa':30., 're':0.5, 'n':1}]
    lensmassinfo = [{'Type':'sie', 'sigma':200, 'x_center':0, 'y_center':0,\
                     'ellipticity':0.3, 'pa':30., 'r_core':1e-2}]

    emptylens = False
    if emptylens:
        lensinfo = []
        lensmassinfo = []

    PSsurvey = surveys.PanStarrsSurvey()
    filt = 'PSi'
    source_z = 4
    lens_z = 0.5

    lens = sim.OneObject(lens_z, mass_component=lensmassinfo,\
                         light_component=lensinfo)
    source = sim.OneObject(source_z, light_component=sourceinfo)
    lenssystem = sim.LensSystem(lens, source)

    lenssystem.update_lensmag(22)
    lenssystem.update_sourcemag(21)

    sim_a_lens(lenssystem, PSsurvey, filt, dir_code+'/lens_psf.fits')

if __name__=='__main__':
    main()
