import numpy as np
from astropy.io import fits
import surveys
import sim_image_glafic as sim

def luminosity_func():
    do_something = 1

def sim_a_lens(source_z, sourceinfo, lens_z, lensinfo, lensmassinfo,\
               survey, filt, prefix='lens', imgsize=10, output='mock.fits',\
               noisy=True, noisy_output='noisymock.fits', clean=False,\
               **kwargs):
    """
    Parameters
    ----------
    sourceinfo: list of dict
        The information of the source. One point source by default. 
        Elements hould have the following keys: Type, Mag, x_center, y_center,
        and any other keys needed to describe the source.

    lensinfo: list of dict
        The information of the lensing galaxy. One Sersic profile by default.
        Elements should have the following keys: Type, Mag, x_center, y_center,
        and any other keys needed to describe the source.

    lensmassinfo: list of dict
        The information of the lensing galaxy. Sie profile by default. 
        Elements should have the following keys: Type, masskey,\
        x_center, y_center, and any other keys needed to describe the source.

    survey: surveys._survey

    filt: str
    """
    if not 'psfinfo' in kwargs.keys():
        psf1 = [survey.FWHM[filt], 0, 0, 3]
        psfinfo = [psf1, psf1, 1]

    for source in sourceinfo:
        sourcemag = source['Mag']
        sourceflux = survey.getcounts(filt, sourcemag)
        print(sourceflux)
        source['flux'] = sourceflux

    for lenslight in lensinfo:
        lensmag = lenslight['Mag']
        lensflux = survey.getcounts(filt, lensmag)
        lenslight['flux'] = lensflux

    background = survey.getbackground(filt)
    pixscale = survey.pixscale

    sourceobj = sim.OneObject(source_z, light_component=sourceinfo)
    lensobj = sim.OneObject(lens_z, mass_component=lensmassinfo,\
                            light_component=lensinfo)

    lenssystem = sim.LensSystem(lensobj, sourceobj)
    lenssystem.sim_image(prefix=prefix, x_range=[-imgsize/2, imgsize/2],\
                         y_range=[-imgsize/2, imgsize/2], psfinfo=psfinfo,\
                         pix_ext=pixscale, pix_poi=pixscale,\
                         output=output, psfconv_size=imgsize/5, \
                         background=background)

    hdulist = fits.open(output)
    rawimg = hdulist[3].data
    noisyimg = survey.noisyimg(filt, rawimg)
    noisyhdu = fits.ImageHDU(noisyimg, name='NOISY')
    hdulist.append(noisyhdu)

    hdulist.writeto(noisy_output, overwrite=True)

    if clean:
        os.system('rm ' + output)

def main():

    sourceinfo = [{'Type':'point', 'Mag':22.3, 'x_center':0., 'y_center':0.}]
    lensinfo = [{'Type':'sersic', 'Mag':30, 'x_center':0, 'y_center':0,\
                 'ellipticity':0.3, 'pa':30., 're':0.5, 'n':1}]
    lensmassinfo = [{'Type':'sie', 'sigma':200, 'x_center':0, 'y_center':0,\
                     'ellipticity':0.3, 'pa':30., 'r_core':1e-2}]

    emptylens = True
    if emptylens:
#        lensinfo = []
        lensmassinfo = []

    PSsurvey = surveys.PanStarrsSurvey()
    filt = 'PSi'
    source_z = 4
    lens_z = 0.5
    sim_a_lens(source_z, sourceinfo, lens_z, lensinfo, lensmassinfo,\
               PSsurvey, filt)

if __name__=='__main__':
    main()
