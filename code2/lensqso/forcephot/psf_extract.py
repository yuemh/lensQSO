import os, sys
import numpy as np

from astropy.io import fits

from mylib.utils import cutout, galfit

dir_data = os.path.abspath(os.getcwd()+'/../../../data/')
dir_config = os.path.abspath(os.getcwd()+'/../config/')

sex_config = dir_config + '/psf.sex'
sex_param = dir_config + '/psf.param'
psfex_config = dir_config + '/default.psfex'

def generate_psfex_config(filename, **kwargs):
    dummy = 1

class PSFimage(object):
    def __init__(self, imagefilename, catfilename):
        self.imagename = imagefilename
        self.catname = catfilename
        self.rmsname = imagefilename[:-5] + '.rms.fits'

    def run_sextractor(self, config=sex_config, param=sex_param):
        command = 'sextractor ' + self.imagename + ' -c ' + sex_config\
                  + ' -CATALOG_NAME ' + self.catname\
                  + ' -PARAMETERS_NAME ' + sex_param\
                  + ' -CHECKIMAGE_TYPE BACKGROUND_RMS'\
                  + ' -CHECKIMAGE_NAME ' + self.rmsname
        os.system(command)
        return self.catname

    def run_psfex(self):
        command = 'psfex ' + self.catname + ' -c ' + psfex_config\
                + ' -CHECKIMAGE_TYPE NONE ' \
                + ' -PSF_SUFFIX .psf.fits -CHECKPLOT_DEV NULL'
        os.system(command)
        return self.imagename + '.psf.fits'

class PSFmodel(object):
    def __init__(self, psffilename):
        self.psffilename = psffilename

        psfex_model = fits.open(self.psffilename)[1]
#        print(psfex_model.header)
        self.x0, self.y0, self.kx, self.ky = \
            psfex_model.header['POLZERO1'], psfex_model.header['POLZERO2'],\
            psfex_model.header['POLSCAL1'], psfex_model.header['POLSCAL2']

        self.header = psfex_model.header
        self.model = psfex_model.data

    def get_psf(self, x_image=-1, y_image=-1):

        if x_image<0:
            x_image = self.x0

        if y_image<0:
            y_image = self.y0

        x = (x_image - self.x0) / self.kx
        y = (y_image - self.y0) / self.ky

        psf = self.model[0][0][0]
        psf_corr = x*self.model[0][0][1] + y*self.model[0][0][2]

        #+\
        # y*x**2*self.model[0][0][6]+y**2*self.model[0][0][7]+\
        # x*y**2*self.model[0][0][8]+y**3*self.model[0][0][9]

        psf_total = psf + psf_corr
        return psf_total

    def save_psf(self, x_image, y_image, output):

        psf_image = self.get_psf(x_image, y_image)
        hdu = fits.PrimaryHDU(data=psf_image)
        hdu.writeto(output, overwrite=True)

class Objectimage(object):
    def __init__(self, imagefilename, hduindex=0):
        self.imagename = imagefilename

        hdulist = fits.open(self.imagename)
        header = hdulist[hduindex].header
        data = hdulist[hduindex].data

        self.shape = data.shape
        self.xc = (self.shape[0]-1) / 2
        self.yc = (self.shape[1]-1) / 2

#        self.pixscale = np.sqrt(header['CD1_1']**2 + header['CD1_2']**2) * 3600
        self.pixscale=0.25

    def set_psf(self, psffilename):
        self.psfname = psffilename

    def set_galfit_basic(self, **kwargs):
        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        if 'zp' in kwargs.keys():
            zp = kwargs['zp']
        else:
            zp = 25.0

        if 'fitted' in kwargs.keys():
            fitted = kwargs['fitted']
        else:
            fitted = imagename[:-5] + '.fit'

        if 'rmsimage' in kwargs.keys():
            rmsimage = kwargs['rmsimage']
        else:
            rmsimage = imagename[:-5] + '.rms.fits'

        if 'radius' in kwargs.keys() and kwargs['radius']>0:
            radius_pix = kwargs['radius'] / self.pixscale
            region = [self.xc-radius_pix+1, self.xc+radius_pix+1,\
                     self.yc-radius_pix+1, self.yc+radius_pix+1]
        else:
            region = [1, 1+self.shape[0], 1, 1+self.shape[1]]

        # set galfit variable

        gf = galfit.galfit(imagename, fitted, direct+'/')
        gf.setconstraints('constrains')
        gf.noise = rmsimage
        gf.setpsf(self.psfname)
        gf.setplatescale(self.pixscale)
        gf.setzeropoint(zp)
        gf.setboxsize(400)
        gf.psfovrsamp = 1

        # set fitting region #
        gf.setimageregion((region[0], region[1], region[2], region[3]))

        return gf

    def fit(self, maglist, bkg=0, **kwargs):
        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        if 'radius' in kwargs.keys() and kwargs['radius']>0:
            radius_pix = kwargs['radius'] / self.pixscale
            gf = self.set_galfit_basic(**kwargs)
        else:
            radius = -1
            gf = self.set_galfit_basic(radius=radius, **kwargs)

        # set constraints #
        os.system('rm constrains')

        f=open(direct+'/constrains','w')
        string=\
'''
1    re    %.1f  to  %.1f
1    n     %.1f  to  %.1f
1    ar    %.1f  to  %.1f
'''%(2, 10, 1, 5, 0.6, 1)
        f.write(string)
        f.close()

        # add_components #
        n_component = len(maglist)

        if 'coordlist' in kwargs.keys()\
           and kwargs['coordlist'].shape==(n_component, 2):
            coordlist = np.array(kwargs['coordlist'])
        else:
            coordlist = np.tile([self.xc, self.yc],\
                            n_component).reshape([n_component, 2])

        xlist = coordlist[:, 0]
        ylist = coordlist[:, 1]

        sersic_params = {'Re': 5, 'n': 4, 'axis_ratio': 1, 'pa': 0}
        sersic_freelist = [True, True, True, True, True, True]

        gf = add_components(gf, xlist, ylist, maglist, bkg,\
                           sersic_params, sersic_freelist)

        # fit #
        gf.fit()
        result = saferead_galfit(gf)

        # cleaning
        os.system('rm ' + direct + '/galfit.*')
        os.system('rm ' + direct + '/fit.log')
        del gf

        return result

    def fit_random(self, mag, radius, niter=10, **kwargs):
        direct = os.path.dirname(os.path.realpath(self.imagename))
        imagename = os.path.basename(self.imagename)

        radius_pix = radius / self.pixscale
        xmin, xmax = [self.xc - radius_pix, self.xc + radius_pix]
        ymin, ymax = [self.yc - radius_pix, self.yc + radius_pix]
        mmin, mmax = [mag, mag + 2]

        results = []
        rchisqs = []

        for index in range(niter):
            xlist = (xmin + xmax) / 2 \
                    + np.random.normal(0, 0.05, 3) * (xmax - xmin) / 2
            ylist = (ymin + ymax) / 2 \
                    + np.random.normal(0, 0.05, 3) * (ymax - ymin) / 2
            maglist = mmin + np.random.rand(3) * (mmax - mmin)
            coordlist = np.transpose([xlist, ylist])

            success, result = self.fit(maglist=maglist, coordlist=coordlist,\
                                       radius=radius, **kwargs)
            if success:
                rchisq = result['chisqr']

                rchisqs.append(rchisq)
                results.append(result)

        bestindex = np.argmin(rchisqs)
        bestresult = results[bestindex]

        return bestresult

def saferead_galfit(gf):
    # read results
    if os.path.exists(gf.output):
        success = True
        params = galfit.read_fit_results(gf.output)
    else:
        success = False
        params = {}

    return [success, params]

def add_components(gf, xlist, ylist, maglist, bkg,\
                   sersic_params, sersic_freelist):

    # add objects #
    if 0:
        gf.addobject(galfit.galfit_sersic(\
                xlist[0], ylist[0], maglist[0], **sersic_params,\
                freelist=sersic_freelist))

    for index in range(0, len(xlist)):
        gf.addobject(galfit.galfit_psf(\
                    xlist[index], ylist[index], maglist[index],\
                    freelist=[True, True]))
    gf.addobject(galfit.galfit_sky(bkg, freelist=[False]))

    return gf

def test():
    # DECaLS image test: J0011-0845

    ra = 120.990643
    dec = 39.1397378

    band = 'z'
    '''
    cutout.PS1_cutout_downloader.download_images(ra, dec, 300,\
                                                filters=band,\
                                                outdir='./')

    imagename = './J0803+3908.%s.fits'%band
    catname = './J0803+3908.%scat.fits'%band

    psfimg = PSFimage(imagename, catname)
    psfimg.run_sextractor()
    psfimg.run_psfex()
    model = PSFmodel('./J0803+3908.%scat.psf.fits'%band)
    model.save_psf(-1, -1, './ps1psf.%s.fits'%band)

#    print(os.path.basename('./J001120.24-084551.48.zcat.fits'))
    '''
    image = Objectimage(imagefilename='./J0803+3908.%s.fits'%band)
    image.set_psf('./ps1psf.%s.fits'%band)
    image.set_galfit_basic()
    print(image.fit_random(mag=17, radius=3))

def main():
    test()

if __name__=='__main__':
    main()
