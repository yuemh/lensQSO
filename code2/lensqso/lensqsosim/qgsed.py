from __init__ import *
import numpy as np
import os
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager, rcParams
import matplotlib.gridspec as gridspec
import mylib.spec_measurement as spec
import json
from astropy.table import Table

from lenspop import sim_a_lens
import surveys

from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk, buildQsoSpectrum
from simqso import hiforest
from simqso import sqphoto
from simqso import sqmodels

fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
prop = font_manager.FontProperties(fname=fontpath)
rcParams['font.family'] = prop.get_name()

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)
np.random.seed(0)

def moffat(x, y, paras):
    FWHM, beta = paras
    alpha = FWHM / 2 / np.sqrt(2**(1.0/beta)-1)

    return (1 + (x**2 + y**2)/alpha**2)**(-beta)

def generate_psf(func, paras, size, pixscale):
    size_pix = int(size / pixscale)

    x = (np.arange(size_pix) - (size_pix-1) / 2) * pixscale
    y = (np.arange(size_pix) - (size_pix-1) / 2)  * pixscale

    xx, yy = np.meshgrid(x, y)
    psf = func(xx, yy, paras)
    return psf

def FP_sigma(mag, re, redshift, kcorr, kpcsep):
    mu = mag + 2.5 * np.log10 (2 * np.pi * re**2)\
            - 10 * np.log10(1+redshift) - kcorr
    logRe = np.log10(re / kpcsep / 0.7)

    a, b, c = 1.52, -0.78, -8.895
    logsigma = (logRe + b*mu/2.5 - c)/a

    return float(10**logsigma)

def MStar_sigma(mstar, re, n):
    '''
    Bezanson et al. 2011
    '''
    Kv = 73.32 / (10.465 + (n-0.94)**2) + 0.954
    Ks = 0.557 * Kv
    G = const.G
    Ms = 10**mstar*const.M_sun
    re_phys = re * u.kpc
    sigma = np.sqrt(G * Ms / Ks / re_phys)
    sigma_num = sigma.to('km/s').value

    return sigma_num

def quasarsample_ian(redshift, Mabs, **kwargs):

    obswave = sqbase.fixed_R_dispersion(1000, 5e4, 1000)

    # cover 3000A to 5um at R=500
    forestModel = sqmodels.forestModels['McGreer+2013']
    forestSpec = hiforest.IGMTransmissionGrid(obswave, forestModel,\
                                              1, zmax=redshift)
    forestVar = HIAbsorptionVar(forestSpec)

    # Basic setting 

    M = AbsMagVar(FixedSampler([Mabs]), restWave=1450)
    z = RedshiftVar(FixedSampler([redshift]))
    qsos = QsoSimPoints([M, z], cosmo=defaultcosmo, units='luminosity')

    # use the canonical values for power law continuum slopes in FUV/NUV,
    # with breakpoint at 1215A
    contVar = BrokenPowerLawContinuumVar(\
            [GaussianSampler(-1.5,0.3), GaussianSampler(-0.5,0.3)], [1215.])

    # add two dust components as in Lyu+Rieke 2017,
    # but reduce the hot dust flux by a factor of 2
    subDustVar = DustBlackbodyVar([ConstSampler(0.05),ConstSampler(1800.)],\
                                  name='sublimdust')
    subDustVar.set_associated_var(contVar)
    hotDustVar = DustBlackbodyVar([ConstSampler(0.1),ConstSampler(880.)],\
                                  name='hotdust')
    hotDustVar.set_associated_var(contVar)

    # generate lines using the Baldwin Effect emission line model from BOSS DR9
    emLineVar = generateBEffEmissionLines(qsos.absMag)

    # the default iron template from Vestergaard & Wilkes 2001
    # was modified to fit BOSS spectra
    fescales = [(0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),\
                (1868,2140,1.0),(2140,3500,1.0)]
    feVar = FeTemplateVar(VW01FeTemplateGrid(qsos.z, obswave, scales=fescales))

    # Now add the features to the QSO grid
    qsos.addVars([contVar, subDustVar, hotDustVar, emLineVar,\
                  feVar, forestVar])

    # ready to generate spectra
    _, obsflux = buildSpectraBulk(obswave, qsos, saveSpectra=True)

    return [obsflux[0], obswave, redshift]

def save_galaxy_params(output_file):
    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'

    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_info_hdulist = fits.open(jaguar_info_masterfile)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data
    jaguar_info = jaguar_info_hdulist[1].data

    mask_gal = (jaguar_info['redshift']<1.5) \
            & (jaguar_info['redshift']>0.5)
#            & (jaguar_info['mStar']>11)
    print(len(jaguar_info[mask_gal]))

    redshiftlist = jaguar_info['redshift']

    PSg = spec.read_filter('Pan-Starrs', 'PSg')
    PSr = spec.read_filter('Pan-Starrs', 'PSr')
    PSi = spec.read_filter('Pan-Starrs', 'PSi')
    PSz = spec.read_filter('Pan-Starrs', 'PSz')
    PSy = spec.read_filter('Pan-Starrs', 'PSy')

    filters = [PSg, PSr, PSi, PSz, PSy]
    filternames = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']

    galaxy_list = []

    ID_list = []
    z_list = []
    re_list = []
    n_list = []
    pa_list = []
    ellip_list = []
    sigma_list = []
    mag_list = []

    for index in range(len(jaguar_info[mask_gal])):
        flux = jaguar_flux[mask_gal][index]
        wave = jaguar_wave
        redshift = redshiftlist[mask_gal][index]
        galobj = ObjectSED(wave, flux, redshift, 'OBS', 'lens')

        ID = float(jaguar_info['ID'][mask_gal][index])
        re = float(jaguar_info['Re_circ'][mask_gal][index])
        n = float(jaguar_info['sersic_n'][mask_gal][index])
        pa = float(jaguar_info['position_angle'][mask_gal][index])
        ellip = float(1 - jaguar_info['axis_ratio'][mask_gal][index])
        mstar = float(jaguar_info['mStar'][mask_gal][index])
        sigma = MStar_sigma(mstar, re, n)

        PSmag = [galobj.magnitude(f) for f in filters]

        ID_list.append(ID)
        z_list.append(redshift)
        re_list.append(re)
        n_list.append(n)
        pa_list.append(pa)
        ellip_list.append(ellip)
        sigma_list.append(sigma)
        mag_list.append(PSmag.copy())

    tbl = Table({'ID':ID_list, 'redshift':z_list, 'sersic_re':re_list,\
                 'sersic_index':n_list, 'position_angle':pa_list,\
                 'ellipticity':ellip_list, 'sigma':sigma_list,\
                 'mag':mag_list})
    tbl.write(output_file, overwrite=True)

def save_quasar_params(output_file):
    Mabs_list = np.arange(-29, -24, 1)
    redshift_list = np.arange(4.5, 5.5, 0.2)

    PSg = spec.read_filter('Pan-Starrs', 'PSg')
    PSr = spec.read_filter('Pan-Starrs', 'PSr')
    PSi = spec.read_filter('Pan-Starrs', 'PSi')
    PSz = spec.read_filter('Pan-Starrs', 'PSz')
    PSy = spec.read_filter('Pan-Starrs', 'PSy')

    filters = [PSg, PSr, PSi, PSz, PSy]
    filternames = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']

    ID = 0

    ID_list = []
    mag_list = []
    M_list = []
    z_list = []

    Nrand = 5

    for Mabs in Mabs_list:
        for redshift in redshift_list:
            for loop in range(Nrand):
                flux, wave, redshift = quasarsample_ian(redshift, Mabs)
                qsoobj = ObjectSED(wave, flux, redshift, 'OBS', 'source')
                PSmag = [qsoobj.magnitude(f) for f in filters]

                mag_list.append(PSmag)
                ID_list.append(ID)
                M_list.append(Mabs)
                z_list.append(redshift)

                ID += 1

    tbl = Table({'ID':ID_list, 'redshift':z_list, 'Mabs':M_list,\
                 'mag':mag_list})
    tbl.write(output_file, overwrite=True)


class ObjectSED(spec.Spectrum):
    def __init__(self, wavelength, flux, redshift, mode, objtype,\
                units=[u.Angstrom, u.erg/u.s/u.cm/u.cm/u.Angstrom]):

        super(ObjectSED, self).__init__(wavelength, flux,\
                                        units=units, mode=mode)
        self.redshift = redshift
        self.objtype = objtype
        self.to_obs(redshift)

        self.angu_dist =\
            float(defaultcosmo.angular_diameter_distance(self.redshift)/u.kpc)
        self.kpc_sep = 206265./self.angu_dist
        self.lumi_dist =\
            float(defaultcosmo.luminosity_distance(self.redshift)/u.kpc)

    def lightinfo(self, filt, x_center, y_center, **kwargs):
        self.to_obs(self.redshift)
        mag = self.magnitude(filt)
        if self.objtype == 'source':
            infodict = {'Type':'point', 'Mag':mag,\
                    'x_center':x_center, 'y_center':y_center}
        elif self.objtype == 'lens':
            re = kwargs['re'] * self.kpc_sep
            n = kwargs['n']
            ellip = kwargs['ellipticity']
            pa = kwargs['pa']

            infodict = {'Type':'sersic', 'Mag':mag, \
                        'x_center':x_center, 'y_center':y_center,\
                        're':re, 'n':n, 'ellipticity':ellip, 'pa':pa}

        return [infodict]

    def massinfo(self, x_center, y_center, **kwargs):
        self.to_obs(self.redshift)

        if self.objtype == 'source':
            return []

        elif self.objtype == 'lens':
            stdfilt = spec.read_filter('SDSS', 'SDSSi')
            mag = self.magnitude(stdfilt)
            re = kwargs['re'] * self.kpc_sep
            n = kwargs['n']
            ellip = kwargs['ellipticity']
            pa = kwargs['pa']

            sigma = FP_sigma(mag, re, self.redshift, 0, self.kpc_sep)

            infodict = {'Type':'sie', 'sigma':sigma, 'ellipticity':ellip,\
                        'pa':pa, 'x_center':x_center, 'y_center':y_center,\
                        'r_core':1e-3}

            return [infodict]

def plot_lens_system(lenssys, mockimages, filename):
    image_dat = lenssys.image_dat
    crit_dat = lenssys.findcrit()

    image_x, image_y, image_mu, image_dt = image_dat
    source_x = lenssys.source.Lightlist[0]['x_center']
    source_y = lenssys.source.Lightlist[0]['y_center']

    xi1, yi1, xs1, ys1, xi2, yi2, xs2, ys2 = crit_dat

    omages = lenssys.image_dat
    images = lenssys.image_dat

#    figure = plt.figure(constrained_layout=True)
#    gs = gridspec.GridSpec(2, 3)
#    gs = figure.add_gridspec(2, 3)

    figure, gs = plt.subplots(2, 3, figsize=[8, 6])

    gs[0,0].plot(image_x, image_y, 'c+')
    gs[0,0].plot(source_x, source_y, 'co')

    for index in range(len(xi1)):
        gs[0,0].plot([xi1[index], xi2[index]], [yi1[index], yi2[index]],\
                      'r', lw=0.5)
        gs[0,0].plot([xs1[index], xs2[index]], [ys1[index], ys2[index]],\
                      'b', lw=0.5)

    gs[0,0].set_xlabel('x (arcsec)')
    gs[0,0].set_xlim([-3, 3])
    gs[0,0].set_ylabel('y (arcsec)')
    gs[0,0].set_ylim([-3, 3])
    gs[0,0].set_aspect('equal')

    index_to_plot = range(1, len(mockimages)+1, 1)
    print(index_to_plot)

    filternames = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']

    for pindex in index_to_plot:
        print(pindex)
        xindex = int(pindex / 3)
        yindex = pindex % 3
        print(xindex, yindex)
        gs[xindex, yindex].imshow(mockimages[pindex-1], cmap='gray',\
                        interpolation='none', origin='lower')
        gs[xindex, yindex].set_axis_off()
        gs[xindex, yindex].set_title(filternames[pindex-1])

    plt.tight_layout()
#    plt.show()
    plt.savefig(filename)

def generate_test_images():
#    save_galaxy_params('galaxy.fits')
#    save_quasar_params('quasar.fits')

#    save_all_images()

    # define survey and filters #
    PSg = spec.read_filter('Pan-Starrs', 'PSg')
    PSr = spec.read_filter('Pan-Starrs', 'PSr')
    PSi = spec.read_filter('Pan-Starrs', 'PSi')
    PSz = spec.read_filter('Pan-Starrs', 'PSz')
    PSy = spec.read_filter('Pan-Starrs', 'PSy')


    filters = [PSg, PSr, PSi, PSz, PSy]
    filtnamelist = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']
    PSsurvey = PanStarrsSurvey()
    PSF_dir = {'PSg': dir_data + '/psf/PanStarrs/PSg_psf.fits',\
               'PSr': dir_data + '/psf/PanStarrs/PSr_psf.fits',\
               'PSi': dir_data + '/psf/PanStarrs/PSi_psf.fits',\
               'PSz': dir_data + '/psf/PanStarrs/PSz_psf.fits',\
               'PSy': dir_data + '/psf/PanStarrs/PSy_psf.fits'}

    beta = 2.5

   # define a galaxy #
    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'

    jaguar_data_hdulist = fits.open(jaguar_file)
    jaguar_flux = jaguar_data_hdulist[1].data
    jaguar_wave = jaguar_data_hdulist[2].data

    jaguar_info = Table.read(jaguar_info_masterfile)

    mask_gal = (jaguar_info['redshift']<1)\
            &(jaguar_info['redshift']>0.2)\
            &(jaguar_info['mStar']>10)\
            &(jaguar_info['mStar']>9)

    print(jaguar_info[mask_gal])


    # output lens galaxy info #
    IDlist = []
    zlist = []
    mslist = []
    sigmalist = []
    elliplist = []
    maglist = []

    f = open('./galmags.txt', 'w')
    for index in range(2, 102, 2):

        flux = jaguar_flux[mask_gal][index]
        wave = jaguar_wave
        z_lens = jaguar_info[mask_gal]['redshift'][index]
        galobj = ObjectSED(wave, flux, z_lens, 'OBS', 'lens')

        ID = float(jaguar_info['ID'][mask_gal][index])
        re = float(jaguar_info['Re_circ'][mask_gal][index])
        n = float(jaguar_info['sersic_n'][mask_gal][index])
        pa = float(jaguar_info['position_angle'][mask_gal][index])
        ellip = float(1 - jaguar_info['axis_ratio'][mask_gal][index])
        mstar = float(jaguar_info['mStar'][mask_gal][index])
        sigma = MStar_sigma(mstar, re, n)

        print(index, z_lens, mstar, ellip)
        PSmag_lens = np.array([galobj.magnitude(f) for f in filters])
        print(PSmag_lens)
        f.write('%d '%index)
        f.write('%.2f %.2f %.2f %.2f %.2f\n'\
                %(PSmag_lens[0], PSmag_lens[1], PSmag_lens[2], PSmag_lens[3], PSmag_lens[4]))
        continue

        IDlist.append(index)
        zlist.append(z_lens)
        mslist.append(mstar)
        sigmalist.append(sigma)
        elliplist.append(ellip)
        maglist.append(PSmag_lens)

        if 1:

            lens_massinfo = [{'Type':'sie', 'sigma':sigma, 'r_core':0.01,\
                    'ellipticity':ellip, 'pa':pa, 'x_center':0, 'y_center':0}]
            lens_lightinfo = [{'Type':'sersic', 're':re, 'n':n,\
                    'ellipticity':ellip, 'pa':pa, 'x_center':0, 'y_center':0}]
            lens = OneObject(z_lens, mass_component=lens_massinfo,\
                             light_component=lens_lightinfo)

            # define a quasar #

            Mabs = -25
            z_source = 5

            flux, wave, redshift = quasarsample_ian(z_source, Mabs)
            qsoobj = ObjectSED(wave, flux, z_source, 'OBS', 'source')
            PSmag_source = [qsoobj.magnitude(f) for f in filters]

            # sim a lens #

            for xindex in range(6):
                qso_x = xindex * 0.2
                qso_y = 0

                source_lightinfo = [{'Type':'point', 'x_center':qso_x,\
                                     'y_center':qso_y}]
                source = OneObject(z_source,\
                                   light_component=source_lightinfo)

                lenssys = LensSystem(lens, source)

                # save images #
                outdir = './M10/%d'%(index)
                outdir_thisq = outdir + '/' + 'x%.1f'%(qso_x)
                if not os.path.exists(outdir_thisq):
                    os.system('mkdir -p ' + outdir_thisq)

#                lenssys.plot_lensconfig(outdir_thisq+'/lensconfig.pdf',\
#                                        [-3, 3], [-3, 3])

                mockimages = []
                for k_filt in range(len(filtnamelist)):
                    filtname = filtnamelist[k_filt]
                    psf_file = PSF_dir[filtname]
                    mag_lens = PSmag_lens[k_filt]
                    mag_source = PSmag_source[k_filt]

                    lenssys.update_lensmag(mag_lens)
                    lenssys.update_sourcemag(mag_source)

#                    sim_a_lens(lenssys, PSsurvey, filtname,\
#                        psf_file, noisy_output=outdir_thisq+\
#                        '/noisy_%s.fits'%(filtname))

                    mockimages.append(fits.open(outdir_thisq+\
                        '/noisy_%s.fits'%filtname)[1].data[8:32, 8:32])

                filename = outdir + '/../plots/%d_%.2f.pdf'%(index, qso_x)
                plot_lens_system(lenssys, mockimages, filename)
    f.close()
#    tbl = Table({'ID':IDlist, 'redshift':zlist, 'mstar':mslist,\
#                 'sigma': sigmalist, 'ellip':elliplist})
#    tbl.write('./M9_lenses.csv')

def save_all_images():
    filtnamelist = ['PSg', 'PSr', 'PSi', 'PSz', 'PSy']
    PSsurvey = PanStarrsSurvey()
    PSF_dir = {'PSg': dir_data + '/psf/PanStarrs/PSg_psf.fits',\
               'PSr': dir_data + '/psf/PanStarrs/PSr_psf.fits',\
               'PSi': dir_data + '/psf/PanStarrs/PSi_psf.fits',\
               'PSz': dir_data + '/psf/PanStarrs/PSz_psf.fits',\
               'PSy': dir_data + '/psf/PanStarrs/PSy_psf.fits'}

    beta = 2.5
    '''
    for filt in filtnamelist:
        paras = [PSsurvey.FWHM[filt], beta]
        PSF = generate_psf(moffat, paras, 4, PSsurvey.pixscale)
        hdu = fits.PrimaryHDU(data=PSF)
        hdu.writeto(PSF_dir[filt], overwrite=True)
    '''
    gal_data = fits.open('galaxy.fits')[1].data
    qso_data = fits.open('quasar.fits')[1].data

#    outdir = dir_data + '/simimg/sim/PanStarrs'
    outdir = '/media/minghao/73882b03-7347-4b5e-9da5-30f0bfa9ee91/lensqso/sim'
    if not os.path.exists(outdir):
        os.system('mkdir -p %s'%(outdir))

    for i_gal in range(len(gal_data)):
        for j_qso in range(len(qso_data)):

            sigma = gal_data['sigma'][i_gal]
            ellip = gal_data['ellipticity'][i_gal]
            pa = gal_data['position_angle'][i_gal]
            re = gal_data['sersic_re'][i_gal]
            n = gal_data['sersic_index'][i_gal]
            z_lens = gal_data['redshift'][i_gal]
            gal_id = gal_data['ID'][i_gal]

            z_source = qso_data['redshift'][j_qso]
            qso_id = qso_data['ID'][j_qso]

            lens_massinfo = [{'Type':'sie', 'sigma':sigma, 'r_core':0.01,\
                    'ellipticity':ellip, 'pa':pa, 'x_center':0, 'y_center':0}]
            lens_lightinfo = [{'Type':'sersic', 're':re, 'n':n,\
                    'ellipticity':ellip, 'pa':pa, 'x_center':0, 'y_center':0}]
#            lens = OneObject(z_lens, mass_component=lens_massinfo,\
#                             light_component=lens_lightinfo)

            mag_lens = gal_data['mag'][i_gal][k_filt]
            mag_source = qso_data['mag'][j_qso][k_filt]

            print(i_gal, j_qso)
            print(mag_lens)
            print(mag_source)

            continue

            for qso_x in np.arange(-0.4, 0.4, 0.2):
                for qso_y in np.arange(-0.4, 0.4, 0.2):

                    source_lightinfo = [{'Type':'point', 'x_center':qso_x,\
                                         'y_center':qso_y}]
                    source = OneObject(z_source,\
                                       light_component=source_lightinfo)

                    lenssys = LensSystem(lens, source)

                    random_flag = np.random.rand(1)
                    if random_flag<0.01:

                        for k_filt in range(len(filtnamelist)):
                            filtname = filtnamelist[k_filt]
                            psf_file = PSF_dir[filtname]
                            lenssys.update_lensmag(mag_lens)
                            lenssys.update_sourcemag(mag_source)

                            outdir_thisq = outdir + '/' + filtname
                            if not os.path.exists(outdir_thisq):
                                os.system('mkdir -p ' + outdir_thisq)
#                            print(outdir_thisq, qso_id, gal_id)

#                            sim_a_lens(lenssys, PSsurvey, filtname,\
#                                psf_file, noisy_output=outdir_thisq+\
#                                '/noisy_%d_%d_%.1f_%.1f.fits'\


def main():
    M10_direct = './M10/'
    generate_test_images()

if __name__=='__main__':
    main()
