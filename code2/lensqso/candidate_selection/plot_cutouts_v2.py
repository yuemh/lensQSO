import os, sys
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import mylib.zscale as zs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy.optimize import minimize

from matplotlib import font_manager, rcParams
#import mylib.spec_measurement as spec
from astropy.nddata.utils import NoOverlapError

fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

#prop = font_manager.FontProperties(fname=fontpath)
#rcParams['font.family'] = prop.get_name()

dir_root = os.path.abspath(os.getcwd()+'/../../')
dir_data = dir_root + '/data'

def richards_sed(z):
    datFile = os.path.join('./Richards2006_t3_sed.dat')

    # Read Richards SED
    cat = np.genfromtxt(datFile, delimiter="")
    fre, lum = 10 ** cat[:, 0], 10 ** cat[:, 1]
    wave = 3e14/fre
    flux = lum / 1e55

    return (np.flip(wave, axis=0)*(1+z), np.flip(flux, axis=0))

def ian_qso_sed(z):
    alldat = np.loadtxt('./ianqso.txt')
    wave = alldat[:, 0]
    flux = alldat[:, 1]
    return (wave*(1+z)/1e4, flux*1e-19)

def jaguar_gal_sed(z):
    alldat = np.loadtxt('./jaguargal.txt')
    wave = alldat[:, 0]
    flux = alldat[:, 1]

    return (wave*(1+z)/1e4, flux*1e-19)

def treat_oneobj():
    ### plot ###
    plot_oneobj()

    ### match information ###

    SDSS = match_sdss()
    PanStarrs = mags[:10]
    WISE = mags[13:15]
    UKIDSS = mags[10:13]
    FIRST = match_first()
    ROSAT = match_rosat()

    # Output info file

    info_keys = ['RA', 'DEC', 'SDSSu', 'PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                'UKIDSS_J', 'UKIDSS_H', 'UKIDSS_Ks','W1', 'W2', 'FIRST',\
                'XRAY']

def trim_image_size(filename, size):
    hdulist = fits.open(filename)

    header = hdulist[0].header
    data = hdulist[0].data

    if type(data)==type(None):
        return np.array([])
        hdulist.close()

    else:
        pos = np.array(data.shape)/2
        size = u.Quantity((size, size), u.arcsec)
        wcs = WCS(header)

        cutouts = Cutout2D(data, pos, size, wcs=wcs)
        hdulist.close()
        return cutouts.data

def residual_sed(x, paras):
    wavelength, flux, wavelength_sed, flux_sed = paras
    redshift, scale = x

    newflux = np.interp(wavelength, wavelength_sed * (1+redshift), flux_sed)\
            *scale

    return np.sum((flux-newflux)**2)

def fit_sed(wavelength, flux, wavelength_sed, flux_sed):
    bounds = ((0, 6.), (0, None))

    num_scale = 10 / np.sum(flux)

    flux = flux * num_scale
    flux_sed = flux_sed * num_scale
    scale_init = np.median(flux) / np.median(flux_sed)

    result = minimize(residual_sed, x0=(5, scale_init),\
                      args=[wavelength, flux, wavelength_sed, flux_sed],\
                      bounds=bounds,\
                      method='SLSQP')
    result = minimize(residual_sed, x0=result.x,\
                      args=[wavelength, flux, wavelength_sed, flux_sed],\
                      bounds=bounds,\
                      method='TNC')

    return result.x

def residual_sed_2(x, paras):
    wavelength, flux, wavelength_sed_q, flux_sed_q,\
            wavelength_sed_g, flux_sed_g = paras
    redshift_q, scale_q, redshift_g, scale_g = x

    newflux = np.interp(wavelength, wavelength_sed_q * (1+redshift_q),\
                        flux_sed_q) * scale_q \
            + np.interp(wavelength, wavelength_sed_g * (1+redshift_g),\
                        flux_sed_g) * scale_g \

    return np.sum((flux-newflux)**2)

def fit_sed_2(wavelength, flux, wavelength_sed_q, flux_sed_q,\
              wavelength_sed_g, flux_sed_g):
    bounds = ((0, 6.), (0, None), (0, 2), (0, None))

    num_scale = 10 / np.sum(flux)

    flux = flux * num_scale
    flux_sed_q = flux_sed_q * num_scale
    flux_sed_g = flux_sed_g * num_scale

    scale_init_q = np.median(flux) / np.median(flux_sed_q) / 2
    scale_init_g = np.median(flux) / np.median(flux_sed_g) / 2

    result = minimize(residual_sed_2, x0=(5, scale_init_q, 0.1, scale_init_g),\
                args=[wavelength, flux, wavelength_sed_q, flux_sed_q,\
                      wavelength_sed_g, flux_sed_g],\
                bounds=bounds, method='SLSQP')

    for i in range(10):
        result = minimize(residual_sed_2, x0=result.x,\
                    args=[wavelength, flux, wavelength_sed_q, flux_sed_q,\
                          wavelength_sed_g, flux_sed_g],\
                    bounds=bounds, method='TNC',\
                    options={'stepmx': 0.1})
#        print(result)

    return result.x

def read_images(filenames, size):

    Nimg = len(filenames)

    data_list = []
    flag_list = []

    for index in range(Nimg):
        filename = filenames[index]

        if filename[-5:]=='.fits':

            try:
                data = trim_image_size(filename, size)
                data_list.append(data)

                if len(data) == 0:
                    flag_list.append(False)
                else:
                    flag_list.append(True)
            except NoOverlapError:
                print('Wrong image shape: %s'%(filename))

                data_list.append(np.array([]))
                flag_list.append(0)

            except FileNotFoundError:
                try:
                    new_filename = 'J' + filename[1:]

                    data = trim_image_size(new_filename, 5)
                    data_list.append(data)

                    if len(data) == 0:
                        flag_list.append(0)
                    else:
                        flag_list.append(1)

                except FileNotFoundError:
                    print('No file named %s'%(filename))
                    data_list.append(np.array([]))
                    flag_list.append(0)


        elif filename[-4:]=='.jpg' or filename[-5:]=='.jpeg':
            data = plt.imread(filename)
            data_list.append(data)
            flag_list.append(2)

    return [data_list, flag_list]

def plot_oneobj(filenames, labels, size, wavelength, flux, fluxerr, \
                mag_to_plot = [], mag_to_ccplot1=[], mag_to_ccplot2=[],\
                title='', output='', redshift=-1, infolist=[],\
                style='SED', specfile=''):

    data_list, flag_list = read_images(filenames, size)

    ### Fitting SED ###

#    wavelength_sed, flux_sed = richards_sed(0)
#    redshift, scale = fit_sed(wavelength, flux, wavelength_sed, flux_sed)
#    wavelength_sed_plot = wavelength_sed * (1 + redshift)
#    flux_sed_plot = flux_sed * scale

    wavelength_sed_q, flux_sed_q = ian_qso_sed(0)
    wavelength_sed_g, flux_sed_g = jaguar_gal_sed(0)

    redshift_q, scale_q, redshift_g, scale_g = \
            fit_sed_2(wavelength, flux, wavelength_sed_q, flux_sed_q,\
                    wavelength_sed_g, flux_sed_g)

    wavelength_sed_q = wavelength_sed_q * (1 + redshift_q)
    wavelength_sed_g = wavelength_sed_g * (1 + redshift_g)
    flux_sed_q = flux_sed_q * scale_q
    flux_sed_g = flux_sed_g * scale_g

    wavelength_sed_plot = np.arange(1000, 100000, 10) / 1e4
    flux_sed_plot =\
        np.interp(wavelength_sed_plot, wavelength_sed_q, flux_sed_q)\
        + np.interp(wavelength_sed_plot, wavelength_sed_g, flux_sed_g)

    ### Making Plot ###

    figure = plt.figure(figsize=[8, 8])
    ax_grri = figure.add_subplot(111, position=[0.10, 0.7, 0.25, 0.25])
    ax_riiz = figure.add_subplot(111, position=[0.45, 0.7, 0.25, 0.25])

    ax_spec = figure.add_subplot(111, position=[0.1, 0.44, 0.6, 0.16])
    ax_text = figure.add_subplot(111, position=[0.715, 0.05, 0.25, 0.9])

    gs = gridspec.GridSpec(2, 5, figure=figure, height_ratios=[3, 3])
    gs.update(left=0.05, top=0.35, right=0.7, bottom=0.03)

    ### Plot color ###
    if not len(mag_to_ccplot1)==4:
        mag_to_ccplot1 = ['g', 'r', 'r', 'i']

    ax_grri.set_xlabel('%s - %s'%(mag_to_ccplot1[0], mag_to_ccplot1[1]), fontsize=12)
    ax_grri.set_ylabel('%s - %s'%(mag_to_ccplot1[2], mag_to_ccplot1[3]), fontsize=12)
    ax_grri.plot(np.full(20, 1.8), np.linspace(1, 3, 20), 'k--', alpha=0.5)
    ax_grri.plot(np.linspace(1.8, 3, 20), np.full(20, 1), 'k--', alpha=0.5)
    ax_grri.plot(np.linspace(0, 3, 20), np.full(20, 0.5), 'k--', alpha=1)

    ax_grri.set_xlim([0, 3])
    ax_grri.set_ylim([0, 3])

    ax_grri.plot([mag_to_plot[0]-mag_to_plot[1]],\
                 [mag_to_plot[2]-mag_to_plot[3]], 'r*', ms=10)

    if not len(mag_to_ccplot2)==4:
        mag_to_ccplot2 = ['r', 'i', 'i', 'z']

    ax_riiz.set_xlabel('%s - %s'%(mag_to_ccplot2[0], mag_to_ccplot2[1]), fontsize=12)
    ax_riiz.set_ylabel('%s - %s'%(mag_to_ccplot2[2], mag_to_ccplot2[3]), fontsize=12)

    ax_riiz.plot(np.full(20, 1), np.linspace(-1, 0.325, 20), 'k--', alpha=0.5)
    ax_riiz.plot(np.linspace(1.632, 3, 20), np.full(20, 0.72), 'k--', alpha=0.5)
    ax_riiz.plot(np.full(20, 0.5), np.linspace(-1, 0.025, 20), 'k--', alpha=1)
#    ax_riiz.plot(np.linspace(0.7, 2, 20), np.full(20, 0.4), 'k--', alpha=1)
    ax_riiz.plot(np.linspace(0.5, 3, 20),\
                 np.linspace(0.5, 3, 20)*0.625-0.3, 'k--', alpha=1)

    ax_riiz.set_xlim([0.5, 3])
    ax_riiz.set_ylim([-0.5, 1.5])

    ricolor = np.linspace(0, 3, 10)
    izcolor = ricolor * 0.5

    ax_riiz.plot(ricolor, izcolor, 'b-')
    ax_riiz.plot([mag_to_plot[4]-mag_to_plot[5]],\
                 [mag_to_plot[6]-mag_to_plot[7]], 'r*', ms=10)

    ### Plot SED ###
    if style == 'SED':
        ax_spec.errorbar(wavelength, flux, yerr=fluxerr, fmt='bo')
        ax_spec.plot(wavelength_sed_plot, flux_sed_plot, label='all')
        ax_spec.plot(wavelength_sed_q, flux_sed_q,\
                     label='$z_q=%.2f$'%(redshift_q), alpha=0.3)
        ax_spec.plot(wavelength_sed_g, flux_sed_g,\
                     label='$z_g=%.2f$'%(redshift_g), alpha=0.3)

        ax_spec.legend()
        ax_spec.set_xlim([0.1, 15])
        ax_spec.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])

        ax_spec.set_xscale('log')
    #    ax_spec.set_yscale('log')

        ax_spec.set_xlabel(r'Wavelength ($\mu m$)')
        ax_spec.set_ylabel(r'$\lambda F_\lambda (erg s^{-1} cm^{-2})$')

    elif style=='SPEC':
        spec = read_sdss_spec(specfile)
        ax_spec.plot(spec.wavelength, spec.value, lw=1)
        ax_spec.ticklabel_format(axis='y', style='sci', scilimit=[-2, 2])

        ax_spec.set_xlabel(r'Wavelength ($\mu m$)')
        ax_spec.set_ylabel(r'$F_\lambda (erg s^{-1} cm^{-2} {\AA}^{-1})$')

    ### Add info ###
    ax_text.set_axis_off()

    for index_text in range(len(infolist)):
        x_pos = 0.05
        y_pos = 0.99 - 0.03 * index_text
        ax_text.text(x_pos, y_pos, infolist[index_text], fontsize=12)

    ### Plot cutouts ###

    index_to_plot = range(len(data_list))

    for ax_idx in index_to_plot:
        ax_idx_1 = int(ax_idx / 5)
        ax_idx_2 = ax_idx % 5

        ax_img = plt.subplot(gs[ax_idx_1, ax_idx_2])
        ax_img.set_axis_off()

        data = data_list[ax_idx]
        flag = flag_list[ax_idx]

        ax_img.set_title(labels[ax_idx], fontsize=14)

        if flag==1:
            ax_img.imshow(data_list[ax_idx], cmap='gray',\
                          interpolation='none', origin='lower')
        elif flag==2:
            ax_img.imshow(data_list[ax_idx], interpolation='none')
        else:
            ax_img.imshow(np.full((2, 2), np.NaN), cmap='gray',\
                          interpolation='none', origin='lower')

    plt.suptitle(title, fontsize=16)
    plt.savefig(output)
    plt.close('all')
#    plt.show()

def read_mag(tbl, index, mag_keys, magerr_keys):

    mag_list = []
    magerr_list = []

    for nkey in range(len(mag_keys)):
        mag = tbl[mag_keys[nkey]][index]
        magerr = tbl[magerr_keys[nkey]][index]

#        print(mag_keys[nkey], mag)

        try:
            mag = float(mag)
            if mag>0:
                mag_list.append(mag)
            else:
                mag_list.append(np.NaN)
        except ValueError:
            mag_list.append(np.NaN)

        try:
            magerr = float(magerr)
            if magerr>0:
                magerr_list.append(magerr)
            else:
                magerr_list.append(np.NaN)
        except ValueError:
            magerr_list.append(np.NaN)

    return (np.array(mag_list, dtype=float),\
            np.array(magerr_list, dtype=float))

def get_mag_keys(surveyname):

    if surveyname=='DES':
        magkeys = ['mag_aper_8_g_dered', 'mag_aper_8_r_dered', \
                   'mag_aper_8_i_dered', 'mag_aper_8_z_dered', \
                   'mag_aper_8_y_dered']
        magerrkeys = ['magerr_aper_8_g', 'magerr_aper_8_r',\
                      'magerr_aper_8_i', 'magerr_aper_8_z',\
                      'magerr_aper_8_y']

    elif surveyname=='VHS':
        magkeys = ['ZAPERMAG3', 'YAPERMAG3', 'JAPERMAG3',\
                   'HAPERMAG3', 'KSAPERMAG3']
        magerrkeys = ['ZAPERMAG3ERR', 'YAPERMAG3ERR', 'JAPERMAG3ERR',\
                   'HAPERMAG3ERR', 'KSAPERMAG3ERR']

    elif surveyname=='WISE':
        magkeys = ['w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']
        magerrkeys = ['w1sigmpro', 'w2sigmpro', 'w3sigmpro', 'w4sigmpro']

    return [magkeys, magerrkeys]

def get_vegazp(surveyname):
    if surveyname=='VHS':
        return [0.502, 0.600, 0.916, 1.366, 1.827]
    elif surveyname=='WISE':
        return [2.699, 3.339, 5.174, 6.620]
    else:
        return [0, 0, 0, 0, 0]

def get_centralwave(surveyname):
    if surveyname=='DES':
        return [5270, 6590, 7890, 9760, 10030]
    elif surveyname=='VHS':
        return [8770, 10200, 12520, 16450, 21470]
    elif surveyname=='WISE':
        return [34000, 46000, 120000, 220000]

def find_des_name(ra, dec, des_dir):
    for filename in os.listdir(des_dir):
        if filename.startswith('matched'):
            infofile = des_dir + '/' + filename
            break

    info = Table.read(infofile)

    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    infocoord = SkyCoord(ra=info['RA'], dec=info['DEC'], unit='deg')

    idx, d2d, d3d = coord.match_to_catalog_sky(infocoord)

    return info['THUMBNAME'][idx]

def plot_catalog(catalog_file, cutout_dir, ra_col='ra_1', dec_col='dec_1',\
                 optical_survey='DES', nearir_survey='VHS', midir_survey='WISE',\
                 spec_dir='', specinfo_file=''):

    # Define constants #
    catalog_data = Table.read(catalog_file)

    # get a better sample
#    mask = (catalog_data['Prob1'][:, 0]>0.95)\
#            &(catalog_data['Prob2'][:, 0]>0.9)

#    catalog_data = catalog_data[mask]
#    print(len(catalog_data))
#    return 0
#    print(catalog_data)
#    ra_all_candidate = catalog_data['ra']
#    dec_all_candidate = catalog_data['dec']

#    spectrum_list = find_sdss_spec(\
#                ra_all_candidate, dec_all_candidate, spec_info, radius=2)

#    print(len(catalog_data))

    mag_keys = get_mag_keys(optical_survey)[0][:] \
            + get_mag_keys(midir_survey)[0][:2]
#            get_mag_keys(nearir_survey)[0][2:3] + \

    magerr_keys = get_mag_keys(optical_survey)[1][:] \
            + get_mag_keys(midir_survey)[1][:2]
#           get_mag_keys(nearir_survey)[1][2:3] + \

    llist = get_centralwave(optical_survey)[:] \
            + get_centralwave(midir_survey)[:2]
#           get_centralwave(nearir_survey)[2:3] + \

    llist = np.array(llist)/1e4
    nulist = 3e14/llist

    zplist = get_vegazp(optical_survey)[:] \
            + get_vegazp(midir_survey)[:2]
#           get_vegazp(nearir_survey)[2:3] + \

    zplist = np.array(zplist)

    cutout_bands = ['DESg', 'DESr', 'DESi', 'DESz', 'DESy',\
                    'LSg', 'LSr', 'LSz', 'LSgrz', 'LSres']
    mag_to_plot_keys = ['mag_aper_8_g_dered', 'mag_aper_8_r_dered',\
                        'mag_aper_8_r_dered', 'mag_aper_8_i_dered',\
                        'mag_aper_8_r_dered', 'mag_aper_8_i_dered',\
                        'mag_aper_8_i_dered', 'mag_aper_8_z_dered']

    ps_dir = '../cutouts/gaia/PanStarrs'
    ls_dir = '/home/minghao/Desktop/research/2018/LensQSOSurvey/data/catalog/2019/LDSS3/20190920/cutout/faint/highz/chisq/legacy'
    des_dir = '/home/minghao/Desktop/research/2018/LensQSOSurvey/data/catalog/2019/LDSS3/20190920/cutout/faint/highz/chisq/des'

    done_id_list = []

    for index in range(len(catalog_data)):

        # get jname #

        ra = catalog_data[ra_col][index]
        dec = catalog_data[dec_col][index]

        coadd_id = catalog_data['coadd_object_id'][index]

        if coadd_id in done_id_list:
            continue

        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        jname = 'J' + str(name)
        pname = 'P' + str(name)

        desname = find_des_name(ra, dec, des_dir)

        # get magnitudes #

        maglist, magerrlist =\
                read_mag(catalog_data, index, mag_keys, magerr_keys)

        maglist_AB = maglist + zplist

        flux = 3631e-23*10**(-0.4*(maglist_AB))
        fluxerr = 3631e-23*10**(-0.4*(maglist_AB))*np.log(10)*0.4*magerrlist

        lfl = flux * nulist
        lflerr = fluxerr * nulist

        # mask magnitudes #

        mask1 = (~np.isnan(maglist))|(~np.isinf(maglist))|(np.abs(maglist)<100)
        mask2 = (~np.isnan(magerrlist))|(~np.isinf(magerrlist))|(np.abs(magerrlist)<100)

        mask = (mask1 & mask2)

        lfl_input = lfl[mask]
        lflerr_input = lflerr[mask]
        llist_input = llist[mask]

        # get image names #

#        sdss_files = [sdss_dir + '/%s/%s.sdssu.fits'%(jname, jname)]
        ps_files = [ps_dir + '/%s/%s.%s.fits'%(jname, jname, band)\
                      for band in 'grizy']
        des_files = [des_dir + '/%s_%s.fits'%(desname, band)\
                      for band in 'grizY']
        ls_files = [ls_dir + '/%s/%s.%s.fits'%(jname, jname, band)\
                      for band in 'grz']
        lscolor_files = [ls_dir + '/%s/%s%s.grz.jpeg'%(jname, layer, jname)\
                         for layer in ['', 'res-']]

        if optical_survey=='DES':
            filenames = des_files + ls_files + lscolor_files
        else:
            filenames = ps_files + ls_files + lscolor_files

        jpg_filenames = lscolor_files

        # get mags to plot #

        mag_to_plot = [catalog_data[key][index] for key in mag_to_plot_keys]

        # get info #

        # coordinates
        infolist = []

        infolist.append(r'RA = %.5f'%(ra))
        infolist.append(r'DEC = %.5f'%(dec))
        infolist.append(r'l = %.5f'%(catalog_data['galactic_l'][index]))
        infolist.append(r'b = %.5f'%(catalog_data['galactic_b'][index]))

        # magnitudes

        mag_keys_label = ['g', 'r', 'i', 'z', 'y', 'W1', 'W2']
        for mag_idx in range(len(mag_keys)):

            infolist.append(\
                            r'%s = %.2f $\pm$ %.2f'%(mag_keys_label[mag_idx],\
                            maglist[mag_idx], magerrlist[mag_idx]))
        '''
        SDSSu_str = r'SDSSu = %.2f $\pm$ %.2f'%(maglist[0], magerrlist[0])
        SDSSg_str = r'SDSSg = %.2f $\pm$ %.2f'%(maglist[1], magerrlist[1])
        SDSSr_str = r'SDSSr = %.2f $\pm$ %.2f'%(maglist[2], magerrlist[2])
        SDSSi_str = r'SDSSi = %.2f $\pm$ %.2f'%(maglist[3], magerrlist[3])
        SDSSz_str = r'SDSSz = %.2f $\pm$ %.2f'%(maglist[4], magerrlist[4])
        w1_str = r'W1mag = %.2f $\pm$ %.2f'%(maglist[5], magerrlist[5])
        w2_str = r'W2mag = %.2f $\pm$ %.2f'%(maglist[6], magerrlist[6])
        w3_str = r'W3mag = %.2f $\pm$ %.2f'%(maglist[7], magerrlist[7])

        infolist += \
                [SDSSu_str, SDSSg_str, SDSSr_str, SDSSi_str, SDSSz_str,\
                 w1_str, w2_str, w3_str]
        '''
        # additional magnitudes

        # gaia
#        parallax_str = 'parallax = %.2f $\pm$ %.2f'%(catalog_data['parallax_g'][index],\
#                                        catalog_data['parallax_error_g'][index])
#        pmra_str = 'pmra = %.2f $\pm$ %.2f'%(catalog_data['pmra_g'][index],\
#                                          catalog_data['pmra_error_g'][index])
#        pmdec_str = 'pmdec = %.2f $\pm$ %.2f'%(catalog_data['pmdec_g'][index],\
#                                          catalog_data['pmdec_error_g'][index])

#        infolist += [parallax_str, pmra_str, pmdec_str]

        # other surveys
#        FIRST_str = 'FIRST = %.2f'%(catalog_data['FINT'][index])
#        XMM_str = 'XMM = %.2f'%(catalog_data['SC_DET_ML'][index])
#        ROSAT_str = 'ROSAT = %.2f'%(catalog_data['CTS'][index])

#        infolist += [XMM_str]

#        # simbad
#        simbad_main_type = '%s'%(catalog_data['main_type'][index])
#        simbad_other_type = '%s'%(catalog_data['main_type'][index])
#        simbad_z = '%s'%(catalog_data['redshift'][index])

#        infolist += [simbad_main_type, simbad_other_type, simbad_z]

        # extended
#        magdiff = catalog_data['iPSFMag'][index]-catalog_data['iApMag'][index]\
#                + catalog_data['zPSFMag'][index]-catalog_data['zApMag'][index]\
#                + catalog_data['yPSFMag'][index]-catalog_data['yApMag'][index]

#        extended_string = '%.2f'%(magdiff)

#        infolist += ['SDSS Type = %s'%(catalog_data['type'][index])]

        # residuals
        LSg_rchisq = 'LSg chi2 = %.2f'%(catalog_data['rchisq_g'][index])
        LSr_rchisq = 'LSr chi2 = %.2f'%(catalog_data['rchisq_r'][index])
        LSz_rchisq = 'LSz chi2 = %.2f'%(catalog_data['rchisq_z'][index])

        infolist += [LSg_rchisq, LSr_rchisq, LSz_rchisq]

        # distance to the locus
#        p3 = np.array([catalog_data['rFiberMag'][index] - catalog_data['iFiberMag'][index],\
#                       catalog_data['iFiberMag'][index] - catalog_data['zFiberMag'][index]])
#        p1 = np.array([0, 0])
#        p2 = np.array([2, 1.1])

#        dist = np.abs(np.cross(p2-p1, p3-p1) / np.linalg.norm(p2-p1))
#        dist_string = 'Dist = %.2f'%(dist)

        # random forest scores
        score1 = catalog_data['Prob1'][:, 0][index]
        score2 = catalog_data['Prob2'][:, 0][index]

        string_score1 = 'Prob1 = %.2f'%(score1)
        string_score2 = 'Prob2 = %.2f'%(score2)

        infolist += [string_score1, string_score2]
        # PosDiff
#        posdiff_string = 'PosDiff = %.2f'%(catalog_data['POSDIFF'][index])


        # info_othersample
#        z45_qso_info = 'z_jinyi = %.2f'%(catalog_data['redshift'][index])
#        milliquas_info = 'z_milliquas = %.2f %s'%(catalog_data['Z'][index],\
#                                                 catalog_data['Descrip'][index])

#        infolist += [milliquas_info, z45_qso_info]

        # pair size
#        minsize = catalog_data['minsize'][index]
#        maxsize = catalog_data['maxsize'][index]

#        string_minsize = 'minsize = %.2f'%(minsize)
#        string_maxsize = 'maxsize = %.2f'%(maxsize)

#        infolist += [string_minsize, string_maxsize]

        # morphology
        subtable = catalog_data[catalog_data['coadd_object_id']==coadd_id]
        for subidx in range(len(subtable)):
            infolist.append('Type%d = %s'%(subidx, subtable['type'][subidx]))

        # read spectrum #

        if 0:
            mask_thisq = (index==spectrum_list[0])
            specfile = spec_dir + '/' + str(spectrum_list[1][mask_thisq][0])
            plot_style = 'SPEC'
        else:
            specfile=''
            plot_style = 'SED'

        # save plots #

        n_output = 0
        output = cutout_dir+'/%s.pdf'%(jname)

        if os.path.exists(output):
            continue
        print(jname)

        try:
            plot_oneobj(filenames, cutout_bands, 6,\
                        llist_input, lfl_input, lflerr_input, \
                        title=jname, mag_to_plot=mag_to_plot,\
                        output=output, infolist=infolist,\
                        style=plot_style, specfile=specfile)
        except FileNotFoundError:
            continue

        done_id_list.append(coadd_id)

def read_sdss_spec(specfile):
    hdulist = fits.open(specfile)
    data = hdulist[1].data

    flux = np.array([array[0] for array in data])
    wave = 10**np.array([array[1] for array in data])

    qsospec = spec.Spectrum(wavelength=wave, value=flux*1e-17,\
                            units=[u.Angstrom, u.erg/u.s/u.cm/u.cm/u.Angstrom])

    hdulist.close()
    return qsospec

def find_sdss_spec(ra, dec, infotable, radius=-1):
#    allra = np.char.replace(infotable['ra'], "'", "")
#    alldec = np.char.replace(infotable['dec'], "'", "")

    allra = infotable['ra']
    alldec = infotable['dec']

    allcoord = SkyCoord(ra=allra, dec=alldec, unit=(u.deg, u.deg))
    objcoord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

    idx, d2d, d3d = objcoord.match_to_catalog_sky(allcoord)
    d2d = d2d.to(u.arcsecond)

    orig_index = np.arange(len(idx))
    match_index = np.array(idx)
    dist = np.array(d2d/u.arcsecond, dtype=float)

    if radius>0:
        mask = (dist<radius)
        orig_index = orig_index[mask]
        match_index = match_index[mask]
        dist = dist[mask]

    infotable_new = infotable[match_index]

    plate = np.array(infotable_new['#plate'], dtype=int)
    fiber = np.array(infotable_new['fiberid'], dtype=int)
    mjd = np.array(infotable_new['mjd'], dtype=int)

    filenames = np.array(['spec-%04d-%05d-%04d.fits'%\
                 (plate[index], mjd[index], fiber[index]) \
                for index in range(len(plate))], dtype=str)

    return (orig_index, filenames)

def main():
    catalog_file = '/home/minghao/Desktop/research/2018/LensQSOSurvey/data/catalog/2019/LDSS3/20190920/code/highz/faint/chisq/chisq_selcted.fits'
#    dir_spec = dir_data \
#            + '/catalog/cutouts/PS_WISE_GAIA_stationary/MorphPos/spec'
#    specinfo_file = dir_spec + '/optical_search_155613.csv'
    output_dir = '/home/minghao/Desktop/research/2018/LensQSOSurvey/data/catalog/2019/LDSS3/20190920/plot/highz/faint/chisq'

    if not os.path.exists(output_dir):
        os.system('mkdir -p ' + output_dir)

    plot_catalog(catalog_file, output_dir)

if __name__=='__main__':
    main()
