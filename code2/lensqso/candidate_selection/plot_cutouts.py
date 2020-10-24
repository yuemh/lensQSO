import os, sys
import numpy as np
import matplotlib

#matplotlib.use('pgf')
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mylib.spec_measurement as spec
import mylib.zscale as zs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy.optimize import minimize

from matplotlib import font_manager, rcParams
import mylib.spec_measurement as spec

fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

prop = font_manager.FontProperties(fname=fontpath)
rcParams['font.family'] = prop.get_name()

dir_root = os.path.abspath(os.getcwd()+'/../../')
dir_data = dir_root + '/data'

def richards_sed(z):
    datFile = os.path.join(dir_data, 'Richards2006_t3_sed.dat')

    # Read Richards SED
    cat = np.genfromtxt(datFile, delimiter="")
    fre, lum = 10 ** cat[:, 0], 10 ** cat[:, 1]
    wave = 3e14/fre
    flux = lum / 1e55

    return (np.flip(wave)*(1+z), np.flip(flux))

def qso_template_ian():
    dummy = 1

def gal_template_jaguar():
    dummy = 1

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

    else:
        pos = np.array(data.shape)/2
        size = u.Quantity((size, size), u.arcsec)
        wcs = WCS(header)

        cutouts = Cutout2D(data, pos, size, wcs=wcs)
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

def read_images(filenames, size):

    Nimg = len(filenames)

    data_list = []
    flag_list = []

    for index in range(Nimg):
        filename = filenames[index]
        try:
            data = trim_image_size(filename, 5)
            data_list.append(data)

            if len(data) == 0:
                flag_list.append(False)
            else:
                flag_list.append(True)

        except FileNotFoundError:
            try:
                new_filename = 'J' + filename[1:]

                data = trim_image_size(new_filename, 5)
                data_list.append(data)

                if len(data) == 0:
                    flag_list.append(False)
                else:
                    flag_list.append(True)

            except FileNotFoundError:
                print('No file named %s'%(filename))
                data_list.append(np.array([]))
                flag_list.append(False)

    return [data_list, flag_list]

def plot_oneobj(filenames, labels, size, wavelength, flux, fluxerr, \
                mag_to_plot = [],\
                title='', output='', redshift=-1, infolist=[],\
                style='SED', specfile=''):

    data_list, flag_list = read_images(filenames, size)

    ### Fitting SED ###

    wavelength_sed, flux_sed = richards_sed(0)
    redshift, scale = fit_sed(wavelength, flux, wavelength_sed, flux_sed)
    wavelength_sed_plot = wavelength_sed * (1 + redshift)
    flux_sed_plot = flux_sed * scale

    ### Making Plot ###

    figure = plt.figure(figsize=[8, 8])
    ax_grri = figure.add_subplot(111, position=[0.10, 0.7, 0.25, 0.25])
    ax_riiz = figure.add_subplot(111, position=[0.45, 0.7, 0.25, 0.25])

    ax_spec = figure.add_subplot(111, position=[0.1, 0.44, 0.6, 0.16])
    ax_text = figure.add_subplot(111, position=[0.715, 0.05, 0.25, 0.9])

    gs = gridspec.GridSpec(2, 5, figure=figure, height_ratios=[3, 3])
    gs.update(left=0.05, top=0.35, right=0.7, bottom=0.03)

    ### Plot color ###
    ax_grri.set_xlabel('PSg - PSr', fontsize=12)
    ax_grri.set_ylabel('PSr - PSi', fontsize=12)
    ax_grri.plot(np.full(20, 1.5), np.linspace(0.9, 3, 20), 'k--', alpha=0.5)
    ax_grri.plot(np.linspace(1.5, 3, 20), np.full(20, 0.9), 'k--', alpha=0.5)
    ax_grri.plot(np.linspace(0, 3, 20), np.full(20, 0.7), 'k--', alpha=1)

    ax_grri.set_xlim([0, 3])
    ax_grri.set_ylim([0, 3])

    ax_grri.plot([mag_to_plot[0]-mag_to_plot[1]],\
                 [mag_to_plot[2]-mag_to_plot[3]], 'r*', ms=10)

    ax_riiz.set_xlabel('PSr - PSi', fontsize=12)
    ax_riiz.set_ylabel('PSi - PSz', fontsize=12)

    ax_riiz.plot(np.full(20, 0.9), np.linspace(-1, 0.4, 20), 'k--', alpha=0.5)
    ax_riiz.plot(np.linspace(0.9, 2, 20), np.full(20, 0.4), 'k--', alpha=0.5)
    ax_riiz.plot(np.full(20, 0.7), np.linspace(-1, 0.4, 20), 'k--', alpha=1)
    ax_riiz.plot(np.linspace(0.7, 2, 20), np.full(20, 0.4), 'k--', alpha=1)
    ax_riiz.set_xlim([0, 2])
    ax_riiz.set_ylim([-1, 1])

    ricolor = np.linspace(0, 2, 10)
    izcolor = ricolor * 0.5

    ax_riiz.plot(ricolor, izcolor, 'b-')
    ax_riiz.plot([mag_to_plot[4]-mag_to_plot[5]],\
                 [mag_to_plot[6]-mag_to_plot[7]], 'r*', ms=10)

    ### Plot SED ###
    if style == 'SED':
        ax_spec.errorbar(wavelength, flux, yerr=fluxerr, fmt='bo')
        ax_spec.plot(wavelength_sed_plot, flux_sed_plot,\
                     label='Richard+06, z=%.2f'%(redshift))
        ax_spec.legend()
        ax_spec.set_xlim([0.1, 15])
        ax_spec.ticklabel_format(axis='y', style='sci', scilimit=[-2, 2])

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
    if 1:
        ax_text.text(0.05, 0.99, r'RA = %.5f'%(infolist[0]), fontsize=12)
        ax_text.text(0.05, 0.96, r'DEC = %.5f'%(infolist[1]), fontsize=12)
        ax_text.text(0.05, 0.93, r'l = %.5f'%(infolist[2]), fontsize=12)
        ax_text.text(0.05, 0.90, r'b = %.5f'%(infolist[3]), fontsize=12)
        ax_text.text(0.05, 0.87, r'gPSFMag = %s'%(infolist[4]), fontsize=12)
        ax_text.text(0.05, 0.84, r'rPSFMag = %s'%(infolist[5]), fontsize=12)
        ax_text.text(0.05, 0.81, r'iPSFMag = %s'%(infolist[6]), fontsize=12)
        ax_text.text(0.05, 0.78, r'zPSFMag = %s'%(infolist[7]), fontsize=12)
        ax_text.text(0.05, 0.75, r'yPSFMag = %s'%(infolist[8]), fontsize=12)
        ax_text.text(0.05, 0.63, r'w1mpro = %s'%(infolist[9]), fontsize=12)
        ax_text.text(0.05, 0.60, r'w2mpro = %s'%(infolist[10]), fontsize=12)
        ax_text.text(0.05, 0.57, r'w3mpro = %s'%(infolist[11]), fontsize=12)
        ax_text.text(0.05, 0.54, r'parallax = %s'%(infolist[12]), fontsize=12)
        ax_text.text(0.05, 0.51, r'pmra = %s'%(infolist[13]), fontsize=12)
        ax_text.text(0.05, 0.48, r'pmdec = %s'%(infolist[14]), fontsize=12)
        ax_text.text(0.05, 0.45, r'FIRST = %s'%(infolist[15]), fontsize=12)
        ax_text.text(0.05, 0.42, r'XMM = %s'%(infolist[16]), fontsize=12)
        ax_text.text(0.05, 0.39, r'ROSAT = %s'%(infolist[17]), fontsize=12)
        ax_text.text(0.05, 0.36, r'simbad (main)= %s'%(infolist[18]), fontsize=12)
        ax_text.text(0.05, 0.33, r'simbad (other)= %s'%(infolist[19]), fontsize=12)
        ax_text.text(0.05, 0.30, r'simbad (redshift)= %s'%(infolist[20]), fontsize=12)
        ax_text.text(0.05, 0.27, r'PSF - Aper = %s'%(infolist[21]), fontsize=12)
        ax_text.text(0.05, 0.24, r'distance to locus = %s'%(infolist[22]), fontsize=12)
        ax_text.text(0.05, 0.21, r'PosDiff = %s'%(infolist[23]), fontsize=12)

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

        if flag:
            ax_img.imshow(data_list[ax_idx], cmap='gray',\
                          interpolation='none', origin='lower')
        else:
            ax_img.imshow(np.full((2, 2), np.NaN), cmap='gray',\
                          interpolation='none', origin='lower')


    plt.suptitle(title, fontsize=16)
    plt.savefig(output)
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

def plot_catalog(catalog_file, spec_dir, specinfo_file, cutout_dir):

    # Define constants #
    catalog_data = Table.read(catalog_file)
    spec_info = Table.read(specinfo_file)

    ra_all_candidate = catalog_data['raStack']
    dec_all_candidate = catalog_data['decStack']

    spectrum_list = find_sdss_spec(\
                ra_all_candidate, dec_all_candidate, spec_info, radius=2)

#    print(len(catalog_data))

    mag_keys = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag',\
                'w1mpro', 'w2mpro', 'w3mpro']

    llist = np.array([4750, 6220, 7630, 9050, 10050,\
                      33680, 46180, 120820])/1e4
    nulist = 3e14/llist

    zplist = np.array([0, 0, 0, 0, 0,\
                      2.699, 3.339, 5.174])

    magerr_keys =\
        ['gPSFMagErr', 'rPSFMagErr', 'iPSFMagErr', 'zPSFMagErr', 'yPSFMagErr', \
         'w1sigmpro', 'w2sigmpro', 'w3sigmpro']

    cutout_bands = ['SDSSu', 'PSg', 'PSr', 'PSi', 'PSz', 'PSy',\
                   'LSg', 'LSr', 'LSz']

    sdss_dir = dir_data + '/catalog/cutouts/PS_WISE_GAIA_stationary/Morphriz/sdss'
    ps_dir = dir_data + '/catalog/cutouts/PS_WISE_GAIA_stationary/Morphriz/PanStarrs'
    ls_dir = dir_data + '/catalog/cutouts/PS_WISE_GAIA_stationary/Morphriz/Legacy'

    for index in range(len(catalog_data)):

        # get jname #

        ra = catalog_data['raStack'][index]
        dec = catalog_data['decStack'][index]

        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        coords_str_3 = coords.to_string('hmsdms', sep='', precision=3)
        name_3 = np.char.replace(coords_str_3, ' ', '')

        jname = 'J' + str(name)
        pname = 'P' + str(name)
        jname_3 = 'J' + str(name_3)
        pname_3 = 'P' + str(name_3)

        # get magnitudes #

        maglist, magerrlist =\
                read_mag(catalog_data, index, mag_keys, magerr_keys)

        maglist_AB = maglist + zplist

        flux = 3631e-23*10**(-0.4*(maglist_AB))
        fluxerr = 3631e-23*10**(-0.4*(maglist_AB))*np.log(10)*0.4*magerrlist

        lfl = flux * nulist
        lflerr = fluxerr * nulist

        # mask magnitudes #

        mask1 = (~np.isnan(maglist))
        mask2 = (~np.isnan(magerrlist))
        mask = (mask1 & mask2)

        lfl_input = lfl[mask]
        lflerr_input = lflerr[mask]
        llist_input = llist[mask]

        # get image names #

        sdss_files = [sdss_dir + '/%s/%s.sdssu.fits'%(jname, jname)]
        ps_files = [ps_dir + '/%s/%s.%s.fits'%(jname, pname_3, band)\
                      for band in 'grizy']
        ls_files = [ls_dir + '/%s/%s.%s.fits'%(jname, jname_3, band)\
                      for band in 'grz']

        filenames = sdss_files + ps_files + ls_files

        # get mags to plot #

        mag_to_plot_keys = ['gPSFMag', 'rPSFMag', 'rPSFMag', 'iPSFMag',\
                           'rPSFMag', 'iPSFMag', 'iPSFMag', 'zPSFMag']
        mag_to_plot = [catalog_data[key][index] for key in mag_to_plot_keys]

        # get info #

        # coordinates
        infolist = []
        infolist.append(catalog_data['raStack'][index])
        infolist.append(catalog_data['decStack'][index])
        infolist.append(catalog_data['l_gaia'][index])
        infolist.append(catalog_data['b_gaia'][index])

        # magnitudes
        gstack_str = '%.2f $\pm$ %.2f'%(maglist[0], magerrlist[0])
        rstack_str = '%.2f $\pm$ %.2f'%(maglist[1], magerrlist[1])
        istack_str = '%.2f $\pm$ %.2f'%(maglist[2], magerrlist[2])
        zstack_str = '%.2f $\pm$ %.2f'%(maglist[3], magerrlist[3])
        ystack_str = '%.2f $\pm$ %.2f'%(maglist[4], magerrlist[4])
        w1_str = '%.2f $\pm$ %.2f'%(maglist[5], magerrlist[5])
        w2_str = '%.2f $\pm$ %.2f'%(maglist[6], magerrlist[6])
        w3_str = '%.2f $\pm$ %.2f'%(maglist[7], magerrlist[7])

        infolist += \
                [gstack_str, rstack_str, istack_str, zstack_str, ystack_str,\
                 w1_str, w2_str, w3_str]

        # gaia
        parallax_str = '%.2f $\pm$ %.2f'%(catalog_data['parallax'][index],\
                                        catalog_data['parallax_error'][index])
        pmra_str = '%.2f $\pm$ %.2f'%(catalog_data['pmra'][index],\
                                          catalog_data['pmra_error'][index])
        pmdec_str = '%.2f $\pm$ %.2f'%(catalog_data['pmdec'][index],\
                                          catalog_data['pmdec_error'][index])

        infolist += [parallax_str, pmra_str, pmdec_str]

        # other surveys
        FIRST_str = '%.2f'%(catalog_data['FINT'][index])
        XMM_str = '%.2f'%(catalog_data['SC_DET_ML'][index])
        ROSAT_str = '%.2f'%(catalog_data['CTS'][index])

        infolist += [FIRST_str, XMM_str, ROSAT_str]

        # simbad
        simbad_main_type = '%s'%(catalog_data['main_type'][index])
        simbad_other_type = '%s'%(catalog_data['main_type'][index])
        simbad_z = '%s'%(catalog_data['redshift'][index])

        infolist += [simbad_main_type, simbad_other_type, simbad_z]

        # extended
        magdiff = catalog_data['iPSFMag'][index]-catalog_data['iApMag'][index]\
                + catalog_data['zPSFMag'][index]-catalog_data['zApMag'][index]\
                + catalog_data['yPSFMag'][index]-catalog_data['yApMag'][index]

        extended_string = '%.2f'%(magdiff)

        # distance to the locus
        p3 = np.array([catalog_data['rPSFMag'][index] - catalog_data['iPSFMag'][index],\
                       catalog_data['iPSFMag'][index] - catalog_data['zPSFMag'][index]])
        p1 = np.array([0, 0])
        p2 = np.array([2, 1])

        dist = np.abs(np.cross(p2-p1, p3-p1) / np.linalg.norm(p2-p1))
        dist_string = '%.2f'%(dist)

        # PosDiff
        posdiff_string = '%.2f'%(catalog_data['PosDiff'][index])

        infolist += [extended_string, dist_string, posdiff_string]

        # read spectrum #

        if index in spectrum_list[0]:
            mask_thisq = (index==spectrum_list[0])
            specfile = spec_dir + '/' + str(spectrum_list[1][mask_thisq][0])
            plot_style = 'SPEC'
        else:
#            continue
            specfile=''
            plot_style = 'SED'

        # save plots #

        n_output = 0
        output = cutout_dir+'/%s_%d.pdf'%(jname, n_output)
        flag_output = os.path.exists(output)

        while flag_output:
            n_output += 1
            output = cutout_dir+'/%s_%d.pdf'%(jname, n_output)
            flag_output = os.path.exists(output)

        plot_oneobj(filenames, cutout_bands, 5,\
                    llist_input, lfl_input, lflerr_input, \
                    title=jname, mag_to_plot=mag_to_plot,\
                    output=output, infolist=infolist,\
                    style=plot_style, specfile=specfile)

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

#    print(len(plate))
#    print(len(fiber))
#    print(len(mjd))

    filenames = np.array(['spec-%04d-%05d-%04d.fits'%\
                 (plate[index], mjd[index], fiber[index]) \
                for index in range(len(plate))], dtype=str)

#    print(filenames)
    return (orig_index, filenames)

def main():
    catalog_file = dir_data +\
        '/catalog/catalog_download/2019_LDSS3/'\
        + 'PS_WISE_GAIA_Stationary_Morphriz_simbad_2.fits'
    dir_spec = dir_data \
            + '/catalog/cutouts/PS_WISE_GAIA_stationary/MorphPos/spec'
    specinfo_file = dir_spec + '/optical_search_155613.csv'
    output_dir = dir_data\
        + '/catalog/cutouts/PS_WISE_GAIA_stationary/Morphriz/plot_sed'

    if not os.path.exists(output_dir):
        os.system('mkdir -p ' + output_dir)

    plot_catalog(catalog_file, dir_spec, specinfo_file, output_dir)

if __name__=='__main__':
    main()
