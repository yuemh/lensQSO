import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mylib.spec_measurement as spec
import mylib.zscale as zs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

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

def plot_oneobj(filenames, wavelength, flux, fluxerr,\
                labels='ugrizy', title='', output=''):
    Nimg = len(filenames)
    print(Nimg)
    data_list = []

    for filename in filenames:
        try:
            data = fits.open(filename)[0].data
            data_list.append(data)
        except FileNotFoundError:
            try:
                new_filename = 'J' + filename[1:]
                data = fits.open(filename)[0].data
                data_list.append(data)

            except FileNotFoundError:
                print('No file named %s'%(filename))
                data_list.append([])

    ### Making SED

    gs = gridspec.GridSpec(2, Nimg)
    ax_sed = plt.subplot(gs[0,:])
#    ax_sed.set_xscale("log", nonposx="clip")
#    ax_sed.set_yscale("log", nonposy="clip")
    ax_sed.errorbar(wavelength, flux, yerr=fluxerr, fmt='bo')
    ax_sed.set_xlabel('Wavelength (micron)')
    ax_sed.set_ylabel('$\lambda F_{\lambda} (erg/s/cm^2)$')
    ax_sed.set_xlim([0.3, 15])

    r06_sed = richards_sed(5)
    sed_flux_at_wave = np.interp(x=wavelength[3], xp=r06_sed[0], fp=r06_sed[1])

    scale = np.median(flux[3]/sed_flux_at_wave)
#    print(sed_flux_at_wave)

#    print(wavelength, scale)
    ax_sed.plot(r06_sed[0], r06_sed[1]*scale, label='Richard+06, z=5')
    ax_sed.legend()

    ### Making cutouts
    if len(data_list)==Nimg:
        index_to_plot = range(Nimg)
    else:
        index_to_plot = range(len(data_list))

    print(index_to_plot)
    for ax_idx in index_to_plot:
        ax_img = plt.subplot(gs[1,ax_idx])
        ax_img.set_axis_off()
        data = data_list[ax_idx]
        if len(data)<1:
            continue
        clow, chigh = zs.zscale(data, nsamples=100)
        ax_img.imshow(data_list[ax_idx], cmap='gray',\
                             interpolation='none')#, clim=[clow, chigh])
        ax_img.set_title(labels[ax_idx])

    plt.suptitle(title)

    if len(output):
        plt.savefig(output)
        plt.close('all')

    else:
        plt.show()

def plot_catalog(catalog_file, cutout_dir):
    mag_keys = ['gApMag', 'rApMag', 'iApMag', 'zApMag', 'yApMag',\
            'j_1AperMag3', 'hAperMag3', 'kAperMag3', 'W1mag', 'W2mag', \
            'W3mag']

    llist = np.array([4750, 6220, 7630, 9050, 10050, 12480, 16310, 22010, 
                  33680, 46180, 120820])/1e4
    nulist = 3e14/llist

    zplist = np.array([0, 0, 0, 0, 0, 0.938, 1.379, 1.900, 2.699, 3.339, 5.174])

    magerr_keys =\
        ['gApMagErr', 'rApMagErr', 'iApMagErr', 'zApMagErr', 'yApMagErr', \
        'j_1AperMag3Err', 'hAperMag3Err', 'kAperMag3Err', \
         'e_W1mag', 'e_W2mag', 'e_W3mag']

    catalog_data = Table.read(catalog_file)

    cutout_bands = ['sdssu', 'g', 'r', 'i', 'z', 'y']

    for index in range(len(catalog_data)):
        maglist = np.array([catalog_data[key][index] for key in mag_keys])
        magerrlist = np.array([catalog_data[key][index]\
                               for key in magerr_keys])

        maglist_AB = maglist + zplist

        flux = 3631e-23*10**(-0.4*(maglist_AB))
        fluxerr = 3631e-23*10**(-0.4*(maglist_AB))*np.log(10)*0.4*magerrlist

        lfl = flux * nulist
        lflerr = fluxerr * nulist

        jname = catalog_data['JName'][index]
        for root, whatever, namedir in os.walk(cutout_dir + '/' + jname):
            for name in namedir:
                if name[0]=='P':
                    nameroot = name[:-7]
#                break
#            break

        filenames = [cutout_dir + '/' + jname + '/' + jname + '.sdssu.fits']

        for band in cutout_bands[1:]:
            filenames.append(cutout_dir + '/' + jname + '/' +\
                             nameroot + '.%s.fits'%(band))

        plot_oneobj(filenames, llist, lfl, lflerr, \
                   title=jname)# , output='./plots/%s.pdf'%(jname))

def main():
    plot_catalog(dir_data + '/catalog/PS1_UKIDSS_WISE_GAIA_test.fits',\
                dir_data + '/catalog/cutouts')

if __name__=='__main__':
    main()
