import os, sys
sys.path.append(os.path.abspath('../../'))
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u
from lensqso.sed import lens_sed as sed
import mylib.spectrum.spec_measurement as spec

#from astroML.datasets import fetch_sdss_S82standards
from sklearn.manifold import LocallyLinearEmbedding


#np.random.seed(1)
dir_data = os.path.abspath(os.getcwd()+'/../../../data/')

class ccplot(object):
    def __init__(self, xlabel, ylabel):
        self.xlabel = xlabel
        self.ylabel = ylabel

    def add_region(self, criteria):
        do_something = 1

class stellar_locus(object):
    def __init__(self):
        self.name='stellar'

def stellar_locus_test():
    data = fetch_sdss_S82standards()

    # select the first 10000 points
    data = data[:10000]

    print(data.dtype.names)

    # select the mean magnitudes for g, r, i
    g = data['mmu_g']
    r = data['mmu_r']
    i = data['mmu_i']
    z = data['mmu_z']

    sdss_download = Table.read('../codedata/pointsource_test_v0_myue.fit')
    sdss_download = sdss_download[(sdss_download['rPSFMagErr']<0.3)\
                                  &(sdss_download['iPSFMagErr']<0.2)
                                 &(sdss_download['zPSFMagErr']<0.2)]
    print(len(sdss_download))
    g = sdss_download['gPSFMag']
    r = sdss_download['rPSFMag']
    i = sdss_download['iPSFMag']
    z = sdss_download['zPSFMag']

    #------------------------------------------------------------
    # Plot the g-r vs r-i colors
    fig, ax = plt.subplots(figsize=(5, 3.75))
    ax.plot(r - i, i - z, marker='.', markersize=0.1,
                    color='black', linestyle='none')

#    ax.set_xlim(-0.6, 2.0)
#    ax.set_ylim(-0.6, 2.5)

    ax.set_xlabel(r'${\rm r - i}$')
    ax.set_ylabel(r'${\rm i - z}$')

    plt.show()

    ### features ###

    ug = data['mmu_u'] - data['mmu_g']
    gr = data['mmu_g'] - data['mmu_r']
    ri = data['mmu_r'] - data['mmu_i']
    iz = data['mmu_i'] - data['mmu_z']

def ccplot_test():

    ### Wang+16 ccplot
    ricolor_Wang16 = [1, 1, 1.02/0.625, 3, 3]
    izcolor_Wang16 = [0, 0.325, 0.72, 0.72, 0]

    ### low-z LRG color
    ricolor_eBOSS = [0.98, 0.98, 3, 3]
    izcolor_eBOSS = [3, 0.625, 0.625, 3]

    ### high-z LRG color


    SDSSu = spec.read_filter('SDSS', 'SDSSu')
    SDSSg = spec.read_filter('SDSS', 'SDSSg')
    SDSSr = spec.read_filter('SDSS', 'SDSSr')
    SDSSi = spec.read_filter('SDSS', 'SDSSi')
    SDSSz = spec.read_filter('SDSS', 'SDSSz')

    W1 = spec.read_filter('WISE', 'W1', uflag=2)
    W2 = spec.read_filter('WISE', 'W2', uflag=2)

    all_filters = [SDSSu, SDSSg, SDSSr, SDSSi, SDSSz, W1, W2]

    jaguar_file = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1_spec_5A.fits.gz'
    jaguar_info_masterfile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.fits'
    jaguar_info_sigmafile = dir_data +\
            '/simimg/GalaxySED/JAGUAR/JADES_Q_mock_r1_v1.1.sigma.fits'

    LRG_template_file = './LRG_template.csv'

    if not os.path.exists(LRG_template_file):
        g_flux, g_wave, g_redshift, g_sigma =\
        sed.random_galaxysample_JAGUAR(jaguar_file, jaguar_info_sigmafile,\
                                        sigma_lim=200, z_lim=1, seed=-1)

        print(g_redshift, g_sigma)
        gal_spec = sed.SED(wavelength=g_wave, value=g_flux)
        gal_spec.to_luminosity(redshift=g_redshift)

        LRG_template = Table({'wavelength': gal_spec.wavelength,\
                              'luminosity': gal_spec.value})
        LRG_template.write(LRG_template_file)

    else:
        LRG_template = Table.read(LRG_template_file, format='csv')

        g_wave = LRG_template['wavelength']
        g_lum = LRG_template['luminosity']

        gal_spec = sed.SED(wavelength=g_wave, value=g_lum,\
                          units=[u.Angstrom, u.erg/u.s/u.Angstrom], \
                          mode='LUMI')

    gal_spec.plot()

    ### get a galaxy spectrum 
    plt.close('all')
    fig, ax = plt.subplots()

    ax.fill(ricolor_Wang16, izcolor_Wang16, alpha=0.3)
    ax.fill(ricolor_eBOSS, izcolor_eBOSS, alpha=0.3)

#    gal_spec = sed.SED(wavelength=g_wave, value=g_flux)
#    gal_spec.to_luminosity(redshift=g_redshift)

    ### get a quasar spectrum
    qso_flux = sed.random_quasarsample_ian(5, -27)
    q_flux, q_wavelength, q_redshift = qso_flux

    qso_spec = sed.SED(wavelength=q_wavelength,\
                       value=q_flux/(1+q_redshift[0]))
    qso_spec.to_luminosity(redshift=q_redshift[0])

    g_color1 = []
    g_color2 = []
    q_color1 = []
    q_color2 = []

    for redshift in np.arange(0.1, 2, 0.1):
        g_mags = []

        gal_spec.to_obs(redshift)
        for filt in all_filters:
            g_mags.append(gal_spec.magnitude(filt))
        print(g_mags[4])
        g_color1.append(g_mags[1]-g_mags[2])
        g_color2.append(g_mags[1]-g_mags[4])

        gal_spec.to_luminosity(redshift)

    for redshift in np.arange(4., 6, 0.1):
        q_mags = []

        qso_spec.to_obs(redshift)
        for filt in all_filters:
            q_mags.append(qso_spec.magnitude(filt))

        q_color1.append(q_mags[1]-q_mags[2])
        q_color2.append(q_mags[1]-q_mags[4])

#        if redshift < 1:
#            ax.plot([mags[2]-mags[3]], [mags[3]-mags[4]], 'bo')
#        elif redshift >=1:
#            ax.plot([mags[2]-mags[3]], [mags[3]-mags[4]], 'ro')

        qso_spec.to_luminosity(redshift)

    plt.plot(g_color1, g_color2, 'b^-')
    plt.plot(q_color1, q_color2, 'ro-')

    plt.show()

def DES_star_test():
    ### Wang+16 ccplot
    ricolor_Wang16 = [1, 1, 1.02/0.625, 3, 3]
    izcolor_Wang16 = [0, 0.325, 0.72, 0.72, 0]

    DESstarphot = Table.read('../codedata/download_catalog/DES_stars_v3_VHS_WISE.fits')
    print(len(DESstarphot))

    DESstarphot = DESstarphot[(DESstarphot['magerr_aper_8_g']<10000)\
                             &(DESstarphot['magerr_aper_8_r']<0.2)\
                             &(DESstarphot['magerr_aper_8_i']<0.2)\
                             &(DESstarphot['mag_aper_8_i']<21)\
                             &(DESstarphot['magerr_aper_8_z']<0.2)\
                              &(DESstarphot['magerr_aper_8_y']<0.2)\
                             &(DESstarphot['spread_model_i']<1000)]

    print(len(DESstarphot))

    DESstarphot = DESstarphot[(DESstarphot['flags_g']<4)\
                              &(DESstarphot['flags_r']<4)\
                              &(DESstarphot['flags_i']<4)\
                              &(DESstarphot['flags_z']<4)\
                              &(DESstarphot['flags_y']<4)]
    print(len(DESstarphot))

    DESstarphot_mask = ((DESstarphot['mag_aper_8_r']-DESstarphot['mag_aper_8_i'])*2-0.7\
                        <(DESstarphot['mag_aper_8_r']-(DESstarphot['w1mpro'] + 2.699)))\
                        &(DESstarphot['mag_aper_8_r']-DESstarphot['mag_aper_8_i']>0.2)\
                        &(DESstarphot['mag_aper_8_r']-DESstarphot['w1mpro']-2.699>0.1)\
                        &(DESstarphot['mag_aper_8_g']-DESstarphot['mag_aper_8_r']>1)

    DESstarphot_selected = DESstarphot[DESstarphot_mask]
    print(len(DESstarphot_selected))
    gr = DESstarphot['mag_aper_8_g'] - DESstarphot['mag_aper_8_r']
    ri = DESstarphot['mag_aper_8_r'] - DESstarphot['mag_aper_8_i']
    iz = DESstarphot['mag_aper_8_i'] - DESstarphot['mag_aper_8_z']
    zy = DESstarphot['mag_aper_8_z'] - DESstarphot['mag_aper_8_y']
    rw1 = DESstarphot['mag_aper_8_r'] - (DESstarphot['w1mpro'] + 2.699)
    iw1 = DESstarphot['mag_aper_8_i'] - (DESstarphot['w1mpro'] + 2.699)
    w1w2 = (DESstarphot['w1mpro'] + 2.699) - (DESstarphot['w2mpro'] + 3.339)


    lens = Table.read('../codedata/trainingsample/lensqso.fits')

    galstar = Table.read('../codedata/trainingsample/galstar.fits')
    starstar = Table.read('../codedata/trainingsample/starstar.fits')

#    '''
    lens = lens[(lens['magerr'][:, 4]<0.3)\
               &(lens['magerr'][:, 3]<0.3)\
               &(lens['magerr'][:, 2]<0.3)\
               &(lens['magerr'][:, 1]<0.3)]
#    '''
    galstar = galstar[galstar['magerr'][:, 6]<0.1]
    starstar = starstar[starstar['magerr'][:, 6]<0.1]

    print(len(lens))
    lens1 = lens[(lens['mag'][:, 1]-lens['mag'][:, 2])*0.45-0.1\
                >(lens['mag'][:, 2]-lens['mag'][:, 3])]

    lens2 = lens[(lens['mag'][:, 1]-lens['mag'][:, 2])*0.45-0.1\
                <(lens['mag'][:, 2]-lens['mag'][:, 3])]

#    plt.hist(lens_selected['mag0'][:, 7], bins=np.arange(16, 20, 0.1))
#    plt.show()

    idx0 = 1
    idx1 = 2
    idx2 = 2
    idx3 = 3

#    plt.hist(lens['mag'][:, 2], bins=np.arange(15, 25, 0.01))
#    plt.show()
#    plt.fill(ricolor_Wang16, izcolor_Wang16, alpha=0.3)


    plt.plot(ri, iz, 'k,', alpha=0.1)
#    plt.plot(lens['mag'][:, idx0]-lens['mag'][:, idx1],\
#             lens['mag'][:, idx2]-lens['mag'][:, idx3], 'g,', alpha=1)
    plt.plot(lens1['mag'][:, idx0]-lens1['mag'][:, idx1],\
             lens1['mag'][:, idx2]-lens1['mag'][:, idx3], 'r,',\
             label='QSO-dominated', alpha=0.05)
    plt.plot(lens2['mag'][:, idx0]-lens2['mag'][:, idx1],\
             lens2['mag'][:, idx2]-lens2['mag'][:, idx3], 'b,',\
             label='GAL-dominated', alpha=0.05)

#    plt.plot(lens['qsomag0'][:, idx0]-lens['qsomag0'][:, idx1],\
#             lens['qsomag0'][:, idx2]-lens['qsomag0'][:, idx3], 'm.')
#    plt.plot(lens['galmag0'][:, idx0]-lens['galmag0'][:, idx1],\
#             lens['galmag0'][:, idx2]-lens['galmag0'][:, idx3], 'r.')

#    plt.plot([0.4, 2], [0.1, 3.3])

#    plt.plot(galstar['mag'][:, idx0]-(galstar['mag'][:, idx1]),\
#             (galstar['mag'][:, idx2])-(galstar['mag'][:, idx3]-0.916),\
#             'y.')
#    plt.plot(lens['qsomag0'][:, idx0]-(lens['qsomag0'][:, idx1]),\
#             (lens['qsomag0'][:, idx2])-(lens['qsomag0'][:, idx3]),\
#             'g.')
#    plt.xlim([-10, 10])
#    plt.ylim([-10, 10])

    plt.legend()
    plt.xlabel('DESr - DESi')
    plt.ylabel('DESi - DESz')
    plt.show()


def main():
#    stellar_locus_test()
    ccplot_test()
#    DES_star_test()

if __name__=='__main__':
    main()

