import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
import utils
from collections import Counter

dir_root = os.path.abspath(os.getcwd()+'/../../')
dir_data = dir_root + '/data'
dir_catalog = dir_data + '/catalog/catalog_download/2019_LDSS3'

'''
Info to summarize:
    SDSS Photometry / cutout
    Pan-Starrs Photometry / cutout
    Legacy Survey Photometry / cutout
    2MASS Photometry
    VHS Photometry
    WISE photometry
    FIRST
    XMM
    Gaia
    ???

Task:
    Start from a list of coordinates, match all the information listed above.

Details:
    SDSS, Pan-Starrs: match through CASJOBS
    Legacy Survey: through NOAO
    2MASS Survey: ?
    VHS Survey: ?
    WISE: (known)
    Gaia: Casjob

High_level info:
    Astrometry in all the surveys
    Light curve

'''

def clean_lensq():
    known_lens_quasar = Table.read(dir_data +\
                            '/catalog/catalog_download/lensedquasars.fits')
    print(known_lens_quasar)

    newzlist = []
    for index in range(len(known_lens_quasar)):
        try:
            newzlist.append(float(known_lens_quasar['z_qso'][index]))
        except ValueError:
            print(known_lens_quasar['z_qso'][index])
            newstr = input('What is the redshift?')
            try:
                newzlist.append(float(newstr))
            except ValueError:
                newzlist.append(np.NaN)

    objid = range(len(known_lens_quasar))
    name = np.array(known_lens_quasar['Name'], dtype=str)
    ra = np.array(known_lens_quasar['RA'], dtype=float)
    dec = np.array(known_lens_quasar['DEC'], dtype=float)
    sep = np.array(known_lens_quasar['separation'], dtype=float)
    zlens = np.array(newzlist, dtype=float)

    newtable = Table({'Objid': objid, 'RA': ra, 'DEC': dec, 'Redshift':zlens,\
                      'sep': sep})

    J0439 = Table({'Objid': [len(objid)], 'RA':[69.9461667],\
                   'DEC':[16.5710278], 'Redshift':[6.51], 'sep': [0.2]})

    newtable_all = vstack([newtable, J0439])

    newtable_all = newtable_all[newtable_all['Redshift']>0]
    newtable_all.write(dir_data +\
                '/catalog/catalog_download/lensedquasarclean.fits')

def download_sdss_info():
    # Using CASJOB #
    casjob_exe = ''

def test_panstarrs():
    data = Table.read(dir_data +\
                '/catalog/catalog_download/Lensed_PanStarrs_zhenliyi_1.fit')

    data_clean_mask = (data['iPSFMag']>0) &\
            (data['iApMag']>0) &\
            (data['zPSFMag']>0) &\
            (data['zApMag']>0) &\
            (data['yPSFMag']>0) &\
            (data['yApMag']>0)

    data_clean = data[data_clean_mask]

    print(len(data_clean))

    plt.plot(data_clean['sep'],\
             data_clean['zPSFMag']-data_clean['zApMag'], '.')
    plt.show()

    criteria1 = (data_clean['iPSFMag'] > data_clean['iApMag'])
    criteria2 = (data_clean['zPSFMag'] > data_clean['zApMag'])
    criteria3 = (data_clean['rPSFMag'] > data_clean['rApMag'])
    criteria4 = (data_clean['yPSFMag'] > data_clean['yApMag'])

    data_select = data_clean[criteria1 & criteria2 & criteria4 & criteria4]
    print(len(data_select))
    print(len(data_select)/len(data_clean))

def save_priority_table(priority_info_file, master_tbl_file):
    master_tbl = Table.read(master_tbl_file)
    priority_info = Table.read(priority_info_file)
    master_tbl.sort('raStack')

    master_tbl = hstack([master_tbl, priority_info])

    p0_mask = (priority_info['PRIORITY']==0)
    p1_mask = (priority_info['PRIORITY']==1)

    p0_tbl = master_tbl[p0_mask]
    p1_tbl = master_tbl[p1_mask]

    # full ver #
#    p0_tbl.write(dir_catalog + '/by_priority/priority_0.fits')
#    p1_tbl.write(dir_catalog + '/by_priority/priority_1.fits')

    # simple ver #
    cols_to_save = ['objID', 'raStack', 'decStack', 'JNAME', 'PRIORITY', 'NOTE']
    p0_tbl_simple = p0_tbl[cols_to_save]
    p1_tbl_simple = p1_tbl[cols_to_save]
#    p0_tbl_simple.write(dir_catalog + '/by_priority/priority_0_simple.csv')
#    p1_tbl_simple.write(dir_catalog + '/by_priority/priority_1_simple.csv')

    cols_to_save2 = ['raStack', 'decStack']
    p0_coord = p0_tbl[cols_to_save2]
    p1_coord = p1_tbl[cols_to_save2]
    p0_coord.rename_column('raStack', 'ra')
    p0_coord.rename_column('decStack', 'dec')
    p1_coord.rename_column('raStack', 'ra')
    p1_coord.rename_column('decStack', 'dec')

    allp_coord = master_tbl[cols_to_save2]
    allp_coord.rename_column('raStack', 'ra')
    allp_coord.rename_column('decStack', 'dec')
    allp_coord.write(dir_catalog + '/by_priority/allp_coord.csv')

#    p0_coord.write(dir_catalog + '/by_priority/priority_0_coord.csv')
#    p1_coord.write(dir_catalog + '/by_priority/priority_1_coord.csv')

def move_files(catalog, source_dir, object_dir):
    tbl = Table.read(catalog)

    cutout_dir = source_dir + '/plot_sed'
    ps_dir = source_dir + '/PanStarrs'
    sdss_dir = source_dir + '/SDSS'
    ls_dir = source_dir + '/Legacy'

    for index in range(len(tbl)):
        jname = str(tbl['JNAME'][index])
        ra = float(tbl['raStack'][index])
        dec = float(tbl['decStack'][index])

        coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        coords_str = coords.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coords_str, ' ', '')

        coords_str_3 = coords.to_string('hmsdms', sep='', precision=3)
        name_3 = np.char.replace(coords_str_3, ' ', '')

        jname = 'J' + str(name)
        pname = 'P' + str(name)
        jname_3 = 'J' + str(name_3)
        pname_3 = 'P' + str(name_3)

        object_dir_thisobj = object_dir + '/' + jname
        if not os.path.exists(object_dir_thisobj):
            os.system('mkdir %s'%(object_dir_thisobj))

        sdss_files = ['%s/%s.sdssu.fits'%(jname, jname)]
        ps_files = ['%s/%s.%s.fits'%(jname, pname_3, band) for band in 'grizy']
        ls_files = ['%s/%s.%s.fits'%(jname, jname_3, band) for band in 'grz']

        for filename in sdss_files:
            os.system('cp %s/%s %s/%s'\
                      %(sdss_dir, filename, object_dir, filename))

        for filename in ps_files:
            os.system('cp %s/%s %s/%s'\
                      %(ps_dir, filename, object_dir, filename))

        for filename in ls_files:
            os.system('cp %s/%s %s/%s'\
                      %(ls_dir, filename, object_dir, filename))

def get_coord_diff(catalog):
    tbl = Table.read(catalog)
    sizelist = []

    for index in range(len(tbl)):
        allra = [tbl[key][index]\
                 for key in ['gra', 'rra', 'ira', 'zra', 'yra']]
        alldec = [tbl[key][index]\
                 for key in ['gdec', 'rdec', 'idec', 'zdec', 'ydec']]

        if np.min(allra+alldec)<-100:
            sizelist.append(-999.0)
            continue

        size = utils.get_system_size(allra, alldec)
        sizelist.append(size)

    tbl['PosDiff'] = np.array(sizelist)
    tbl.write(catalog[:-5]+'_v2.fits')

    plt.hist(sizelist, bins=np.arange(0, 0.5, 0.01))
    plt.show()


def main():
    priority_info_file = dir_catalog + '/info.csv'
    master_tbl_file = dir_catalog + \
            '/PS_WISE_GAIA_multi_unique_goodphot_simbad.fits'
    master_tbl_file_stationary = dir_catalog + \
            '/PS_WISE_GAIA_Stationary_2019_LDSS3.fits'


    p1_tbl_file = dir_catalog + '/by_priority/priority1/priority_1.fits'
    source_dir = dir_data + '/catalog/cutouts/PS_WISE_GAIA_multi'
    object_dir = dir_catalog + '/by_priority/priority1'

#    move_files(p1_tbl_file, source_dir, object_dir)
#    save_priority_table(priority_info_file, master_tbl_file)
    get_coord_diff(master_tbl_file_stationary)

if __name__=='__main__':
    main()
