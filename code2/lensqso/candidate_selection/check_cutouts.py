import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord

dir_root = os.path.abspath(os.getcwd()+'/../../')
dir_data = dir_root + '/data'
dir_catalog = dir_data + '/catalog/catalog_download/2019_LDSS3'

def getinfo(tbl, jname, ra_col='raStack', dec_col='decStack'):
    coord = SkyCoord.from_name(jname)
    #coord_str = coord.to_string('hmsdms', sep='', precision=2)
    #name = np.char.replace(coords_str, ' ', '')

    coord_tbl = SkyCoord(ra=tbl[ra_col], dec=tbl[dec_col], unit='deg')

    idx, d2d, d3d = coord.match_to_catalog_sky(coord_tbl)
    for key in tbl.colnames:
        print('%s:'%(key), tbl[key][idx])

def check_images(catalog):
    '''
    Priority=0: Very likely
    Priority=1: Good shape AND Good SED
    Priority=2: Good shape OR Good SED
    Priority=3: Others
    '''
    catalog_new = catalog.sort('raStack')

    if os.path.exists(dir_catalog + '/info.csv'):
        existing_info = Table.read(dir_catalog + '/info.csv')
        ndone = len(existing_info)
        print('Objects inspected:', ndone)
        jname_list = list(existing_info['JNAME'])
        priority_list = list(existing_info['PRIORITY'])
        note_list = list(existing_info['NOTE'])
    else:
        ndone = 0
        print('Objects inspected:', 0)
        jname_list = []
        priority_list = []
        note_list = []

    for index in range(ndone, len(catalog)):
        ra = catalog['raStack'][index]
        dec = catalog['decStack'][index]

        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        coord_str = coord.to_string('hmsdms', sep='', precision=2)
        name = np.char.replace(coord_str, ' ', '')
        jname = 'J' +str(name)

        print(jname)

        priority = input('Proprity: ')
        change = True
        while change:
            flag = input('Priority is %s. Is it OK?'%(priority))
            if flag=='y':
                change = False
            else:
                priority = input('Proprity: ')

        note = input('Note: ')

        jname_list.append(jname)
        priority_list.append(int(priority))
        note_list.append(note)

        info_tbl = Table({'JNAME': jname_list,\
                          'PRIORITY': priority_list,\
                          'NOTE': note_list})

        info_tbl.write(dir_catalog + '/info.csv', overwrite=True)

def main():
    catalog_file = dir_catalog + \
            '/PS_WISE_GAIA_multi_unique_goodphot_simbad.fits'
    catalog = Table.read(catalog_file)
    check_images(catalog)

if __name__=='__main__':
    main()
