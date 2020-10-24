import os, sys
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, FK5

def transform_coordinate(ra, dec, equinox1, equinox2,\
                         frame='icrs', unit=u.deg):

    coords = SkyCoord(ra=ra, dec=dec, unit=unit,\
                      frame=frame, equinox=equinox1)

    coords_standard = SkyCoord(ra=1.0, dec=2.0, unit=u.deg,\
                              frame=frame, equinox=equinox2)
    newcoords = coords.transform_to(coords_standard)

    return (newcoords.ra.deg, newcoords.dec.deg)

def get_system_size(ra_list, dec_list):
    coord = SkyCoord(ra=ra_list, dec=dec_list, unit='deg')

    maxdist = 0
    for index in range(len(coord)):
        coord_oneobj = coord[index]
        dist = coord_oneobj.separation(coord).arcsecond
        dist_arcsec = np.array(dist, dtype=float)
        maxdist_temp = np.max(dist_arcsec)
        if maxdist_temp > maxdist:
            maxdist = maxdist_temp

    return maxdist

if __name__=='__main__':
    ra_list = [10,10.01]
    dec_list = [10.1, 10.0]
    print(get_system_size(ra_list, dec_list))

#    main()
