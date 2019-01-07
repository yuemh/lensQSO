import numpy as np
import os, sys
import mylib as ml
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table

dir_root = os.path.abspath('../../')
dir_code = dir_root + '/code'
dir_data = dir_root + '/data'

class SurveyQuery(object):
    def __init__(self):
        pass

    def xmatch(self, coords, radius = 2./3600.):
        data = self.read_data()

        ralist_survey = data[self.ra_key]
        declist_survey = data[self.dec_key]

        ralist_object, declist_object = coords
        matched = ml.crossmatch(ralist_object, declist_object,\
                                ralist_survey, declist_survey,\
                                radius = radius)

        matched_index = matched[1]
        matched_table = data[matched_index]
        return matched_table

    def read_data(self):
        tbl_data = Table.read(self.datafile)
        return tbl_data


class FIRST(SurveyQuery):
    def __init__(self):
        self.datafile = dir_data + '/catalog/first_14dec17.fits.gz'
        self.ra_key = 'RA'
        self.dec_key = 'DEC'


class ROSAT2(SurveyQuery):
    def __init__(self):
        self.datafile = dir_data + '/catalog/cat2rxs.fits.gz'
        self.ra_key = 'RA_DEG'
        self.dec_key = 'DEC_DEG'


def main():
    firstsurvey = FIRST()
    rass = ROSAT2()

    testcoord = ([162.5302917], [30.6770889])

    print(rass.xmatch(testcoord))

if __name__ == '__main__':
    main()
