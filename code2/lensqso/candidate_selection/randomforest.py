import os, sys
import numpy as np
from astropy.table import Table

from sklearn.ensemble import RandomForestClassifier

def construct_trainingset():
    do_something = 1

    '''
    Features to use:
        ver 0: PS g, r, i, z, y, VHS J, WISE W1, W2 (current ver)

    '''

    # lensed quasars

    '''
    This sample is constructed by combining the flux of an LRG and a quasar
    LRG flux (magnitude) from SDSS LRG catalog, or simulated + luminosity function.
    Quasar flux from simqso.

    The magnification draw from an analytical distribution.
    '''

    # double M-stars

    '''
    S82 standard stars + errors?
    '''

    # emission line galaxies

