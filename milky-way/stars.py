#!/usr/bin/python3
# Import statements
import numpy as np
import astropy as ap

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SphericalDifferential

import matplotlib.pyplot as plt

# given RA, dec, distance from Sun D, proper motions in RA and dec

# 1) Convert to 6D cartesian coordinates
# - Convert from RA, dec to galactic coordinates
# - Convert to heliocentric Cartesian coordinates
# - Move origin to get galactocentric coordinates
# STONE THE CROWS astropy can do this in one line!
# This sorts positions, how to get velocities?

def stars(infile):
    # Create structured datatype correspond to given format
    # and load data file
    record = np.dtype([('ra'  , 'f'),
                      ('dec'  , 'f'),
                      ('d'    , 'f'),
                      ('vlos' , 'f'),
                      ('pmra' , 'f'),
                      ('pmdec','f')])
    data = np.loadtxt(infile,dtype=record,skiprows=1)

    # This converts the input data to Cartesian position components
    data_gcen = to_gcen(data)

    # astropy.SkyCoords doesn't support doing the velocity components though
    # so we need to do these manually

    # plot positions
    from matplotlib import rc
    rc('text', usetex=True)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.scatter(data_gcen.x,data_gcen.y) 
    plt.show()

def to_gcen(data):
    return SkyCoord(ra=data['ra'] * u.degree,
                    dec=data['dec'] * u.degree,
                    distance=data['d'] * u.kpc,
                    frame='icrs').galactocentric

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str)
    args = parser.parse_args()
    
#    np.random.seed(42)
    
    stars(args.filename)
