#!/usr/bin/python3
# Import statements
import numpy as np
import astropy as ap

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactocentric
from astropy.coordinates import ICRS

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
    intype = np.dtype([('ra'  , 'f'),
                      ('dec'  , 'f'),
                      ('d'    , 'f'),
                      ('vlos' , 'f'),
                      ('pmra' , 'f'),
                      ('pmdec','f')])
    data = np.loadtxt(infile,dtype=intype,skiprows=1)

    # astropy.SkyCoords doesn't support doing the velocity components though
    # so we need to do these manually
    icrs = ICRS(ra=data['ra'] * u.degree,
                dec=data['dec'] * u.degree,
                distance=data['d'] * u.kpc,
                pm_ra_cosdec=data['pmra'] * u.mas / u.yr * np.cos(data['dec']),
                pm_dec=data['pmdec'] * u.mas / u.yr,
                radial_velocity=data['vlos'] * u.km / u.s)
    gal_cen = icrs.transform_to(Galactocentric)

    out_data = np.column_stack((gal_cen.x, gal_cen.y, gal_cen.z, gal_cen.v_x, gal_cen.v_y, gal_cen.v_z))
    cols = ('x[kpc]', 'y[kpc]', 'z[kpc]', 'v_x[km/s]', 'v_y[km/s]', 'v_z[km/s]')
    delim = ' '# * 6
    np.savetxt('6d.txt', out_data, fmt='%9.4f', delimiter=delim, header=delim.join(cols))    

    # plot positions
    from matplotlib import rc
    rc('text', usetex=True)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    #plt.scatter(gal_cen.x,gal_cen.y) 
    #plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str)
    args = parser.parse_args()
    
#    np.random.seed(42)
    
    stars(args.filename)
