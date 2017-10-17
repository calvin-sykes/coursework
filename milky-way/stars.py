#!/usr/bin/python3
# Import statements
import numpy as np
import astropy as ap

from astropy import units as u
from astropy.coordinates import Galactocentric
from astropy.coordinates import ICRS
from astropy.coordinates import SphericalRepresentation

import matplotlib.pyplot as plt

# given RA, dec, distance from Sun D, proper motions in RA and dec

# 1) Convert to 6D cartesian coordinates
# - Convert from RA, dec to galactic coordinates
# - Convert to heliocentric Cartesian coordinates
# - Move origin to get galactocentric coordinates
# STONE THE CROWS astropy can do this in one line!
# This sorts positions, how to get velocities?

def vel_anisotropy(tan_dev, rad_dev):
    return 1 - tan_dev**2 / (2 * rad_dev**2)

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

    # astropy.SkyCoords doesn't support automatically converting the velocity components
    # so we need to transform the data manually
    icrs = ICRS(ra=data['ra'] * u.degree,
                dec=data['dec'] * u.degree,
                distance=data['d'] * u.kpc,
                pm_ra_cosdec=data['pmra'] * u.mas / u.yr * np.cos(data['dec']),
                pm_dec=data['pmdec'] * u.mas / u.yr,
                radial_velocity=data['vlos'] * u.km / u.s)
    # Cartesian galactocentric coordinates
    gal_cen = icrs.transform_to(Galactocentric)
    # Spherical galactocentric coordinates
    gal_cen_sph = gal_cen.represent_as(SphericalRepresentation)
    # Spherical velocity coordinates (from http://www.astrosurf.com/jephem/library/li110spherCart_en.htm)
    vr = ((gal_cen.x * gal_cen.v_x + gal_cen.y * gal_cen.v_y + gal_cen.z * gal_cen.v_z)
          / gal_cen_sph.distance)
    vtheta = ((gal_cen.v_x * gal_cen.y - gal_cen.v_y * gal_cen.x)
              / (gal_cen.x**2 + gal_cen.y**2))
    vphi = ((gal_cen.z * (gal_cen.x * gal_cen.v_x + gal_cen.y * gal_cen.v_y)
             - gal_cen.v_z * (gal_cen.x**2 + gal_cen.y**2))
            / (gal_cen_sph.distance**2 * np.sqrt(gal_cen.x**2 + gal_cen.y**2)))
    # v**2 = vr**2 + vtan**2 --> vtan = sqrt(v**2 - vr**2
    vtan = np.sqrt(gal_cen.v_x**2 + gal_cen.v_y**2 + gal_cen.v_z**2 - vr**2)
    out_data = np.column_stack((gal_cen.x, gal_cen.y, gal_cen.z, gal_cen.v_x, gal_cen.v_y, gal_cen.v_z))
    cols = ('x[kpc]', 'y[kpc]', 'z[kpc]', 'v_x[km/s]', 'v_y[km/s]', 'v_z[km/s]')
    np.savetxt('6d.txt', out_data, fmt='%9.4f', header=' '.join(cols))    

    # plot positions
    from matplotlib import rc
    rc('text', usetex=True)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.scatter(gal_cen.x, gal_cen.y, s=2) 
    
    # radius histogram
    #rads = np.sqrt(gal_cen.x**2 + gal_cen.y**2 + gal_cen.z**2)
    hist_fig = plt.figure()
    plt.hist(gal_cen_sph.distance, bins='auto')

    # velocities
    vel_fig = plt.figure()
    plt.hist(vr, bins='auto', label='vr')
    plt.hist(vtheta, bins='auto', label='vtheta')
    plt.hist(vphi, bins='auto', label='vphi')
    plt.hist(vtan, bins='auto', label='vtan')
    plt.legend()

    vr_dev = np.std(vr)
    vtheta_dev = np.std(vtheta)
    vphi_dev = np.std(vphi)
    vtan_dev = np.std(vtan)
    v_anis = vel_anisotropy(vtan_dev, vr_dev)
    
    print(vr_dev, vtheta_dev, vphi_dev, vtan_dev, v_anis)
    
    
    plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str)
    args = parser.parse_args()
    
#    np.random.seed(42)
    
    stars(args.filename)
