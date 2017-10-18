#!/usr/bin/python3
# Import statements
import numpy as np
import astropy as ap

from astropy import units as u
from astropy.coordinates import Galactocentric
from astropy.coordinates import ICRS
from astropy.coordinates import SphericalRepresentation
from astropy.coordinates import CartesianDifferential

import matplotlib.pyplot as plt

# given RA, dec, distance from Sun D, proper motions in RA and dec

# 1) Convert to 6D cartesian coordinates
# - Convert from RA, dec to galactic coordinates
# - Convert to heliocentric Cartesian coordinates
# - Move origin to get galactocentric coordinates
# STONE THE CROWS astropy can do this in one line!
# This sorts positions, how to get velocities?

def filter_outliers(vals):
    """Filter an array by removing values with magnitude greater than a limit.
    Currently this limit is fixed to be 1000, this should probably be changed! (FIXME)"""
    return vals[np.abs(vals) < 1000 * vals[0].unit]

def mean_sq(vals):
    """Calculates the mean square value of a quantity
    Parameters: the array to find the mean square value of."""
    filt = filter_outliers(vals)
    return np.mean(filt ** 2)

def vel_anisotropy(vr, vtheta, vphi):
    """Calculate the velocity anistropy parameter, beta
    Parameters: the radial, latitudinal, and longitudinal velocity components."""
    return 1 - ((mean_sq(vtheta) + mean_sq(vphi)) / (2 * mean_sq(vr)))

def stars(infile, plots):
    # Create structured datatype correspond to given format
    # and load data file
    intype = np.dtype([('ra'  , 'f'),
                      ('dec'  , 'f'),
                      ('d'    , 'f'),
                      ('vlos' , 'f'),
                      ('pmra' , 'f'),
                      ('pmdec', 'f')])
    data = np.loadtxt(infile,dtype=intype,skiprows=1)

    # astropy.SkyCoords doesn't support automatically converting the velocity components
    # so we need to transform the data manually
    icrs = ICRS(ra=data['ra'] * u.degree,
                dec=data['dec'] * u.degree,
                distance=data['d'] * u.kpc,
                pm_ra_cosdec=data['pmra'] * u.mas / u.yr * np.cos(np.radians(data['dec'] * u.degree)),
                pm_dec=data['pmdec'] * u.mas / u.yr,
                radial_velocity=data['vlos'] * u.km / u.s)
    
    # Cartesian galactocentric coordinates
    v_sun = CartesianDifferential(11.1, 12.24 + 238, 7.25, unit=u.km / u.s)
    gal_cen = icrs.transform_to(Galactocentric(galcen_v_sun=v_sun, galcen_distance=8.0 * u.kpc, z_sun=0 * u.kpc))

    # Output the 6D Cartesian coordinates
    out_file = '6d.txt'
    out_data = np.column_stack((gal_cen.x, gal_cen.y, gal_cen.z, gal_cen.v_x, gal_cen.v_y, gal_cen.v_z))
    cols = ('x[kpc]', 'y[kpc]', 'z[kpc]', 'v_x[km/s]', 'v_y[km/s]', 'v_z[km/s]')
    np.savetxt(out_file, out_data, fmt='%9.4f', header=' '.join(cols))
    print('6D coordinates written to {}'.format(out_file))

    # Spherical galactocentric coordinates
    gal_cen_sph = gal_cen.spherical
    gal_cen_sph_v = gal_cen.spherical.differentials['s']
    
    # Alternative way of getting spherial coordinates
    #test = gal_cen.copy()
    #test.representation = 'spherical'
    #print(test)

    # Calculate spherical velocity components
    # Not 100% sure that the theta/phi ones are the right way around
    # But for these purposes it shouldn't matter
    vr = gal_cen_sph_v.d_distance.to(u.km/u.s)
    vtheta = (gal_cen_sph_v.d_lat * gal_cen_sph.distance).to(u.rad * u.km/u.s) / u.rad
    vphi = (gal_cen_sph_v.d_lon * gal_cen_sph.distance).to(u.rad * u.km/u.s) / u.rad

    # Spherical velocity coordinates (from http://www.astrosurf.com/jephem/library/li110spherCart_en.htm)
    # vr = ((gal_cen.x * gal_cen.v_x + gal_cen.y * gal_cen.v_y + gal_cen.z * gal_cen.v_z)
    #       / gal_cen_sph.distance)
    # vtheta = gal_cen_sph.distance * ((gal_cen.v_x * gal_cen.y - gal_cen.v_y * gal_cen.x)
    #           / (gal_cen.x**2 + gal_cen.y**2))
    # vphi = gal_cen_sph.distance * ((gal_cen.z * (gal_cen.x * gal_cen.v_x + gal_cen.y * gal_cen.v_y)
    #          - gal_cen.v_z * (gal_cen.x**2 + gal_cen.y**2))
    #         / (gal_cen_sph.distance**2 * np.sqrt(gal_cen.x**2 + gal_cen.y**2)))
    # v**2 = vr**2 + vtan**2 --> vtan = sqrt(v**2 - vr**2)
    # vtan = np.sqrt(gal_cen.v_x**2 + gal_cen.v_y**2 + gal_cen.v_z**2 - vr**2)
    
    # Calculate sigmas for three velocity components
    vr_dev = np.std(vr)
    vtheta_dev = np.std(vtheta)
    vphi_dev = np.std(vphi)
    
    # Calculate the velocity anisotropy parameter beta
    v_anis = vel_anisotropy(vr, vtheta, vphi)

    # Calculate the rotation velocity
    v2 = gal_cen.v_x**2 + gal_cen.v_y**2 +  gal_cen.v_z**2
    v_rot_arr = np.sqrt(vtheta**2 + vphi**2)

    print('sigma_r={}\nsigma_theta={}\nsigma_phi={}\nbeta={}'.format(vr_dev, vtheta_dev, vphi_dev, v_anis))
    
    if plots:
        # plot positions
        from matplotlib import rc
        rc('text', usetex=True)
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.scatter(gal_cen.x, gal_cen.y, s=2) 

        # radius histogram
        hist_fig = plt.figure()
        plt.hist(gal_cen_sph.distance, bins='auto')

        # xyz histogram
        xyz_hist_fig = plt.figure()
        plt.hist(np.array(gal_cen.x), bins=50, alpha=0.4, label='x')
        plt.hist(np.array(gal_cen.y), bins=50, alpha=0.4, label='y')
        plt.hist(np.array(gal_cen.z), bins=50, alpha=0.4, label='z')
        plt.legend()

        # velocities
        vel_fig = plt.figure()
        plt.hist(np.array(vr), bins=50, range=(-1000,1000),alpha=0.4,label='vr')
        plt.hist(np.array(vtheta), bins=50, range=(-1000,1000), alpha=0.4,label='vtheta')
        plt.hist(np.array(vphi), bins=50, range=(-1000,1000),alpha=0.4,label='vphi')
        #plt.hist(np.array(vtheta), bins=50, alpha=0.4,label='vtheta')
        #plt.hist(np.array(vphi), bins=50, alpha=0.4,label='vphi')
        plt.legend()

        # rotation velocity
        vrot_hist_fig = plt.figure()
        plt.hist(np.array(np.sqrt(v2 - vr**2)), bins=50, alpha=0.4, label='$v_{rot}1$')
        plt.hist(np.array(v_rot_arr), bins=50, range=(0,1000), alpha=0.4, label='$v_{rot}2$')
        plt.legend()

        plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', dest='filename', required='true', type=str, help='File to read star data from')
    parser.add_argument('--plots', dest='plots', action='store_true', help='Flag to enable plotting')
    args = parser.parse_args()

    if not args.plots:
        print("Run with the argument '--plots' to get plots (what else!?)")
    
    stars(args.filename, args.plots)
