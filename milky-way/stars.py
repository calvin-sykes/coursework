#!/usr/bin/python3
# Import statements
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import Galactocentric
from astropy.coordinates import ICRS
from astropy.coordinates import SphericalRepresentation
from astropy.coordinates import CartesianDifferential

# This gets used to remove a couple of massive outliers from one of the angular velocity
# components. Leaving them in skews the results and makes some of the plots look stupid.
def filter_outliers(vals):
    """Filter an array by removing values with magnitude greater than a limit.
    Currently this limit is fixed to be 1000, should probably change this.
    Parameters: the array to filter."""
    return vals[np.abs(vals) < 1000 * vals[0].unit]

def mean_sq(vals):
    """Calculates the mean square value of a quantity, removing outliers if necessary.
    Parameters: the array to operate on."""
    filt = filter_outliers(vals)
    return np.mean(filt ** 2)

def vel_anisotropy(vr, vtheta, vphi):
    """Calculate the velocity anistropy parameter, beta
    Parameters: the radial, latitudinal, and longitudinal velocity components."""
    return 1 - ((mean_sq(vtheta) + mean_sq(vphi)) / (2 * mean_sq(vr)))

def stars(in_file, plots_flag):
    # Create structured datatype correspond to given format
    # and load data file
    in_type = np.dtype([('ra'  , 'f'),
                      ('dec'  , 'f'),
                      ('d'    , 'f'),
                      ('vlos' , 'f'),
                      ('pmra' , 'f'),
                      ('pmdec', 'f')])
    data = np.loadtxt(in_file,dtype=in_type,skiprows=1)

    # astropy.SkyCoords doesn't support automatically converting the velocity components
    # so we need to transform the data manually
    icrs = ICRS(ra=data['ra'] * u.degree,
                dec=data['dec'] * u.degree,
                distance=data['d'] * u.kpc,
                pm_ra_cosdec=(data['pmra'] * u.mas / u.yr
                              * np.cos(data['dec'] * u.degree)),
                pm_dec=data['pmdec'] * u.mas / u.yr,
                radial_velocity=data['vlos'] * u.km / u.s)
    
    # Cartesian galactocentric coordinates
    v_sun = CartesianDifferential(11.1, 12.248 + 238, 7.25, unit=u.km / u.s)
    gal_cen = icrs.transform_to(Galactocentric(galcen_v_sun=v_sun,
                                               galcen_distance=8.0 * u.kpc,
                                               z_sun=0 * u.kpc))
    
    # Output the 6D Cartesian coordinates
    out_file = '6d.txt'
    out_data = np.column_stack((gal_cen.x,
                                gal_cen.y,
                                gal_cen.z,
                                gal_cen.v_x,
                                gal_cen.v_y,
                                gal_cen.v_z))
    cols = ('x[kpc]', 'y[kpc]', 'z[kpc]', 'v_x[km/s]', 'v_y[km/s]', 'v_z[km/s]')
    np.savetxt(out_file, out_data, fmt='%9.4f', header=' '.join(cols))
    print('6D coordinates written to {}'.format(out_file))
    
    # r = distance
    # theta = latitude
    # phi = azimuth/longitude
    # NB This is opposite to how spherical coordinates are normally defined!
    
    # Spherical galactocentric coordinates
    gal_cen_sph = gal_cen.copy()
    gal_cen_sph.representation = 'spherical'
    
    # Calculate spherical velocity components and make units sensible
    vr = (gal_cen_sph.d_distance
          .to(u.km/u.s))
    vtheta = ((gal_cen_sph.d_lat * gal_cen_sph.distance)
              .to(u.rad * u.km/u.s) / u.rad)
    vphi = ((gal_cen_sph.d_lon * gal_cen_sph.distance * np.cos(gal_cen_sph.lat))
            .to(u.rad * u.km/u.s) / u.rad)
    
    # Calculate standard deviations (sigma) for three velocity components
    vr_dev = np.std(vr)
    vtheta_dev = np.std(vtheta)
    vphi_dev = np.std(vphi)
    
    # Calculate the velocity anisotropy parameter beta
    v_anis = vel_anisotropy(vr, vtheta, vphi)

    # Calculate the rotation velocity
    # Assumed that this is the expectation value of vphi
    v_rot = np.mean(vphi)
    
    # Report the results
    print('Velocity dispersion and anisotropy results:') 
    print("""    sigma_r = {}
    sigma_theta = {}
    sigma_phi = {}
    v_rot = {}
    beta = {}""".format(vr_dev, vtheta_dev, vphi_dev, v_rot, v_anis))

    # Spit out some plots if the command line flag was given
    if plots_flag:
        #from matplotlib import rc
        #rc('text', usetex=True)
        # plot positions
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title('XY positions scatterplot')
        ax.set_aspect('equal')
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        plt.scatter(gal_cen.x, gal_cen.y, s=2) 

        # radius histogram
        hist_fig = plt.figure()
        ax = plt.gca()
        ax.set_title('Histogram of radial distances')
        ax.set_xlabel('Radial distance [kpc]')
        ax.set_ylabel('Count')
        plt.hist(gal_cen_sph.distance, bins='auto')

        # xyz histogram
        xyz_hist_fig = plt.figure()
        ax = plt.gca()
        ax.set_title('Histogram of Cartesian position components')
        ax.set_xlabel('Position [kpc]')
        ax.set_ylabel('Count')
        plt.hist(np.array(gal_cen.x), bins=50, alpha=0.4, label='x')
        plt.hist(np.array(gal_cen.y), bins=50, alpha=0.4, label='y')
        plt.hist(np.array(gal_cen.z), bins=50, alpha=0.4, label='z')
        plt.legend()

        # velocities
        vel_fig = plt.figure()
        ax = plt.gca()
        ax.set_title('Histogram of spherical velocity components')
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Count')
        plt.hist(np.array(vr), bins=50, range=(-1000,1000),alpha=0.4,label='vr')
        plt.hist(np.array(vtheta), bins=50, range=(-1000,1000), alpha=0.4,label='vtheta')
        plt.hist(np.array(vphi), bins=50, range=(-1000,1000),alpha=0.4,label='vphi')
        #plt.hist(np.array(vtheta), bins=50, alpha=0.4,label='vtheta')
        #plt.hist(np.array(vphi), bins=50, alpha=0.4,label='vphi')
        plt.legend()

        cart_vel_fig = plt.figure()
        ax = plt.gca()
        ax.set_title('Histogram of Cartesian velocity components')
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Count')
        plt.hist(np.array(gal_cen.v_x), bins=50, range=(-1000,1000),alpha=0.4,label='vx')
        plt.hist(np.array(gal_cen.v_y), bins=50, range=(-1000,1000), alpha=0.4,label='vy')
        plt.hist(np.array(gal_cen.v_z), bins=50, range=(-1000,1000),alpha=0.4,label='vz')
        #plt.hist(np.array(vtheta), bins=50, alpha=0.4,label='vtheta')
        #plt.hist(np.array(vphi), bins=50, alpha=0.4,label='vphi')
        plt.legend()

        print("Plots have been opened. Close them all to terminate the program.")
        plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', dest='filename', required='true', type=str,
                        help='File to read star data from')
    parser.add_argument('--plots', dest='plots', action='store_true',
                        help='Flag to enable plotting')
    args = parser.parse_args()

    if not args.plots:
        print("Run with the argument '--plots' to get plots (what else!?)")
    
    stars(args.filename, args.plots)
