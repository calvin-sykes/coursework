import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# number of pixels in CCD image
NX = 100
NY = 100

def ccd(infile):
    """Produces a mock CCD image from a provided data file
    giving locations and count rates of stars. 

    Accepts a file name as argument containing data to plot.
    """

    # Create structured datatype correspond to given format
    # and load data file
    record = np.dtype([('name', 'S4'),
                      ('x', np.float64),
                      ('y', np.float64),
                      ('counts', np.float64),
                      ('sigma', np.int8)])
    data = np.loadtxt(infile,dtype=record)

    # Samples corresponding to stars
    samples = np.empty((0,2))
    for star in data:
        samples = np.concatenate((samples, spread(star)))
    xsam, ysam = np.hsplit(samples, 2)

    # Noise field
    noise = np.random.poisson(10.0, (NX, NY))

    # Bin star samples and add noise field
    hist, xedges, yedges = np.histogram2d(ysam.flatten(), xsam.flatten(), (NY, NX))
    hist += noise

    # Make plot
    from matplotlib import rc
    rc('text', usetex=True)
    
    fig = plt.figure()
    ax  = plt.gca()
    ax.set_xlabel(r'$\Delta X$', fontsize=14)
    ax.set_ylabel(r'$\Delta Y$', fontsize=14)
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_aspect('equal')

    nrm = mp.colors.LogNorm(vmin=0.1, vmax=10000)
    img = plt.imshow(hist, cmap=plt.cm.viridis, norm=nrm, origin='lower')

    fmt = mp.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf))
    txs = mp.ticker.LogLocator(base=10.0,subs=(1.0,3.16,))

    cbar = plt.colorbar(img, ax=ax, format=fmt, ticks=txs)
    cbar.set_label('$\log_{10}\mathrm{counts}$', fontsize=14)

    # Calculate and report total counts and maximum count rate
    totalcounts = np.sum(hist)
    maxcount = np.max(hist)
    print('Total counts: {:e}'.format(totalcounts))
    print('Maximum count: {:e}'.format(maxcount))

    # Save the figure
    outfile = 'challenge.png'
    plt.savefig(outfile)
    print('Wrote output to {}'.format(outfile))

def spread(star):
    """Return a list of normally-distributed xy count positions for a star.
    
    The input star should be provided as a tuple with fields as follows:
    - name
    - x centre
    - y centre
    - no. of counts
    - standard deviation in position (assumed equal in x and y) 
    """
    name, x, y, counts, sigma = star
    return np.random.multivariate_normal((x, y),
                                         ((sigma,0),(0,sigma)),
                                         int(counts))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str)
    args = parser.parse_args()
    
    np.random.seed(42)
    
    ccd(args.filename)
