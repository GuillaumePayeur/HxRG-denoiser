import numpy as np
from astropy.io import fits
import glob
from scipy.ndimage.filters import median_filter
import os

def sigma(im):
    return (np.nanpercentile(im,84.13448) - np.nanpercentile(im,15.86552))/2

def create_ramp(ramp_folder,scene):
    """
    We take a folder containing a set of unilluminated dark frames and we add
    a science 'scene' that is considered noiseless. This scene is added as a
    Poisson accumulation and is the basis for the ML traning.
    """

    scene = fits.getdata(scene)

    # loop through ramp directories
    files = glob.glob(ramp_folder+'/*.fits')
    files.sort()

    # output directory
    outdir = 'pp_'+ramp_folder
    # we crate a
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # find ribbon pixels
    x, y = np.indices([ 4096,128])
    x = x/4095.

    # left-right reference pixels
    ref_pix_index = [0,1,2,3,4092,4093,4094,4095]

    # first readout is used zp. It sets the pedestal for ramps. This is
    # 'transparent' in the slope measurement as it is a measurement of the
    # intercept.
    zp = np.array(fits.getdata(files[0]),dtype = float)

    # for linear fit
    sy = np.zeros([4096,4096])
    sxy = np.zeros([4096,4096])
    n = 0

    # swap left/right 1 quadrant out of 2 as the H4RG quadrants are left in a
    # 'butterfly wing' fasion. This simplifies things later on
    for i in range(32):
        if (i % 2) == 0:
            scene[:, i * 128:(i + 1) * 128] = scene[:, i * 128:(i + 1) * 128][:, ::-1]

    # scale the scene such that it has a peak (actually 99th percentile) flux at the
    # level of 30 electrons. This is in the sweet-spot for noise filtering as it has
    # a Poisson noise of 5-6 e-.
    scene = 30*scene/np.nanpercentile(scene,99)
    scene[~np.isfinite(scene)] = 0
    scene[scene<0] = 0
    scene[scene>60] = 0

    # we save a fits file containing the scene
    fits.writeto('scene_'+ramp_folder+'.fits', scene , overwrite = True)

    # we determine the Poisson accumulation rate per file.
    per_frame_poisson = scene/len(files)

    # we have a running Poisson scene
    running_scene = np.zeros_like(scene)

    # loop through files within directory
    for file in files[1:]:
        print(ramp_folder,'->',file)
        outname = outdir+'/pp_'+file.split('/')[1]

        # reading image
        im,hdr = fits.getdata(file, header=True)
        im = im - zp

        print('\tsigma first readout : {:.2f} ADUs'.format(sigma(im)))
        # we remove the very low-frequency (slope along amplifier) noise
        # with the reference pixels in each of the 32 amplifiers
        for i in range(32):
            bottom = np.nanmedian(im[0:4,i*128:(i+1)*128])
            top = np.nanmedian(im[-4:,i*128:(i+1)*128])
            im[:,i*128:(i+1)*128] -= (bottom + (x*(top-bottom)) )

            # flipping left/right even orders
            if (i % 2) == 0:
                im[:, i * 128:(i + 1) * 128] = im[:,i*128:(i+1)*128][:,::-1]

        # just to see how we are cutting down the noise at each step
        print('\tsigma after top-bottom ref : {:.2f} ADUs'.format(sigma(im)))

        # we remove the common mode noise in the reference pixels
        tmp = median_filter(np.nanmedian(im[:,ref_pix_index],axis=1),12)
        im -= np.repeat(tmp,4096).reshape(4096,4096)
        print('\tsigma after all refs : {:.2f} ADUs'.format(sigma(im)))
        print('we write : {}\n'.format(outname))

        # we add the running Poisson contribution
        running_scene += np.random.poisson(per_frame_poisson)
        im = im + running_scene

        fits.writeto(outname, im, hdr,overwrite = True)
