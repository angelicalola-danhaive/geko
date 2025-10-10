"""
    geko - utils
    This module contains utility functions for the Geko package.
    -- Any function mutliple times that is repeated in the code?
	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['oversample', 'resample', 'scale_distribution', 'find_best_sample', 'compute_gal_props',
           'load_psf', 'compute_inclination', 'compute_axis_ratio', 'add_v_re', 'sersic_profile', 
           'compute_adaptive_sersic_profile', 'flux_to_Ie', 'Ie_to_flux', 'rotate_coords']
           
# geko imports

# import run_pysersic as py

#other imports
import jax
from jax import image
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import gamma


# from skimage.filters import threshold_otsu
# from skimage.morphology import closing
# from skimage.measure import label, regionprops
# from skimage.segmentation import clear_border
# from skimage.morphology import dilation, disk, ellipse, binary_closing
# from skimage.color import label2rgb

from astropy.modeling.models import Sersic2D
from scipy import stats
from scipy.ndimage import zoom
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources, SegmentationImage
from photutils.background import Background2D, MedianBackground
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse, EllipseGeometry, Isophote
from photutils.isophote import build_ellipse_model

from skimage import color, data, restoration

from matplotlib import pyplot as plt

from astropy.io import fits

import numpy as np

import xarray as xr

from scipy import stats as st

from . import  models

import os

from pathlib import Path

from astropy.wcs import WCS



def oversample(image_low_res, factor_y, factor_x, method='nearest'):
    '''
		Oversample an image by a given factor in the x and y direction

		Parameters
		----------
        image_low_res: the 2D array to be oversampled
        factor_y,_x factor_z: oversampling factor for the y-axis, x-axis

		Returns
		----------
        image_high_res: the oversampled array with shape (image_low_res.shape[0]*factor_y, image_low_res.shape[1]*factor_x)

    '''

    image_high_res = image.resize(image_low_res, (int(
        image_low_res.shape[0]*factor_y), int(image_low_res.shape[1]*factor_x)), method=method)
    
    image_high_res /= factor_y*factor_x 

    return image_high_res


def resample(image_high_res, factor_y, factor_x):
    '''
		Downsample an image by a given factor in the x and y direction.
        This is done by summing pixels to reduce the resolution.

		Parameters
		----------
        image_high_res: the 2D array to be downsampled
        factor_y,_x factor_z: downsample factor for the y-axis, x-axis --> must be able to divide the shape of that axis

		Returns
		----------
        image_low_res: the downsampled array with shape (image_low_res.shape[0]/factor_y, image_low_res.shape[1]/factor_x)

    '''
    blocks = image_high_res.reshape(
        (int(image_high_res.shape[0]/(factor_y)), factor_y, int(image_high_res.shape[1]/factor_x), factor_x))
    image_low_res = np.sum(blocks, axis=(1, 3))
    return image_low_res

def downsample_error(error_array):
    """
    Downsample a 2D error array by a factor of 2 with proper error propagation.

    Uses RMS (root-mean-square) averaging of 2x2 blocks to properly propagate
    uncertainties when downsampling error maps.

    Parameters
    ----------
    error_array : numpy.ndarray
        2D array of error/uncertainty values to downsample

    Returns
    -------
    numpy.ndarray
        Downsampled error array with shape (h//2, w//2)

    Notes
    -----
    If dimensions are odd, the array is cropped to even dimensions before downsampling.
    Error propagation: sigma_downsampled = sqrt(mean(sigma_i^2)) for 2x2 blocks.
    """
    # Ensure dimensions are even
    h, w = error_array.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    error_array = error_array[:h2, :w2]  # crop if necessary

    # Reshape and compute RMS of 2x2 blocks (error propagation)
    reshaped = error_array.reshape(h2//2, 2, w2//2, 2)
    downsampled = np.sqrt(np.mean(reshaped**2, axis=(1, 3)))
    return downsampled


def scale_distribution(distribution, mu, sigma, max, min):
    '''
        Rescale a base distribution to its target distribution. 
        This is used for parameters that are rescaled to their target parameter space by simply applying 
        a function, and not using numpyro.deterministic.
        Based on the non-None parameters given to this function, the target distribution is inferred.

		Parameters
		----------
        distribution: base distribution (2D array containing posterior/prior data)
        mu: mean of distribution, None if mean not needed to define the target distribution
        sigma: std of the distribution, None if std not needed to define the target distribution
        max: upper bound of the distribution,  None if max not needed to define the target distribution
        min: lower bound of the distribution,  None if min not needed to define the target distribution

		Returns
		----------
        rescaled_distribution: the new target distribution

    '''

    if mu == None:
        #if mu is None, then the distribution is uniform
        rescaled_distribution = scale_uniform(distribution, max, min)
    elif max == None:
        #if mu is not None but max is None, then the distribution is Gaussian
        rescaled_distribution = scale_gaussian(distribution, mu, sigma)
    else:
        #if mu and max are not None, then the distribution is truncated Gaussian
        rescaled_distribution = scale_truncated_gaussian(
            distribution, mu, sigma, max, min)
        
    return rescaled_distribution


def scale_uniform(distribution, max, min):
    '''
        Rescale a base distribution to a uniform distribution U(min,max)

		Parameters
		----------
        distribution: base distribution (2D array containing posterior/prior data)
        max: upper bound of the distribution
        min: lower bound of the distribution

		Returns
		----------
        rescaled_distribution: data rescaled to a uniform distribution U(min,max)

    '''
    rescaled_distribution = distribution * (max - min) + min
    return rescaled_distribution


def scale_gaussian(distribution, mu, sigma):
    '''
        Rescale a base distribution to a normal distribution N(mu,sigma)

		Parameters
		----------
        distribution: base distribution (2D array containing posterior/prior data)
        mu: mean of distribution
        sigma: std of the distribution

		Returns
		----------
        rescaled_distribution: data rescaled to a normal distribution  N(mu,sigma)

    '''
    rescaled_distribution = norm.ppf(distribution) * sigma + mu
    return rescaled_distribution

def scale_truncated_gaussian(distribution, mu, sigma, max, min):
    '''
        Rescale a base distribution to a truncated normal distribution N(mu,sigma), min<x<max

		Parameters
		----------
        distribution: base distribution (2D array containing posterior/prior data)
        mu: mean of distribution
        sigma: std of the distribution
        max: upper bound of the distribution (None if no upper bound)
        min: lower bound of the distribution (None if no lower bound)
z
		Returns
		----------
        rescaled_distribution: data rescaled to a truncated normal distribution  N(mu,sigma), min<x<max

    '''
    rescaled_distribution = norm.ppf(norm.cdf(min) + distribution*(norm.cdf(max)-norm.cdf(min)))*sigma + mu
    return rescaled_distribution


def find_best_sample(inference_data, variable, mu, sigma, max, min, MLS):
    '''
        Find the best sample (either median or highest likelihood) from input posterior,
        for each parameter in the parameter list.
        Before finding the best sample, each posterior is rescaled to its target distribution.

		Parameters
		----------
        inference_data: arviz inference data type -- results from the numpyro fitting
        mu: mean of distribution
        sigma: std of the distribution
        max: upper bound of the distribution (None if no upper bound)
        min: lower bound of the distribution (None if no lower bound)

		Returns
		----------
        rescaled_distribution: data rescaled to a truncated normal distribution  N(mu,sigma), min<x<max
    '''

    #even if only one variable is passed, make sure it is still a list (with only one entry)
    variable = variable
    best_sample = []

    for i, var in enumerate(variable):

        #inference_data is modified even if it is not returned
        inference_data.posterior[var].data = scale_distribution(inference_data.posterior[var].data, mu[i], sigma[i], max[i], min[i])
        inference_data.prior[var].data = scale_distribution(inference_data.prior[var].data, mu[i], sigma[i], max[i], min[i])
        
        if MLS == None:
            # print('Taking posterior median sample')
            best_sample.append(jnp.array(inference_data.posterior[var].median(dim=["chain", "draw"])))
        else:
            # print('Taking the maximum likelihood sample')
            best_sample.append(jnp.array(inference_data.posterior[var].isel(chain=MLS[0], draw=MLS[1])))

    
    return best_sample

def find_central_object(image,threshold_sigma, npixels = 10):
    """
       Make a mask that selects only the central object
    """
    size = 20
    if image.shape[0]<62:
        size = 10
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (size,size) , filter_size=(5, 5),bkg_estimator = bkg_estimator, exclude_percentile=20)
    threshold = threshold_sigma * bkg.background_rms

    segment_map = detect_sources(image, threshold, npixels)
    if segment_map == None:
        print('No sources found')
        return None, None
    cat = SourceCatalog(image, segment_map)

    #select only the central source
    idx_source = jnp.argmin((cat.xcentroid-image.shape[1]//2)**2 + (cat.ycentroid-image.shape[0]//2)**2)

    label_source = idx_source + 1
    segment_map.keep_labels([label_source])
    segment_map.reassign_label(label_source, 1)

    return segment_map, cat

def make_mask(image, n = 1, threshold_sigma = 3):
    """
       Make n masks of the image.
    """

    segment_map, cat = find_central_object(image, threshold_sigma)

    mask_1comp = segment_map.copy()
    mask_1comp = dilation(mask_1comp.data, disk(6))
    plt.imshow(mask_1comp, origin='lower')
    plt.scatter(cat.xcentroid, cat.ycentroid, color='red')
    plt.title('Flux Mask')
    plt.show()
    plt.close()
    # plt.imshow(image*segment_map.data, origin='lower')
    # plt.title('Mask of the central source')
    # plt.show()

    #separate two sources if n == 2
    if n == 2:
        segm_deblend = deblend_sources(image, segment_map, npixels=10, nlevels=32, contrast=0.001)
        plt.imshow(segm_deblend.data, origin='lower')
        plt.title('Make mask - segm debelnd')
        plt.show()
        plt.close()

        #if more than 2 labels, keep the central one and merge the others
        if segm_deblend.nlabels > 2:
            idx_source = jnp.argmin((cat.xcentroid-image.shape[0]//2)**2 + (cat.ycentroid-image.shape[0]//2)**2)
            label_source = 3 #idx_source + 1
            new_label = segm_deblend.labels[segm_deblend.labels != label_source][0]
            for label in segm_deblend.labels:
                if label != label_source:
                    segm_deblend.reassign_label(label, 2)
            segm_deblend.reassign_label(label_source, 1)

        mask_2comp = segm_deblend.copy()
        mask_2comp = dilation(mask_2comp.data, disk(3))
        cat = SourceCatalog(image, segm_deblend)
        plt.imshow(segm_deblend.data, origin='lower')
        plt.scatter(cat.xcentroid, cat.ycentroid, color='red')
        plt.show()
        plt.close()
    else: 
        segm_deblend = None
        mask_2comp = None
    
    return segment_map, segm_deblend, mask_1comp, mask_2comp


def compute_gal_props(image, n=1, threshold_sigma=3):
	'''
		compute PA of galaxy in the image using skimage measure regionprops

	'''

	seg_1comp, seg_2comp, mask_1comp, mask_2comp = make_mask(image, n, threshold_sigma) #1
	if n == 1:   
		sources = SourceCatalog(image, seg_1comp)
	if n == 2:
		sources = SourceCatalog(image, seg_2comp)
	PA = []
	inc = []
	r_eff = []
	center = []
	ratio = []
	for i in range(len(sources)):
		angle = sources.orientation[i].value #orientation is in degrees [-180,180]
		if angle<0:
			angle += 180 #convert to [0,180]
		PA.append(angle) 
		ratio.append(sources.elongation[i].value)
		inc.append(jnp.arccos(1/sources.elongation[i].value)* (180/jnp.pi)) #convert to degrees
		r_eff.append(sources.equivalent_radius[i].value)
		print('centroid. ', sources.centroid[i])
		center.append(sources.centroid_quad[i])
    
	# print(seg_1comp)
	return seg_1comp, seg_2comp, mask_1comp, mask_2comp, jnp.array(PA), jnp.array(inc), jnp.array(r_eff), jnp.array(center), jnp.array(ratio)


def save_fits_image(image, masked_indices, inference_data, filename):
    """
        Save a fits image to a file
    """
    #create list 
    hdul = fits.HDUList()
    #put the best fit flux map in the primary HDU
    primary_hdu = fits.PrimaryHDU(image)
    primary_hdu.name = 'FLUX_MAP'
    hdul.append(primary_hdu)
    #put the mask in the second HDU
    mask_hdu = fits.ImageHDU(masked_indices)
    mask_hdu.name = 'MASK'
    hdul.append(mask_hdu)

    #put the posteriors in the third HDU
    posteriors = jnp.array(inference_data.posterior['fluxes'].data)
    #put the two chains back to back so its only a 2D array with 1 chain per pixel
    posteriors = posteriors.reshape((posteriors.shape[0]*posteriors.shape[1],posteriors.shape[2]))  
    posteriors_hdu = fits.ImageHDU(posteriors)
    posteriors_hdu.name = 'POSTERIORS'
    hdul.append(posteriors_hdu)

    hdul.writeto(filename, overwrite=True)
    
def downsample_psf_centered(psf_full, size):
    """
    Crop and downsample a PSF around its center.

    Extracts a centered region from the full PSF and downsamples it by a factor of 2.

    Parameters
    ----------
    psf_full : numpy.ndarray
        Full PSF array (typically high-resolution)
    size : int
        Half-width of the region to extract (output will be (2*size+1, 2*size+1) before downsampling)

    Returns
    -------
    numpy.ndarray
        Downsampled PSF with shape approximately (size, size)

    Notes
    -----
    The PSF is cropped symmetrically around its center before downsampling.
    """
    psf_crop = np.array(psf_full[psf_full.shape[0]//2 - size:psf_full.shape[0]//2 + size + 1, \
                                            psf_full.shape[1]//2 - size:psf_full.shape[1]//2 + size + 1])


    #downsample the psf by a factor of 2
    psf_downsampled = np.zeros(((psf_crop.shape[0]-1)//2 , (psf_crop.shape[0]-1)//2))

    print(psf_downsampled.shape)
    #each pixel in the downsized image is equal to the sum of its original pizel plus half of each of the 4 neighboring pixel
    # for i in np.linspace(1, (psf_crop.shape[0]-1)//2, (psf_crop.shape[0]-1)//2):
    #     for j in np.linspace(1, (psf_crop.shape[0]-1)//2, (psf_crop.shape[0]-1)//2):
    i= jnp.linspace(0, (psf_crop.shape[0]-1)//2 -1, (psf_crop.shape[0]-1)//2).astype(int)
    j= jnp.linspace(0, (psf_crop.shape[0]-1)//2 -1, (psf_crop.shape[0]-1)//2).astype(int)   
    i,j = jnp.meshgrid(i,j)
    psf_downsampled[i,j] = psf_crop[2*i + 1,2*j + 1] + 0.5*(psf_crop[2*i + 2,2*j + 1] + psf_crop[2*i,2*j + 1] + psf_crop[2*i + 1,2*j + 2] + psf_crop[2*i + 1,2*j])

    return psf_downsampled

def load_psf( filter, y_factor, size = 9, psf_folder = 'mpsf_gds/'):
    """
    Load and process a PSF file for a given filter.

    Loads a PSF FITS file, crops it to the desired size, and optionally downsamples
    it based on the pixel scale factor.

    Parameters
    ----------
    filter : str
        Filter name (e.g., 'F444W')
    y_factor : int
        Spatial oversampling factor. If 1, PSF is downsampled to 0.063" resolution
    size : int, optional
        Size of output PSF array (default: 9)
    psf_folder : str, optional
        Path to folder containing PSF files (default: 'mpsf_gds/')

    Returns
    -------
    numpy.ndarray
        Normalized PSF array of shape (size, size)

    Notes
    -----
    The PSF is always renormalized to sum to 1 after processing.
    Filename format expected: 'mpsf_{filter}.fits' in psf_folder.
    """
    psf_fits = fits.open(psf_folder + 'mpsf_' + str(filter).lower() + '.fits')
    psf_full = np.array(psf_fits[0].data)

    print(psf_full.shape)
    if y_factor == 1: #then you have to resample it down to 0.063 resolution

        psf = downsample_psf_centered(psf_full, size)
    else:
        #make cutout of the psf to size = size
        psf = psf_full[psf_full.shape[0]//2 - size//2:psf_full.shape[0]//2 + size//2 + 1, \
                    psf_full.shape[1]//2 - size//2:psf_full.shape[1]//2 + size//2 + 1]
    #renormalize the psf
    psf = psf/psf.sum()

    return psf

def oversample_PSF(psf, factor):
    """
        Oversample the PSF by a factor of 2
    """
    psf_high_res = image.resize(psf, (int(psf.shape[0]*factor), int(psf.shape[1]*factor)), method='nearest')
    psf_high_uneven = np.zeros((psf_high_res.shape[0]- 1, psf_high_res.shape[1]-1))
    for i in range(psf_high_res.shape[0]-1):
         for j in range(psf_high_res.shape[1]-1):
             psf_high_uneven[i,j] = 0.25*(psf_high_res[i,j] + psf_high_res[i+1,j] + psf_high_res[i,j+1] + psf_high_res[i+1,j+1])
    psf_high_uneven /= psf_high_uneven.sum()
    return psf_high_uneven

def compute_inclination(axis_ratio = None, ellip = None, q0 = 0.0):
    """
        Compute the inclination of a galaxy given its axis ratio or ellipticity
    """
    if axis_ratio == None:
        axis_ratio = 1 - ellip
    if axis_ratio<q0:
        axis_ratio = q0
    inc = jnp.arccos(jnp.sqrt( (axis_ratio**2 - q0**2)/(1 - q0**2) ))*(180/jnp.pi)
    # inc = jnp.arccos(axis_ratio)*(180/jnp.pi)
    return inc

def compute_axis_ratio(inc, q0 = 0.0):
    """
        Compute the axis ratio of a galaxy given its inclination
    """
    axis_ratio = jnp.sqrt(jnp.cos(jnp.radians(inc))**2*(1-q0**2) + q0**2)
    return axis_ratio

def fit_grism_parameters(obs_map, r_eff, inclination, obs_error, sigma_rms = 2.0):
    obs_map = np.array(obs_map)
    obs_error = np.array(obs_error)
    im_conv, segment_map = py.make_mask(im = obs_map, sigma_rms = sigma_rms)
    #create a mask of the grism data based on the segment map
    main_label = segment_map.data[int(0.5*obs_map.shape[0]), int(0.5*obs_map.shape[1])]
    segment_map = np.where(segment_map.data == main_label, 1, 0)
    # plt.imshow(segment_map, origin='lower')
    # plt.show()
    segment_map_closed = segment_map #binary_closing(segment_map, ellipse(10,5))
    # plt.imshow(segment_map_closed, origin='lower')
    # plt.show()
    masked_map = np.where(segment_map_closed == 1.0, obs_map, 0.0)
    source = SourceCatalog(masked_map, SegmentationImage(segment_map_closed.astype(int)), convolved_data=im_conv, error=obs_error)
    PA_mask = source.orientation[0].value #orientation is in degrees [-180,180]
    i_mask = np.arccos(1/source.elongation[0].value)* (180/np.pi)
    r_eff_mask = source.equivalent_radius[0].value/2 #assume smaller rad for fit
    if r_eff ==None:
         r_eff = r_eff_mask
    if inclination == None:
         inclination = i_mask
    x0 = source.centroid[0][0]
    y0 = source.centroid_quad[0][1]
    geom = EllipseGeometry(x0 = x0, y0 = y0, sma = r_eff, eps = 1 - np.cos(inclination*(np.pi/180)), pa = PA_mask*(np.pi/180))
    ellipse_shape = Ellipse(np.abs(obs_map), geometry = geom, threshold = 0.001)
    # print(geom.sma, np.abs(geom.sma))
    aper = EllipticalAperture((geom.x0, geom.y0), np.abs(geom.sma), np.abs(geom.sma * (1 - geom.eps)),geom.pa)
    # plt.imshow(obs_map, origin='lower')
    # aper.plot(color='red')
    # plt.show()
    isolist = ellipse_shape.fit_image()
    if not isolist:
        print('No isophotes found, using source catalog measurements instead.')
        source_cat = SourceCatalog(obs_map, SegmentationImage(segment_map_closed.astype(int)), convolved_data=im_conv, error=obs_error)
        angle = source_cat.orientation[0].value #orientation is in degrees [-180,180]
        if angle<0:
                angle += 180 #convert to [0,180]
        PA_grism = angle
        inc_grism = np.arccos(1/source_cat.elongation[0].value)* (180/np.pi)
        # print(source_cat.equivalent_radius[0].value, jnp.cos(PA_grism*(jnp.pi/180)))
        r_full_grism = source_cat.equivalent_radius[0].value*np.abs(np.cos(PA_grism*(np.pi/180)))
        x0_vel = source_cat.centroid[0][0]
        y0_vel = source_cat.centroid_quad[0][1]
        #find half-light radius by measuring light inside each isophote
        # Calculate the total flux
        #find the half-light radius
        image = obs_map
        center = (15,15)
        image = np.where(image < 0.01*image.max(), 0, image)
        total_flux = image.sum()
            
        # Create a grid of x and y coordinates
        y, x = np.indices(image.shape)
            
        # Calculate the radius for each pixel from the center
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            
        #put to zero everything below a certain flux threshold
        # plt.imshow(image, origin = 'lower')
        # plt.title('Image to comp half-light radius')
        # plt.show()
        # Flatten the image and radius arrays
        r_flat = r.flatten()
        image_flat = image.flatten()
            
        # Sort the pixels by radius
        sorted_indices = np.argsort(r_flat)
        sorted_r = r_flat[sorted_indices]
        sorted_flux = image_flat[sorted_indices]
            
        # Compute the cumulative sum of the flux
        cumulative_flux = np.cumsum(sorted_flux)
            
        # Find the radius where the cumulative flux reaches half of the total flux
        half_light_flux = total_flux / 2
        half_light_radius_index = np.where(cumulative_flux >= half_light_flux)[0][0]
        half_light_radius = sorted_r[half_light_radius_index]
        print('Turn-over rad: ',  0.4*half_light_radius/1.676)
    else:
        # print(isolist.to_table())  
        model_grism = build_ellipse_model(obs_map.shape, isolist)

        #compute the half-light radius
        image = np.where(obs_map < 0.03*obs_map.max(), 0, obs_map)
        total_flux = image.sum()
        index_half_light = np.nanargmin(np.abs(isolist.tflux_e - 0.5*total_flux))
        half_light_radius = isolist.sma[index_half_light]
        print('Turn-over rad isophote: ',  0.4*half_light_radius/1.676)

        rad_22 = 2.2*half_light_radius/1.676
		#choose the isophote with the sma closest to morphological radius
        best_index = np.argmin(np.abs(isolist.sma - rad_22)) #the 1.5 is a bit arbitrary ...
        PA_grism = isolist.pa[best_index]*(180/np.pi) #put in degrees
        inc_grism =np.arccos(1-isolist.eps[best_index])*(180/np.pi) #put in degrees
        r_full_grism = np.maximum(rad_22*np.abs(np.cos(PA_grism*(np.pi/180))), rad_22*np.abs(np.cos((90-PA_grism)*(np.pi/180)))) #project onto velocity/x axis
        print('r_full: ', r_full_grism)
        #get the centroids
        x0_vel = isolist.x0[best_index]
        y0_vel = isolist.y0[best_index]
        residual = obs_map - model_grism

        
        # plt.plot(isolist.sma, isolist.tflux_e, '.')   #(np.cos(1-isolist.eps)*(180/np.pi))
        # plt.show()
        # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
        # fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
        # ax1.imshow(obs_map, origin='lower')
        # ax1.set_title('Data')

        sma = isolist.sma[best_index]
        iso = isolist.get_closest(sma)
        # x, y, = iso.sampled_coordinates()
        # ax1.plot(x, y, color='white')

        # ax2.imshow(model_grism, origin='lower')
        # ax2.set_title('Ellipse Model')

        # ax3.imshow(residual, origin='lower')
        # ax3.set_title('Residual')
        # plt.show()

        print('Isophote fitting results - ')
        print('PA: ', PA_grism, 'Inclination: ', inc_grism, 'r_full: ', r_full_grism, 'x0: ', x0_vel, 'y0: ', y0_vel)
    
    return PA_grism, inc_grism, r_full_grism, x0_vel, y0_vel, half_light_radius


def add_v_re(inf_data, kin_model, grism_object, num_samples, re_manual = None):
    # Get number of chains from the inference data
    num_chains = inf_data.posterior['PA'].shape[0]
    num_samples_prior = inf_data.prior['PA'].shape[1]

    inf_data.posterior['v_re'] = xr.DataArray(np.zeros((num_chains, num_samples)), dims = ('chain', 'draw'))
    inf_data.prior['v_re'] = xr.DataArray(np.zeros((1, num_samples_prior)), dims = ('chain', 'draw'))
    #make the prior array based on the shape of another prior from inf_data
    for i in range(num_chains):
        for sample in range(num_samples):
            x = np.linspace(0 - kin_model.x0_vel_mean, kin_model.im_shape[1]-1 - kin_model.x0_vel_mean, kin_model.im_shape[1]*kin_model.factor)
            y = np.linspace(0 - kin_model.y0_vel_mean, kin_model.im_shape[0]-1 - kin_model.y0_vel_mean, kin_model.im_shape[0]*kin_model.factor)
            X, Y = np.meshgrid(x, y)
            if re_manual is not None:
                re = re_manual
            else:
                re = float(inf_data.posterior['r_eff'][i,int(sample)].values)
            inf_data.posterior['v_re'][i,int(sample)] = np.abs(kin_model.v_rad(X,Y, np.radians( float(inf_data.posterior['PA'][i,int(sample)].values)), np.radians(float(inf_data.posterior['i'][i,int(sample)].values)), float(inf_data.posterior['Va'][i,int(sample)].values),  float(inf_data.posterior['r_t'][i,int(sample)].values), re)/np.sin(np.radians( float(inf_data.posterior['i'][i,int(sample)].values)))) #np.radians(float(inf_data.posterior['i'][i,int(sample)].values))

    # Process prior samples separately with their own loop
    for sample in range(num_samples_prior):
        x = np.linspace(0 - kin_model.x0_vel_mean, kin_model.im_shape[1]-1 - kin_model.x0_vel_mean, kin_model.im_shape[1]*kin_model.factor)
        y = np.linspace(0 - kin_model.y0_vel_mean, kin_model.im_shape[0]-1 - kin_model.y0_vel_mean, kin_model.im_shape[0]*kin_model.factor)
        X, Y = np.meshgrid(x, y)
        if re_manual is not None:
            re = re_manual
        else:
            # Use posterior median r_eff for prior v_re calculation
            re = float(inf_data.posterior['r_eff'].median(dim=["chain", "draw"]).values)
        inf_data.prior['v_re'][0,int(sample)] = np.abs(kin_model.v_rad(X,Y, np.radians( float(inf_data.prior['PA'][0,int(sample)].values)), np.radians(float(inf_data.prior['i'][0,int(sample)].values)), float(inf_data.prior['Va'][0,int(sample)].values),  float(inf_data.prior['r_t'][0,int(sample)].values), re)/np.sin(np.radians( float(inf_data.prior['i'][0,int(sample)].values)))) #np.radians(float(inf_data.posterior['i'][i,int(sample)].values))

    v_re_16 = inf_data.posterior['v_re'].quantile(0.16).values
    v_re_med =inf_data.posterior['v_re'].quantile(0.50).values
    v_re_84 = inf_data.posterior['v_re'].quantile(0.84).values
    return inf_data, v_re_16, v_re_med, v_re_84


def compute_MAP(inf_data, grism_object, image):
    factor = 2
    wave_factor = 10
    PA_post = np.array(inf_data.posterior['PA'].data)
    vals, counts = np.unique(PA_post, return_counts=True)
    PA_map = vals[counts == np.max(counts)][0]
    inc_post = np.array(inf_data.posterior['i'].data)
    vals, counts = np.unique(inc_post, return_counts=True)
    inc_map = vals[counts == np.max(counts)][0]
    Va_post = np.array(inf_data.posterior['Va'].data)
    vals, counts = np.unique(Va_post, return_counts=True)
    Va_map = vals[counts == np.max(counts)][0]
    r_t_post = np.array(inf_data.posterior['r_t'].data)
    vals, counts = np.unique(r_t_post, return_counts=True)
    r_t_map = vals[counts == np.max(counts)][0]
    sigma0_post = np.array(inf_data.posterior['sigma0'].data)
    vals, counts = np.unique(sigma0_post, return_counts=True)
    sigma0_map = vals[counts == np.max(counts)][0]




    image_shape = 31
    print(image_shape//2)
    x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
    y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
    x,y = jnp.meshgrid(x,y)
    kin_model = models.KinModels()
    # kin_model.compute_factors(jnp.radians((PA_map)), jnp.radians(inc_map), x,y)
    V = kin_model.v( x, y, jnp.radians((PA_map)), jnp.radians(inc_map), Va_map, r_t_map)
    D = sigma0_map*jnp.ones((31*factor, 31*factor))

    oversampled_image = oversample(image, factor, factor)
    mock_grism_high = grism_object.disperse(oversampled_image, V, D)

    grism_MAP = resample(mock_grism_high, 1*factor, wave_factor)

    return grism_MAP, PA_map, inc_map, Va_map, r_t_map, sigma0_map


def find_PA_morph(img, n_draw = 1e3):

    img = np.where(img < 0, 0, img)
    prob = img.flatten()
    prob /= np.sum(prob)
    img_shape = img.shape
    x_cont = np.arange(img_shape[0] * img_shape[1]).reshape(img_shape[0], img_shape[1])
    xy = np.random.choice(x_cont.flatten(), int(n_draw), p=prob)

    x_ind, y_ind = np.unravel_index(xy, img.shape)


    # plt.plot(x_ind, y_ind, '.')
    # plt.show()


    cov = np.cov(x_ind, y_ind)
    lambda_, v = np.linalg.eig(cov)
    angle=np.rad2deg(np.arccos(v[0, 0]))

    print('PA: ',(angle))

    return (180-angle)

def deconvolve_PSF_approx(image, psf, PA, niter = 35):
    limit = 0.12  #will have to play around with this value, seems to work for now
    deconvolved_image = restoration.richardson_lucy(image, psf, clip= False, filter_epsilon=limit, num_iter=niter)

    #using the deconvolved image, rescale the convolved one
    mid = image.shape[0]//2
    factor = float(deconvolved_image[mid,mid]/image[mid,mid])
    full_factors = np.ones((image.shape[0], image.shape[1]))*factor
    for i in range(image.shape[0]):
        for j in range(image.shape[0]):
            # mean_factor[i,j] = mean_factor[i,j]*(1 - np.abs(j-15)/15)
            full_factors[i,j] =  full_factors[i,j]*(1 - np.abs(j-15)/(7*np.sin(np.radians(PA)) + 3))*(1 - np.abs(i-15)/(7*np.cos(np.radians(PA)) + 3))
    renorm_image = image*full_factors

    return deconvolved_image

#things to add here
#sampling uniform
#creating x,y meshgrid (which takes 3 lines usually)
#computing Bayes factyor


#      --> functions not currently being used
# def oversample_errors(error_map, factor, wave_factor):
#     # Repeat each element in the original array along both dimensions
#     repeated_errors = jnp.kron(error_map, jnp.ones((factor, wave_factor)))

#     # Divide each element by the respective oversampling factors
#     oversampled_errors = repeated_errors / jnp.sqrt(factor*wave_factor)

#     return oversampled_errors


def resample_errors(error_map, factor, wave_factor):

    blocks = error_map.reshape(
        (int(error_map.shape[0]/(factor)), factor, int(error_map.shape[1]/wave_factor), wave_factor))
    # resampled_errors = jnp.sqrt(jnp.sum(blocks**2, axis=(1,3)))
    resampled_errors = np.linalg.norm(blocks, axis=(1, 3))
    return resampled_errors


@jax.jit
def bn_approx(n):
    n_safe = jnp.where(n!=0, n, 0.00001)
    bn_value = 2 * n_safe - 1 / 3 + 4 / (405 * n_safe) + 46 / (25515 * n_safe**2) + 131 / (1148175 * n_safe**3) - 2194697 / (30690717750 * n_safe**4)
        
    return jnp.where(n != 0, bn_value, 0.0)  # Still use jnp.where to return 0 for n=0


@jax.jit
def sersic_profile(x,y,amplitude, r_eff , n , x_0 , y_0 , ellip , theta, c=0):
        
    # import tensorflow_probability as tfp
    n_safe = jnp.where(n>0, n, 0.00001)
    bn = bn_approx(n) #tfp.math.igammainv(jnp.array(2.0 * n), 0.5)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    x_maj = jnp.abs((x - x_0) * cos_theta + (y - y_0) * sin_theta)
    x_min = jnp.abs(-(x - x_0) * sin_theta + (y - y_0) * cos_theta)

    ellip_safe = jnp.where(ellip < 1, ellip, 0.9)
    r_eff_safe = jnp.where(r_eff > 0, r_eff, 0.01)
        # r_eff_safe = 4

    # b = (1 - ellip_safe) * r_eff_safe
        # b_safe = jnp.where((ellip!=1) & (r_eff!=0) , jnp.where((ellip!=1) & (r_eff!=0),b,0.00001), 0.000001)
    b_safe = jnp.where(ellip<1 ,(1 - ellip_safe) * r_eff_safe, 0.00001)

    expon = 2.0 #+ c
    inv_expon = 1.0 / expon
    squared = jnp.where(((x_maj>0) | (x_min>0)) & (b_safe!=0) & (r_eff>0), ((x_maj / r_eff_safe) ** expon + (x_min / b_safe) ** expon), 1e-5)
    z = jnp.where(((x_maj>0) | (x_min>0)) & (b_safe!=0) & (r_eff>0), jnp.where(squared>0,jnp.sqrt(squared), 0.0), 1e-5)
        
    return jnp.where((n_safe>0) & ((x_maj>0) | (x_min>0)) & (b_safe!=0) & (r_eff>0), amplitude * jnp.exp(-bn * (z ** (1 / n_safe) - 1.0)), amplitude * jnp.exp(bn))

@jax.jit
def compute_adaptive_sersic_profile(x, y,intensity , r_eff, n, x0, y0, ellip, PA_sersic):
    '''
        Notes about inputs:
        -x and y have to be oversampled to the right amount that you want to obtain at the end 
        -r_eff is given in the original pixel size
        -if x and y are centered on zero then x0=y0 = 0

    '''

    image_shape = x.shape[0]
    # Create grids for different oversampling factors
    oversample_factor_inner = 125
    oversample_factor_outer = 5
    boundary_radius = 0.2 * r_eff
    boundary_half_size = 0.2 * r_eff
   # Define the inner square region mask
    inner_region_mask = jnp.array(
        (jnp.abs(x - x0) <= boundary_half_size) & 
        (jnp.abs(y - y0) <= boundary_half_size)
    )
    x_inner,y_inner = jnp.where(inner_region_mask, x, 0.0), jnp.where(inner_region_mask, y, 0.0)
    x_min,x_max = jnp.min(x_inner), jnp.max(x_inner)
    y_min,y_max = jnp.min(y_inner), jnp.max(y_inner)
    x_size = 9 #jnp.array((x_max-x_min)/(x[0,1]-x[0,0])+1).astype(int)
    y_size = 9 #jnp.array((y_max-y_min)/(y[1,0]-y[0,0])+1).astype(int)

    #define the 2D arrays of indices of the x and y array corresponding to this mask
    # dx,dy =  jnp.abs(x[0,1]-x[0,0]), jnp.abs((y[1,0]-y[0,0]))
    # ind_x_min = ((x_min - x[0,0])/dx).astype(int)
    # ind_x_max = ((x_max - x[0,0])/dx).astype(int)
    # ind_y_min = ((y_min - y[0,0])/dy).astype(int)
    # ind_y_max = ((y_max - y[0,0])/dy).astype(int)

    
    # Outer region mask is the complement of the inner region mask
    outer_region_mask = ~inner_region_mask
    # print('zoomed: ', zoom(x, 25, mode = 'nearest'))
    
    # Inner region (oversampled by 125)
    x_inner_grid_low,y_inner_grid_low = jnp.meshgrid(jnp.linspace(x_min, x_max, x_size), jnp.linspace(y_min, y_max, y_size ))

    x_inner_grid = image.resize(x_inner_grid_low, ((oversample_factor_inner*x_size), (oversample_factor_inner*y_size)), method='linear')
    y_inner_grid = image.resize(y_inner_grid_low, ((oversample_factor_inner*x_size), (oversample_factor_inner*y_size)), method='linear')

    galaxy_model_inner = sersic_profile(
            x_inner_grid, y_inner_grid, intensity, r_eff,n ,
            x0, y0, ellip, PA_sersic
        )/oversample_factor_inner**2

    galaxy_model_inner_resampled = resample(galaxy_model_inner, oversample_factor_inner, oversample_factor_inner)


    galaxy_model_inner_resampled_full = jnp.zeros((image_shape, image_shape))
    galaxy_model_inner_resampled_full = galaxy_model_inner_resampled_full.at[int(image_shape//2 - x_size//2):int(image_shape//2 + x_size//2+1), int(image_shape//2 - y_size//2):int(image_shape//2 + y_size//2+1)].set(galaxy_model_inner_resampled)

    

    x_outer_grid = image.resize(x, (int(oversample_factor_outer*image_shape), int(oversample_factor_outer*image_shape)), method='linear')
    y_outer_grid = image.resize(y, (int(oversample_factor_outer*image_shape), int(oversample_factor_outer*image_shape)), method='linear')

    galaxy_model_outer = sersic_profile(
            x_outer_grid, y_outer_grid, intensity, r_eff,n ,
            x0, y0, ellip,
            PA_sersic)/oversample_factor_outer**2


    # Resample to original grid
    galaxy_model_outer_resampled = resample(galaxy_model_outer, oversample_factor_outer, oversample_factor_outer)


    galaxy_model = jnp.where(inner_region_mask, galaxy_model_inner_resampled_full, galaxy_model_outer_resampled)
    
    # plt.imshow(galaxy_model, origin='lower')
    # plt.title('Galaxy model')
    # plt.show()

    #the returned image has the shape of the input x and y arrays
    return galaxy_model


def flux_to_Ie(flux,n,r_e,ellip):
    # calculate surface brightness at Re [in arcsec] from total flux [arbitrary flux units]
    bn = bn_approx(n)
    G2n = gamma(2*n)
    q = 1 - ellip
    Ie = flux / (r_e**2 * q * 2*jnp.pi * n * jnp.exp(bn) * bn**(-2*n) * G2n )
    return Ie

def Ie_to_flux(Ie,n,r_e,ellip):
    # calculate surface brightness at Re [in arcsec] from total flux [arbitrary flux units]
    bn = bn_approx(n)
    G2n = gamma(2*n)
    q = 1 - ellip
    total_flux = Ie* (r_e**2 * q * 2*jnp.pi * n * jnp.exp(bn) * bn**(-2*n) * G2n )
    return total_flux


def choose_mspf(bithash_file, psf_dir, RA, DEC, image_list):
    '''
    Inpiuts: the bithash file - need to download this
    the directory where the PSFs are stored
    the RA and DEC of the object in the catalog
    a list (with 1 entry) of the image in the wanted filter so the code can read info from the header

    Returns: a 1-item list of the path of the PSF to use for this object!!!

    => based on which program has the most exposures for that object 
    '''

    #initialize empty psf_list for this object
    psf_list = []

    #load bithash file
    bitheader = fits.getheader(bithash_file, 'BITHASH')
    bitmap = fits.getdata(bithash_file, 'BITHASH')
    w = WCS(bitheader)

    #match object coordinates to find bithash
    xx, yy = w.world_to_pixel_values(RA, DEC)
    bithash = bitmap[int(yy), int(xx)]

    #get decoder table
    decoder = fits.getdata(bithash_file,'DECODER')

    #get bithash to decode
    bithash = np.int64(bithash)

    #get binary rep of bit hash
    bin = np.binary_repr(bithash)

    #print(f'Bithash to decode: {bithash}')

    n = len(bin)

    #create dictionary of bits associated with program fits files
    obs_dict = {}

    for i in range(n):
      if(int(bin[n-1-i])==1):
        if(int(bin[n-1-i])==1):
          xi = np.where(decoder['bit']==i)[0]
          obs_dict[i] = []
          for index in xi:
            #print(f"Bit: {i} Field: {decoder['image'][index]}")
            obs_dict[i].append(decoder['image'][index])
    #for each image in image_list, copy the dictionary, remove values that correspond to other images (ie filters),
    #and choose the preferred mpsf files from the pared-down dictionary
    nimgs = len(image_list)

    for k in range(nimgs):

        #look through the obs_dict in order and stop when you find a program with the needed filter
        basename = os.path.basename(image_list[k])
        filter_lower = basename.split('_')[1].lower()
        #re-order the obs_dict so the keys are from smallest to largest
        obs_dict_copy = dict(sorted(obs_dict.items()))
        print(obs_dict_copy)

        if 8 in obs_dict_copy.keys():
            #if we have a 3215 psf, use it
            mpsf_file = obs_dict_copy[8][0].split('/')[-1]
            psf_path = f'{psf_dir}/mpsf_{mpsf_file}'
        else:
            #look through the rest
            for key, value in obs_dict_copy.items():
                #extract the program name from the full title 
                program = value[0].split('/')[-1].split('.')[0]
                #check if that program has a psf in the required filter

                # Set the directory to search in
                search_dir = Path(psf_dir)
                # Multiple words (all must be present)
                words = [program, filter_lower]
                # print(words)
                matching_files = [
                    f for f in search_dir.rglob("*")
                    if f.is_file() and all(word in f.name for word in words)
                ]

                #if matching_files is not empty, we have a match
                if matching_files:
                    #print(f"Found matching file for {program} in filter {filter_lower}: {matching_files[0]}")
                    print(matching_files[0])
                    psf_path = str(matching_files[0])
                    break
            try:
                psf_path
            except NameError:
                #if we didn't find a match, allow for module specific PSFs
                print(f"No matching file found for {program} in filter {filter_lower}, using module specific PSF")
                #remove the .fits from the filter name
                filter_lower_flex = filter_lower[:-5]
                for key, value in obs_dict_copy.items():
                    #extract the program name from the full title 
                    program = value[0].split('/')[-1].split('.')[0]
                    #check if that program has a psf in the required filter

                    # Set the directory to search in
                    search_dir = Path(psf_dir)
                    # Multiple words (all must be present)
                    words = [program, filter_lower_flex]
                    # print(words)
                    matching_files = [
                        f for f in search_dir.rglob("*")
                        if f.is_file() and all(word in f.name for word in words)
                    ]

                    #if matching_files is not empty, we have a match
                    if matching_files:
                        #print(f"Found matching file for {program} in filter {filter_lower}: {matching_files[0]}")
                        # print(matching_files[0])
                        psf_path = str(matching_files[0])
                        break
                try:
                    psf_path
                except NameError:
                    #if there is still no match, use the general mpsf for that filter and field
                    print(f"No matching file found for {program} in filter {filter_lower_flex}, using general mpsf")
                    # Set the directory to search in
                    search_dir = Path(psf_dir)
                    # Multiple words (all must be present)
                    # print(words)
                    psf_path = str(search_dir) + f'/mpsf_{filter_lower}'
                    print(f"Using general mpsf: {psf_path}")

        #change psf_path according to field / lux location
        # psf_path = f'{psf_dir}/mpsf_{mpsf_file}'
        if 'dec' in psf_path:
            psf_path = psf_path.replace('dec', 'oct')

        psf_list.append(psf_path)

    return psf_list

@jax.jit
def rotate_coords(x, y, xc, yc, theta):
    '''
    Rotate (x, y) around (xc, yc) by angle theta (in radians).
    Rotation is CLOCKWISE.
    '''
    dx, dy = x - xc, y - yc
    x_rot = xc + jnp.cos(theta) * dx + jnp.sin(theta) * dy
    y_rot = yc - jnp.sin(theta) * dx + jnp.cos(theta) * dy

    return x_rot, y_rot


def flux_lambda_to_nu(F_lambda, lambda_microm):
    """
    Convert flux density from erg/s/cm²/microm to erg/s/cm²/Hz.

    Parameters:
    -----------
    F_lambda : float or array-like
        Spectral flux density in erg/s/cm²/nm.
    lambda_nm : float or array-like
        Wavelength in nanometers (nm) corresponding to F_lambda.

    Returns:
    --------
    F_nu : float or array-like
        Spectral flux density in erg/s/cm²/Hz.
    """
    c = 2.99792458e10  # speed of light in cm/s
    lambda_cm = lambda_microm * 1e-4  # convert microm to cm
    F_nu = F_lambda * (lambda_cm**2) / c
    return F_nu

def fnu_to_mjy(f_nu):
    """
    Convert flux density from erg/s/cm²/Hz to mJy.

    Parameters:
    -----------
    f_nu : float or array-like
        Spectral flux density in erg/s/cm²/Hz.

    Returns:
    --------
    f_mjy : float or array-like
        Spectral flux density in mJy.
    """
    return f_nu * 1e26  # Convert from erg/s/cm²/Hz to mJy


def int_flux_to_flux_density(int_flux, lambda_micron, delta_lambda_microm):
    """
    Convert integrated flux to flux density.

    int_flux = total EL flux in erg/s/cm2
    """
    delta_lambda_cm = delta_lambda_microm * 1e-4  # convert microm to cm
    int_flux_per_micron = int_flux / delta_lambda_cm  # in erg/s/cm²/cm
    total_disp_flux = flux_lambda_to_nu(int_flux_per_micron, lambda_micron)  #in erg/s/cm²/Hz

    flux_density = fnu_to_mjy(total_disp_flux)  # Convert to mJy

    return flux_density