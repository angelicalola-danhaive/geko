"""
    geko - utils
    This module contains utility functions for the Geko package.
    -- Any function mutliple times that is repeated in the code?
	Written by A L Danhaive: ald66@cam.ac.uk
"""

# geko imports
import run_pysersic as py

#other imports
from jax import image
import jax.numpy as jnp
from jax.scipy.stats import norm


from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import dilation, disk, ellipse, binary_closing
from skimage.color import label2rgb

from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources, SegmentationImage
from photutils.background import Background2D, MedianBackground
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.isophote import build_ellipse_model

from matplotlib import pyplot as plt

from astropy.io import fits

import numpy as np

import xarray as xr



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
    image_low_res = jnp.sum(blocks, axis=(1, 3))
    return image_low_res


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
    bkg = Background2D(image, (size,size) , filter_size=(5, 5),bkg_estimator = bkg_estimator)
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
    mask_1comp = dilation(mask_1comp.data, disk(3))
    plt.imshow(segment_map.data, origin='lower')
    plt.scatter(cat.xcentroid, cat.ycentroid, color='red')
    plt.show()
    plt.imshow(image*segment_map.data, origin='lower')
    plt.title('Mask of the central source')
    plt.show()

    #separate two sources if n == 2
    if n == 2:
        segm_deblend = deblend_sources(image, segment_map, npixels=10, nlevels=32, contrast=0.001)
        plt.imshow(segm_deblend.data, origin='lower')
        plt.title('Make mask - segm debelnd')
        plt.show()

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
    

def load_psf( filter, y_factor, size = 9, psf_folder = 'mpsf_gds/'):
    psf_fits = fits.open(psf_folder + 'mpsf_' + str(filter).lower() + '.fits') 
    psf_full = np.array(psf_fits[0].data)

    print(psf_full.shape)
    if y_factor == 1: #then you have to resample it down to 0.063 resolution
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

        psf = psf_downsampled
    else:
        #make cutout of the psf to size = size
        psf = psf_full[psf_full.shape[0]//2 - size//2:psf_full.shape[0]//2 + size//2 + 1, \
                    psf_full.shape[1]//2 - size//2:psf_full.shape[1]//2 + size//2 + 1]
    #renormalize the psf
    psf = psf/psf.sum()

    return psf

def compute_inclination(axis_ratio = None, ellip = None, q0 = 0.2):
    """
        Compute the inclination of a galaxy given its axis ratio or ellipticity
    """
    if axis_ratio == None:
        axis_ratio = 1 - ellip
    inc = np.arccos(np.sqrt( (axis_ratio**2 - q0**2)/(1 - q0**2) ))*(180/np.pi)
    return inc

def compute_axis_ratio(inc, q0 = 0.2):
    """
        Compute the axis ratio of a galaxy given its inclination
    """
    axis_ratio = np.sqrt(np.cos(np.radians(inc))**2*(1-q0**2) + q0**2)
    return axis_ratio

def fit_grism_parameters(obs_map, r_eff, inclination, obs_error, sigma_rms = 2.0):
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

    geom = EllipseGeometry(x0 = obs_map.shape[1]//2, y0 = obs_map.shape[0]//2, sma = r_eff, eps = 1 - np.cos(inclination*(np.pi/180)), pa = PA_mask*(np.pi/180))
    ellipse_shape = Ellipse(np.abs(obs_map), geometry = geom, threshold = 0.001)
    aper = EllipticalAperture((geom.x0, geom.y0), geom.sma,
	geom.sma * (1 - geom.eps),geom.pa)
    plt.imshow(obs_map, origin='lower')
    aper.plot(color='red')
    plt.show()
    isolist = ellipse_shape.fit_image()
    if not isolist:
        print('No isophotes found, using source catalog measurements instead.')
        source_cat = SourceCatalog(obs_map, SegmentationImage(segment_map_closed.astype(int)), convolved_data=im_conv, error=obs_error)
        angle = source_cat.orientation[0].value #orientation is in degrees [-180,180]
        if angle<0:
                angle += 180 #convert to [0,180]
        PA_grism = angle
        inc_grism = jnp.arccos(1/source_cat.elongation[0].value)* (180/jnp.pi)
        # print(source_cat.equivalent_radius[0].value, jnp.cos(PA_grism*(jnp.pi/180)))
        r_full_grism = source_cat.equivalent_radius[0].value*jnp.abs(jnp.cos(PA_grism*(jnp.pi/180)))
    else:
        # print(isolist.to_table())  
        model_grism = build_ellipse_model(obs_map.shape, isolist)
		#choose the isophote with the sma closest to morphological radius
        best_index = np.argmin(np.abs(isolist.sma - 1.5*r_eff)) #the 1.5 is a bit arbitrary ...
        PA_grism = isolist.pa[best_index]*(180/np.pi) #put in degrees
        inc_grism =jnp.cos(1-isolist.eps[best_index])*(180/np.pi) #put in degrees
        minor_axis = (1-isolist.eps[best_index])*isolist.sma[best_index]
        r_full_grism = jnp.maximum(isolist.sma[best_index]*jnp.abs(jnp.cos(PA_grism*(jnp.pi/180))), minor_axis*jnp.abs(jnp.cos((90-PA_grism)*(jnp.pi/180)))) #project onto velocity/x axis
        residual = obs_map - model_grism

        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
        ax1.imshow(obs_map, origin='lower')
        ax1.set_title('Data')

        sma = isolist.sma[best_index]
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax1.plot(x, y, color='white')

        ax2.imshow(model_grism, origin='lower')
        ax2.set_title('Ellipse Model')

        ax3.imshow(residual, origin='lower')
        ax3.set_title('Residual')
        plt.show()

        print('Isophote fitting results - ')
        print('PA: ', PA_grism, 'Inclination: ', inc_grism, 'r_full: ', r_full_grism)
    
    return PA_grism, inc_grism, r_full_grism


def add_v_re(inf_data, kin_model, grism_object, num_samples, r_eff):
    inf_data.posterior['v_re'] = xr.DataArray(np.zeros((2,num_samples)), dims = ('chain', 'draw'))
    for i in [0,1]:
        for sample in range(num_samples-1):
            x = np.linspace(0 - kin_model.x0_vel, grism_object.direct.shape[1]-1 - kin_model.x0_vel, grism_object.direct.shape[1]*grism_object.factor)
            y = np.linspace(0 - float(inf_data.posterior['y0_vel'][i,int(sample)].values), grism_object.direct.shape[0]-1 - float(inf_data.posterior['y0_vel'][i,int(sample)].values), grism_object.direct.shape[0]*grism_object.factor)
            X, Y = np.meshgrid(x, y)
            inf_data.posterior['v_re'][i,int(sample)] = kin_model.v_rad(X,Y, np.radians(float(inf_data.posterior['PA'][i,int(sample)].values)), np.radians(float(inf_data.posterior['i'][i,int(sample)].values)), float(inf_data.posterior['Va'][i,int(sample)].values), float(inf_data.posterior['r_t'][i,int(sample)].values), r_eff)
    v_re_16 = inf_data.posterior['v_re'].quantile(0.16).values
    v_re_med =inf_data.posterior['v_re'].quantile(0.50).values
    v_re_84 = inf_data.posterior['v_re'].quantile(0.84).values
    return inf_data, v_re_16, v_re_med, v_re_84




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


# def resample_errors(error_map, factor, wave_factor):

#     blocks = error_map.reshape(
#         (int(error_map.shape[0]/(factor)), factor, int(error_map.shape[1]/wave_factor), wave_factor))
#     # resampled_errors = jnp.sqrt(jnp.sum(blocks**2, axis=(1,3)))
#     resampled_errors = jnp.linalg.norm(blocks, axis=(1, 3))
#     return resampled_errors