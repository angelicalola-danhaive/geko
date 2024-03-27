"""

    This module contains utility functions for the Geko package.
    Not sure of everything that will go in here but it should be a place to put
    basic functions that are used in multiple places in the package.
	Written by A L Danhaive: ald66@cam.ac.uk
"""

# imports
from jax import image
import jax.numpy as jnp
from jax.scipy.stats import norm


from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import dilation, disk
from skimage.color import label2rgb

from matplotlib import pyplot as plt

def oversample(image_low_res, factor, wave_factor, method='nearest'):
    image_high_res = image.resize(image_low_res, (int(
        image_low_res.shape[0]*factor), int(image_low_res.shape[1]*wave_factor)), method=method)
    image_high_res /= factor*wave_factor

    return image_high_res


def resample(grism_spectrum, factor, wave_factor):
    # blocks = grism_spectrum[int(factor/4):int(grism_spectrum.shape[0]-factor/4)].reshape((int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
    blocks = grism_spectrum.reshape(
        (int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
    grism_obs_res = jnp.sum(blocks, axis=(1, 3))
    return grism_obs_res


def oversample_errors(error_map, factor, wave_factor):
    # Repeat each element in the original array along both dimensions
    repeated_errors = jnp.kron(error_map, jnp.ones((factor, wave_factor)))

    # Divide each element by the respective oversampling factors
    oversampled_errors = repeated_errors / jnp.sqrt(factor*wave_factor)

    return oversampled_errors


def resample_errors(error_map, factor, wave_factor):

    blocks = error_map.reshape(
        (int(error_map.shape[0]/(factor)), factor, int(error_map.shape[1]/wave_factor), wave_factor))
    # resampled_errors = jnp.sqrt(jnp.sum(blocks**2, axis=(1,3)))
    resampled_errors = jnp.linalg.norm(blocks, axis=(1, 3))
    return resampled_errors


def scale_distribution(distribution, mu, sigma, max, min):
    """

        Rescale a uniform distribution to its target distribution, chosen based on input parameters

    """
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
    """
        Rescale a distribution to a uniform distribution
    """
    rescaled_distribution = distribution * (max - min) + min
    return rescaled_distribution


def scale_gaussian(distribution, mu, sigma):
    """
        Rescale a distribution to a Gaussian distribution
    """
    rescaled_distribution = norm.ppf(distribution) * sigma + mu
    return rescaled_distribution

def scale_truncated_gaussian(distribution, mu, sigma, high, low):
    """
        Rescale a distribution to a truncated Gaussian distribution
    """
    rescaled_distribution = norm.ppf(norm.cdf(low) + distribution*(norm.cdf(high)-norm.cdf(low)))*sigma + mu
    return rescaled_distribution


def find_best_sample(inference_data, variable, mu, sigma, max, min, MLS):
    """
       Find the best sample from an array of variables and return it
       MLS is either none to take median sample or it contains the indices with the higest
       likelihood sample
    """
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


def make_mask(image, n, flux_threshold, dilation_factor):
    """
       Make n masks of the image, dilated by a factor dilation. If n =1, then flux_threshold
       gives the threshold. If n>1, the threshold is set to the smallest number needed to obtain n
       regions (using the find_threshold function).
    """
        
    # threshold =threshold_otsu(image)
    threshold = flux_threshold*image.max()
    mask = jnp.zeros_like(image)
    masked_indices = jnp.where(image[20:40,20:40]>threshold)
    mask = mask.at[masked_indices[0]+20,masked_indices[1]+20 ].set(1)
    masked_indices = jnp.where(image>threshold)
    mask = mask.at[masked_indices[0],masked_indices[1] ].set(1)
    if n == 1:
        mask = dilation(mask, disk(dilation_factor))
        return mask
    else:
        masked_image = image*mask
        #find the sub_regions only within the existing mask
        threshold, masks = find_threshold(masked_image, n, dilation_factor)
        return masks

def find_threshold(image, n, dilation_factor):
    """
       Find the threshold needed to cut the image into n regions
    """

    #initialize the mask at all zeros so that the while loop runs at least once
    label_image = jnp.zeros_like(image)
    #start the guess at the threshold otsu value
    threshold = threshold_otsu(image)
    increment = threshold/10
    while len(jnp.unique(label_image))-1 != n: #-1 because zeros count as one region too
        threshold += increment
        bw = closing(image > threshold)
        cleared = clear_border(bw)
        #label image regions
        label_image = label(cleared)
    #print the successful threshold

    #cut the image into n regions
    masks = []
    for i in range(0,n):
        mask = jnp.zeros_like(image)
        mask = mask.at[jnp.where(label_image==i+1)].set(1)
        mask = dilation(mask, disk(dilation_factor))
        masks.append(mask)

    return threshold, masks

def compute_PA(image):
	'''
		compute PA of galaxy in the image using skimage measure regionprops

	'''
    #compute the threshold from the central pixels only in case there's contaminant sources
	# threshold = threshold_otsu(image[20:40,20:40])
	threshold = threshold_otsu(image)/4
	bw = closing(image > threshold)

	cleared = clear_border(bw)

	#label image regions
	label_image = label(cleared)
	image_label_overlay = label2rgb(label_image, image=image)
	plt.imshow(image_label_overlay, origin = 'lower')
	plt.show()
	regions = regionprops(label_image, image)
	PA = 90 + regions[0].orientation * (180/jnp.pi)
	
	return PA


#things to add here
#sampling uniform
#creating x,y meshgrid (which takes 3 lines usually)
#computing Bayes factyor