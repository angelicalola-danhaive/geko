"""
Put all of the necessary pre-processing functions here so that fit_numpyro is cleaner
Eventually should also add here scripts to automatically create folders for sources, with all of the right cutouts etc
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""

#imports
import numpy as np
from astropy.io import ascii
from astropy import wcs
from astropy.wcs.utils import fit_wcs_from_points
from astropy.io import fits
from scipy import interpolate
from scipy.constants import c #in m/s
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy
import math
from scipy.constants import c,pi
import matplotlib.pyplot as plt
from jax.scipy import special

from scipy import ndimage

import jax.numpy as jnp
from jax import random, vmap
from jax import image
from jax.scipy.special import logsumexp

from reproject import reproject_interp, reproject_adaptive
from photutils.centroids import centroid_com, centroid_quadratic,centroid_1dg

#for the masking
from skimage.morphology import binary_dilation, dilation, disk



def read_config_file(input, output):
	"""

        Read the config file for the galaxy and returns all of the relevant parameters

	"""
	data = input[0]
	params = input[1]
	inference = input[2]
	priors = input[3]

	ID = data['Data']['ID']
		
	broad_filter = data['Data']['broad_filter']
		
	if broad_filter == 'F444W':
		med_filter = 'F410M'
	elif broad_filter == 'F356W':
		med_filter = 'F335M'
	
	#no ../ because the open() function reads from terminal directory (not module directory)
	med_band_path = 'fitting_results/' + output + str(ID) + '_' + med_filter + '.fits'
	broad_band_path = 'fitting_results/' + output + str(ID) + '_' + broad_filter + '.fits'
	grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_GDN_' + broad_filter + '_ID' + str(ID) + '_comb.fits'

	field = data['Data']['field']

	wavelength = data['Data']['wavelength']

	redshift = data['Data']['redshift']

	line = data['Data']['line']
								
	y_factor = params['Params']['y_factor']
	if y_factor == 1:
		print('Fitting the low res image')
		res = 'low'
	elif y_factor == 2:
		print('Fitting the high res image')
		res = 'high'
		
	to_mask = params['Params']['masking']
	flux_threshold = inference['Inference']['flux_threshold']
	
	delta_wave_cutoff = params['Params']['delta_wave_cutoff']
	factor = params['Params']['factor']
	wave_factor = params['Params']['wave_factor']
	
	x0 = params['Params']['x0']
	y0 = params['Params']['y0']
	

	x0_vel = np.array(params['Params']['x0_vel'])
	y0_vel = np.array(params['Params']['y0_vel'])

	model_name = inference['Inference']['model']
	
    #import all of the bounds needed for the priors
	flux_bounds = inference['Inference']['flux_bounds']
	flux_type = inference['Inference']['flux_type']
	PA_normal = inference['Inference']['PA_normal']
	i_bounds = inference['Inference']['i_bounds']
	Va_bounds = inference['Inference']['Va_bounds']
	r_t_bounds = inference['Inference']['r_t_bounds']
	sigma0_bounds = inference['Inference']['sigma0_bounds']
	sigma0_mean = inference['Inference']['sigma0_mean']
	sigma0_disp = inference['Inference']['sigma0_disp']
	obs_map_bounds = inference['Inference']['obs_map_bounds']

	clump_v_prior = inference['Inference']['clump_v_prior']
	clump_sigma_prior = inference['Inference']['clump_sigma_prior']
	clump_flux_prior = inference['Inference']['clump_flux_prior']
	
	clump_bool = inference['Inference']['clump_bool']
	
	num_samples = inference['Inference']['num_samples']
	num_warmup = inference['Inference']['num_warmup']
	
	step_size = inference['Inference']['step_size']
	target_accept_prob = inference['Inference']['target_accept_prob']
	
	#return all of the parameters
	return data, params, inference, priors, ID, broad_filter, med_filter, med_band_path, broad_band_path, grism_spectrum_path, field, wavelength, redshift, line, y_factor, res, to_mask, flux_threshold, factor, wave_factor, x0, y0, x0_vel, y0_vel, model_name, flux_bounds, flux_type, PA_normal, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, sigma0_mean, sigma0_disp, obs_map_bounds, clump_v_prior, clump_sigma_prior, clump_flux_prior, clump_bool, num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff


def renormalize_image(direct, obs_map, flux_threshold, y_factor):
	"""
		Normalize the image to match the total flux in the EL map
	"""

	threshold = flux_threshold*direct.max()
	mask = jnp.zeros_like(direct)
	mask = mask.at[jnp.where(direct>threshold)].set(1)
	mask = dilation(mask, disk(6*y_factor))

	#create a mask for the grism map
	threshold_grism = flux_threshold*obs_map.max()
	mask_grism = jnp.zeros_like(obs_map)
	mask_grism = mask_grism.at[jnp.where(obs_map>threshold_grism)].set(1)
	mask_grism = dilation(mask_grism, disk(6))

	#compute the normalization factor
	normalization_factor = obs_map[jnp.where(mask_grism == 1)].sum()/direct[jnp.where(mask == 1)].sum()
	#normalize the direct image to the grism image
	direct = direct*normalization_factor

	return direct, normalization_factor, mask, mask_grism