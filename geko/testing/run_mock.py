# importing my own modules
from geko import grism_dev
from geko import preprocess_dev as pre
from geko import postprocess as post
from geko import utils
from geko import plotting
from geko import models
from geko import fitting

from geko.fitting import Fit_Numpyro
from numpyro.infer import Predictive
from jax import random

import os

import math

import time

import jax
from jax import image
import jax.numpy as jnp
import numpy as np
import arviz as az

import argparse

import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.modeling.models import Gaussian2D, Sersic2D

from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import convolve as convolve_astropy

from photutils.datasets import make_noise_image

from numpyro.infer.util import log_likelihood

from jax.scipy.signal import convolve
import numpyro
from astropy.cosmology import Planck18 as cosmo

if 'gpu' in jax.devices():
	print('Using GPU')
	numpyro.set_platform('gpu')
numpyro.set_host_device_count(2)
numpyro.enable_validation()
jax.config.update('jax_enable_x64', True)

# import faulthandler
# faulthandler.enable()

import smplotlib


from jax import config
# config.update("jax_debug_nans", True)
# JAX_DEBUG_NANS = True
# JAX_TRACEBACK_FILTERING=False



def read_config_table(config_path, test):
	'''
		Read config table and load values of the parameters for every iteration of the 
		test into arrays
	'''
	config = Table.read(config_path, format='ascii')
	config_test = config[config['test'] == test]
	PA_image = np.array(config_test['PA_image'])
	PA_grism = np.array(config_test['PA_grism'])
	i = np.array(config_test['i'])
	Va = np.array(config_test['Va'])
	r_t = np.array(config_test['r_t'])
	sigma0 = np.array(config_test['sigma0'])
	SN_image = np.array(config_test['SN_image'])
	SN_grism = np.array(config_test['SN_grism'])
	n = np.array(config_test['n'])

	return PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n

def make_image(PA_image, i, r_t, SN_image, n, psf, image_shape):
	'''
		Make mock image from inputs
	'''
	# from inclination infer the ellipticity
	axis_ratio = utils.compute_axis_ratio(i, q0 = 0.2)
	ellip = 1 - axis_ratio
	print('Ellipticity: ' + str(ellip) + ', inclination: ' + str(i))
	#infer r_eff from the turnaround radius
	r_eff = (1.676/0.4)*1 #*r_t
	print('Reff: ', r_eff)
	time_start = time.time()
	# galaxy_model = Sersic2D(amplitude=1/27**2, r_eff = r_eff*27, n =1, x_0 = image_shape//2*27 + 13 , y_0 = image_shape//2*27 +13, ellip = ellip, theta=(90 - PA_image)*np.pi/180) #function takes theta in rads
	ny = nx = image_shape
	# y, x = np.mgrid[0:ny*81, 0:nx*81]
	# image = jnp.array(galaxy_model(x, y))


	# galaxy_model = utils.sersic_profile(x,y,1, r_eff*81, 1, image_shape//2*81 + 40, image_shape//2*81 + 40, ellip, (90 - PA_image)*np.pi/180)/81**2
	# image = utils.resample(galaxy_model, 81, 81)
	x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
	y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
	x,y = jnp.meshgrid(x,y)
	# x_grid = image.resize(x, (image_shape*5*25, image_shape*5*25), method='linear')
	# y_grid = image.resize(y, (image_shape*5*25, image_shape*5*25), method='linear')
	
	# mock_image_superhigh = utils.sersic_profile(x_grid,y_grid,1/(25*5)**2, r_eff, 1, 0, 0, ellip, (90 - PA_image)*np.pi/180)
	# mock_image_highres = utils.resample(mock_image_superhigh, 25, 25)
	# mock_image = utils.resample(mock_image_superhigh, 5*25, 5*25)

	x_grid = image.resize(x, (image_shape*5, image_shape*5), method='linear')
	y_grid = image.resize(y, (image_shape*5, image_shape*5), method='linear')

	Ie = utils.flux_to_Ie(200,1,r_eff, ellip)
	#need to replace with n!!! but running prior test now
	mock_image_highres = utils.compute_adaptive_sersic_profile(x_grid, y_grid, Ie/(5)**2, r_eff,1, 0, 0, ellip, (90 - PA_image)*np.pi/180)
	mock_image = utils.resample(mock_image_highres, 5, 5)

	# image = image.at[15,15].set(4)
	max_image = jnp.max(mock_image)
	print('SN image: ' + str(SN_image) + ', max image: ' + str(max_image) + ', max_image/sn: ' + str(max_image/SN_image))
	# noise = max_image/SN_image*np.random.normal(0,1, (image_shape, image_shape))
	noise = make_noise_image((mock_image.shape[0], mock_image.shape[1]), distribution='gaussian', mean=0, stddev=max_image/SN_image)
	
	noise_image = mock_image + noise
	convolved_image = convolve(mock_image, psf, mode='same')
	convolved_noise_image = convolve(mock_image, psf, mode='same') + noise
	
	# Define X and Y axes based on the image shape
	nx, ny = mock_image.shape
	x = np.linspace(0 - nx//2, nx - 1 - nx//2, nx)*0.06  # X-axis
	y = np.linspace(0 - nx//2, ny - 1 - nx//2, ny)*0.06  # Y-axis
	X, Y = np.meshgrid(x, y)  # Create meshgrid for plotting

	fig, axs = plt.subplots(3, 1, figsize=(4, 7))  # Create a figure with 3 vertical subplots

	# First plot: Mock Image
	pc1 = axs[0].pcolormesh(X, Y, mock_image, cmap='PuBu', shading='auto')
	# axs[0].set_title('Mock Image')
	axs[0].text(0.5, 0.85, r'Sersic profile', transform=axs[0].transAxes, ha='center', fontsize=15, fontweight='bold')
	# axs[0].set_xlabel(r'$\Delta $RA [arcsec]' )
	axs[0].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc1, ax=axs[0], orientation='vertical',  label = r'Flux [a.u.]')  # Add colorbar to the first subplot

	# Second plot: Convolved Mock Image
	pc2 = axs[1].pcolormesh(X, Y, convolved_image, cmap='PuBu', shading='auto')
	# axs[1].set_title('Convolved Mock Image')
	axs[1].text(0.5, 0.85, r'$+$ PSF convolution', transform=axs[1].transAxes, ha='center', fontsize=15, fontweight='bold')

	# axs[1].set_xlabel(r'$\Delta $RA [arcsec]' )
	axs[1].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc2, ax=axs[1], orientation='vertical', label = r'Flux [a.u.]')  # Add colorbar to the second subplot

	# Third plot: Convolved Mock Image with Noise
	pc3 = axs[2].pcolormesh(X, Y, convolved_noise_image, cmap='PuBu', shading='auto')
	# axs[2].set_title('Convolved Mock Image with Noise')
	axs[2].text(0.5, 0.85, r'$+$ Gaussian noise', transform=axs[2].transAxes, ha='center', fontsize=15, fontweight='bold')

	axs[2].set_xlabel(r'$\Delta $RA [arcsec]' )
	axs[2].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc3, ax=axs[2], orientation='vertical', label = r'Flux [a.u.]')  # Add colorbar to the third subplot

	# Adjust layout to prevent overlap
	plt.tight_layout()
	plt.show()



	return mock_image, mock_image_highres, convolved_image, noise_image, convolved_noise_image

def initialize_grism(mock_image, psf, image_shape):
	#create wave space
	wave_factor = 9
	delta_wave = 0.001
	wavelength = 4.5
	delta_wave_cutoff = 0.02
	wave_first = 4.0
	wave_space = jnp.linspace(wave_first, 5.0, int(1/delta_wave)+1)
	# wave_space= jnp.arange(3.0, 4.0, delta_wave)
	# print(wave_space)
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/delta_wave) #+10
	index_max = round((wave_max - wave_first)/delta_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/delta_wave)
	#set other free parameters
	y_factor = 1
	factor = 5
	x0_grism = y0_grism = image_shape//2
	xcenter_detector = ycenter_detector = 1024
	redshift = 5.0
	grism_filter = 'F444W'
	PSF = psf

	# --- go back to this oversampling if otber doesn't work (and implement in preprocess too) ---
	# wave_space_2d = jnp.reshape(wave_space, (1, wave_space.shape[0]))
	# # wave_space_oversampled = jnp.linspace(3.0, 4.0, int(wave_factor/delta_wave)) #+1)
	# wave_space_oversampled = image.resize(wave_space_2d, (1, wave_space.shape[0]*wave_factor), method='linear')[0]

	half_step = (delta_wave / wave_factor)*(wave_factor//2)
	wave_space_oversampled = np.arange(wave_space[0]- half_step, wave_space[-1] + delta_wave + half_step, delta_wave / wave_factor)

	#initialize grism object

	grism_object = grism_dev.Grism(image_shape, 0.0629/factor, icenter = y0_grism, jcenter = x0_grism, wavelength = wavelength, wave_space = wave_space_oversampled, index_min = (index_min)*wave_factor, index_max = (index_max+1)*wave_factor, 
					   grism_filter = grism_filter, grism_module = 'A', grism_pupil = 'R', PSF = PSF)

	return grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min 

def make_vel_fields(PA_grism, i ,Va, r_t, sigma0, image_shape, factor =5):
	'''
		Make velocity and velocity dispersion fields from inputs
	'''
	x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
	y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
	x,y = jnp.meshgrid(x,y)
	x_grid = image.resize(x, (image_shape*factor, image_shape*factor), method='linear')
	y_grid = image.resize(y, (image_shape*factor, image_shape*factor), method='linear')
	
	# x,y= utils.oversample(x, factor, factor)*factor**2, utils.oversample(y, factor, factor)*factor**2
	# print(image_shape//2)
	kin_model = models.KinModels()
	# kin_model.compute_factors(jnp.radians(PA_grism), jnp.radians(i), x,y)
	V = kin_model.v( x_grid, y_grid, PA_grism, i, Va, r_t)
	D = sigma0*jnp.ones_like(V)

	# x_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor*10)
	# y_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor*10)
	# x_10,y_10 = jnp.meshgrid(x_10,y_10)

	# V_10 = kin_model.v( x_10, y_10, PA_grism, i, Va, r_t)
	# D_10 = sigma0*jnp.ones_like(V_10)

	# V_10 = utils.resample(V_10, 10, 10)/10**2
	# D_10 = utils.resample(D_10, 10, 10)/10**2

	# print(jnp.argwhere(V==0))

	plt.imshow(V, origin='lower')
	plt.colorbar()
	plt.title('Mock Velocity Field')
	plt.show()

	return V, D

def make_mock_data(PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n, psf,image_shape = 31, factor = 5):
	'''
		Make mock images and grism spectra from inputs
	'''
	#make direct image
	image, image_highres, convolved_image, noise_image, convolved_noise_image = make_image(PA_image, i, r_t, SN_image, n, psf, image_shape)	
	max_image = jnp.max(image)
	image_error = (max_image/SN_image)*jnp.ones((image_shape, image_shape))
	#make grism object
	grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min = initialize_grism(convolved_image, psf, image_shape)
	#make velocity and velocity dispersion fields
	print('Params for vel fields: PA = ' + str(PA_grism) + ', i = ' + str(i) + ', Va = ' + str(Va) + ', r_t = ' + str(r_t) + ', sigma0 = ' + str(sigma0))
	# V,D = make_vel_fields((180-PA_grism)-180, i, Va, r_t, sigma0, image_shape)
	V,D = make_vel_fields(PA_grism, i, Va, r_t, sigma0, image_shape)


	# V = 200*np.ones((image_shape*factor, image_shape*factor))
	# D = 5*jnp.ones((image_shape*factor, image_shape*factor))

	# plt.imshow(V, origin='lower')
	# plt.colorbar()
	# plt.title('Mock Velocity Field')
	# plt.show()
	#make grism spectrum

	oversample_image = image_highres# utils.oversample(image, factor, factor, method='bilinear')

	# plt.imshow(oversample_image, origin='lower', cmap = 'BuGn')
	# plt.colorbar()
	# plt.title('Oversampled Mock Image')
	# plt.show()

	grism_spectrum = grism_object.disperse(oversample_image, V, D)
	# plt.imshow(grism_spectrum, origin='lower', cmap = 'BuGn')
	# plt.colorbar()
	# plt.show()
	#resample to grism resolution
	grism_spectrum = utils.resample(grism_spectrum, factor, wave_factor)
	# plt.imshow(grism_spectrum, origin='lower', cmap = 'inferno', vmin = 0.0, vmax = grism_spectrum.max())
	# plt.colorbar()
	# plt.title('Mock Grism Spectrum')
	# plt.show()

	# plt.imshow(grism_spectrum, origin='lower', cmap = 'inferno')
	# plt.colorbar()
	# plt.show()


	max_grism = jnp.max(grism_spectrum)
	#add noise to the grism spectrum
	grism_noise = make_noise_image((grism_spectrum.shape[0], grism_spectrum.shape[1]), distribution='gaussian', mean=0, stddev=max_grism/SN_grism) 
	grism_spectrum_noise = grism_spectrum + grism_noise
	grism_error = (max_grism/SN_grism)*jnp.ones((grism_spectrum.shape[0], grism_spectrum.shape[1]))

	mask = jnp.where(grism_spectrum_noise/grism_error < 1, 0, 1)
	# print(grism_spectrum_noise[mask.astype(bool)].sum())
	# print((grism_spectrum_noise*mask).sum())


	fig, axs = plt.subplots(2, 1, figsize=(5, 6))

	# Define the X and Y axes
	X = np.linspace(3.5 - delta_wave_cutoff, 3.5 + delta_wave_cutoff, grism_spectrum.shape[1] + 1)
	Y = np.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape + 1)*0.06

	# First subplot: Mock 2D Grism Spectrum
	pc1 = axs[0].pcolormesh(X, Y, grism_spectrum, cmap='PuBu', shading='auto')
	# axs[0].set_title('Mock 2D Grism Spectrum')
	axs[0].text(0.5, 0.85, r'Dispersed image', transform=axs[0].transAxes, ha='center', fontsize=15, fontweight='bold')
	# axs[0].set_xlabel(r'$\lambda$ [microns]') #r'$$\lambda$$ [microns]
	axs[0].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc1, ax=axs[0], orientation='vertical', label = r'Flux [a.u.]')

	# Second subplot: Mock 2D Grism Spectrum with Noise
	pc2 = axs[1].pcolormesh(X, Y, grism_spectrum_noise, cmap='PuBu', shading='auto')
	# axs[1].set_title('Mock 2D Grism Spectrum with Noise')
	# axs[1].text(0.5, 0.85, r'$+$ Gaussian noise', transform=axs[1].transAxes, ha='center', fontsize=15, fontweight='bold')
	axs[1].set_xlabel(r'$\lambda$ [microns]')
	axs[1].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc2, ax=axs[1], orientation='vertical', label = r'Flux [a.u.]')

	plt.tight_layout()
	plt.show()

	#compute the integrated SN of the grism spectrum
	#sum in quadrature the SN in each pixel
	#make a mask for the high SN pixels
	mask = jnp.where(grism_spectrum_noise/grism_error < 1, 0, 1)
	int_sn = np.sqrt(jnp.where(grism_spectrum_noise/grism_error < 1, 0.0, (grism_spectrum_noise/grism_error)**2).sum())
	print('Integrated SN of the grism spectrum: ' + str(int_sn))
		


	# Define the boxcar height and center the extraction region
	ny, nx = grism_spectrum_noise.shape
	center_y = ny // 2
	box_height = 5
	y_min = max(0, center_y - box_height // 2)
	y_max = min(ny, center_y + box_height // 2)
		
	# Perform boxcar extraction
	extracted_1d = np.sum(grism_spectrum_noise[y_min:y_max, :], axis=0)
	noise_1d = np.sqrt(np.sum(grism_error[y_min:y_max, :]**2, axis=0))  # Combine noise quadratically
	
	# Compute the integrated S/N
	integrated_signal = np.sum(extracted_1d)
	integrated_noise = np.sqrt(np.sum(noise_1d**2))
	integrated_sn = integrated_signal / integrated_noise
	
	# Print the integrated S/N
	print('Integrated S/N:', integrated_sn)

	#plot the 1D spectrum
	x_axis = np.linspace(3.5 - delta_wave_cutoff, 3.5 + delta_wave_cutoff, grism_spectrum.shape[1])
	fig, ax = plt.subplots(1, 1, figsize=(5, 3))
	ax.plot(x_axis, extracted_1d, label='Extracted 1D Spectrum')
	ax.plot(x_axis, noise_1d, label='Noise 1D Spectrum')
	ax.set_xlabel(r'$\lambda$ [microns]')
	ax.set_ylabel('Flux [a.u.]')
	ax.legend()
	plt.show()
		
	size = grism_spectrum_noise.shape[0]
	grism_spectrum_noise_square = grism_spectrum_noise #[:, grism_spectrum_noise.shape[1]//2 - size//2: grism_spectrum_noise.shape[1]//2 + size//2 + 1]
	grism_error_square = grism_error# [:, grism_error.shape[1]//2 - size//2: grism_error.shape[1]//2 + size//2 + 1]

	plt.imshow(grism_spectrum_noise_square)
	plt.show()
	sigma_rms = jnp.minimum((grism_spectrum_noise/grism_error).max(),5)
	im_conv = convolve_astropy(grism_spectrum_noise_square, make_2dgaussian_kernel(3.0, size=5))

	plt.imshow(im_conv)
	plt.show()
	# print('pre-bckg')
	bkg = Background2D(grism_spectrum_noise_square, (15, 15), filter_size=(5, 5), exclude_percentile=99.0)
	print('bckg:', bkg.background_median)
	# print('threshold:', sigma_rms*bkg.background_rms[10])
	# print('grism:',  grism_spectrum_noise_square[10])
	segment_map = detect_sources(im_conv, sigma_rms*np.abs(bkg.background_median), npixels=10)
	# segm_deblend = deblend_sources(im_conv, segment_map, npixels=10, nlevels=32, contrast=1, progress_bar=False)
	source_cat = SourceCatalog(grism_spectrum_noise_square, segment_map, convolved_data=im_conv, error=grism_error_square)
	source_tbl = source_cat.to_table()
	# print(source_tbl)

	plt.imshow(segment_map)
	plt.show()

	# identify main label
	main_label = segment_map.data[int(0.5*grism_spectrum_noise_square.shape[0]), int(0.5*grism_spectrum_noise_square.shape[1])]
	# snr = source_tbl['segment_flux']/source_tbl['segment_fluxerr']
	# idx_signifcant = (snr > 10.0) | (source_tbl['label']==main_label)
	
	print(main_label)

	# construct mask
	mask = segment_map.data
	new_mask = np.zeros_like(mask)
	new_mask[mask == main_label] = 1.0
	# mask[mask > 0] = 0.0

	# print(np.array(mask.shape))
	grism_spectrum_noise_square_masked = jnp.where(new_mask == 1, grism_spectrum_noise_square, np.nan)
	#set to zero the masked_rows in the grism spectrum
	plt.imshow(grism_spectrum_noise_square_masked, origin='lower', cmap = 'PuRd')
	plt.title('Inference model mask')
	plt.xlabel('Wavelength')
	plt.ylabel('Spatial Position')
	plt.colorbar()
	plt.show()
	

	# return image, image_error, image, grism_spectrum, grism_error, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object
	return convolved_image, image_error, image, grism_spectrum_noise, grism_error, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object

def run_fit(mock_params, priors,parametric = False):
	'''
		Run the fitting code
	'''

	line = 'H_alpha'
	#need to make a preprocessing function just for the mock data, probably add an entry to run_full_pre that defaults to none
	
	redshift, wavelength, wave_space, obs_map, obs_error, model_name, kin_model, grism_object, \
		  num_samples, num_warmup, step_size, target_accept_prob, delta_wave, factor = pre.run_full_preprocessing(None, None, line, mock_params, priors)
	
	mask = (jnp.where(obs_map/obs_error < 5.0, 0, 1)).astype(bool) 
	# ----------------------------------------------------------running the inference------------------------------------------------------------------------
	kin_model.disk.set_parametric_priors_test(priors)
	run_fit = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object, kin_model=kin_model, inference_data=None, parametric = parametric)

	rng_key = random.PRNGKey(4)

	#check truth likelihood
	if parametric:
		inference_model = run_fit.kin_model.inference_model_parametric
	else:
		inference_model = run_fit.kin_model.inference_model
	prior_predictive = Predictive(inference_model, num_samples=num_samples)

	prior = prior_predictive(rng_key, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error, mask = mask)

	
	run_fit.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
							  median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2, init_vals = None)

	#get highest likelihood sample and compute liklihood

	inf_data = az.from_numpyro(run_fit.mcmc, prior=prior)	
	# best_indices = np.unravel_index(inf_data['sample_stats']['lp'].argmin(), inf_data['sample_stats']['lp'].shape)
	# best_fit = inf_data.posterior.isel(chain=best_indices[0], draw=best_indices[1])	
	# best_fit_vals = best_fit
	# log_l_array = log_likelihood(run_fit.kin_model.inference_model, best_fit_vals, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error )
	# print('nan indices: ', np.argwhere(np.isnan(log_l_array['obs'])))
	# print('Log-L of best fit:',np.array(log_l_array['obs']))

	return inf_data, kin_model, grism_object, num_samples


def save_results(config_path, inf_data, test, j, r_t, kin_model, grism_object, num_samples, parametric):
	'''
		Save every result in a table so I can easily read it into a file to make all of the plots
		Save the output file + summary ONLY for each mock run, all in the same folder where the mock
		data is saved
		save with name str(test) + index of row for that test
	'''
		# no ../ because the open() function reads from terminal directory (not module directory)
	#save the output file
	inf_data.to_netcdf('testing/' + str(test) + '/' + str(test) + '_' + str(j) + '_'+ 'output')
	#post process results
	# inf_data, model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object,parametric)
	#load results table
	config_table = Table.read(config_path, format='ascii')
	config_table_test = config_table[config_table['test'] == test]

	#create a new table for results
	params_single = ['PA', 'i', 'Va', 'r_t', 'sigma0', 'v_re']
	all_params_single = [[i + "_q16", i + "_q50", i + "_q84"] for i in params_single]
	cat_col = np.append(["v_re"], np.concatenate(all_params_single))
	t_empty = np.zeros((len(cat_col), 1))
	res = Table(t_empty.T, names=cat_col)
	#obtain quantiles for each parameter from the posterior distribution
	params = [ 'PA', 'i', 'Va', 'r_t' ,'sigma0', 'v_re']
	quantiles = [0.16, 0.50, 0.84]
		#compute the azimuthally average velocity at the effective radius
	# r_eff = kin_model.r_eff_mean
	inf_data, v_re_16, v_re_med, v_re_84 = utils.add_v_re(inf_data, kin_model, grism_object, num_samples)

	for ii_p in params_single:
		res[ii_p + "_q16"] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 16)
		res[ii_p + "_q50"] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 50)
		res[ii_p + "_q84"] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 84)
	# for param in params:
	# 	for quantile in quantiles:
	# 		param_quantile = inf_data.posterior[param].quantile(quantile).values
	# 		print(param_quantile)
	# 		res[j] = param_quantile


	# res_test['v_re_16'][j] = v_re_16
	# res_test['v_re_50'][j] = v_re_med
	# res_test['v_re_84'][j] = v_re_84
	#add the truth v_re
	#read truth values from the config table
	PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n = read_config_table(config_path, test)
	#initialize kin_model with the truth values
	PA = np.radians(PA_image[j])
	i = np.radians(i[j])
	Va = Va[j]
	r_t = r_t[j]
	x = np.linspace(0 - 31//2, 31 - 31//2 - 1, 31*grism_object.factor)
	y = np.linspace(0 - 31//2, 31 - 31//2 - 1, 31*grism_object.factor)
	x,y = np.meshgrid(x,y)
	r_eff = (1.676/0.4)*r_t
	v_re_truth = kin_model.v_rad(x,y, PA, i, Va, r_t, r_eff)/np.sin(i)
	res['v_re'] = v_re_truth
	res.write('testing/' + str(test) + '/' + 'results_' + str(j), format='ascii', overwrite=True)
	return v_re_med, v_re_truth, kin_model


def run_test(test, j, config_path, parametric, PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n, psf, params_single, res):
	'''
		Wrapper function to run the test for the mock data
	'''
	convolved_noise_image, image_error, intrinsic_image, grism_spectrum_noise, grism_error, wave_space, \
	wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object \
	= make_mock_data(PA_image[j], PA_grism[j], i[j], Va[j], r_t[j], sigma0[j],SN_image[j], SN_grism[j], n[j], psf, image_shape= 31)
	#summarize ouputs in one mock_params dictionary
	print('Convolved mock image max pixel: ' + str(jnp.max(convolved_noise_image)))
	mock_params = {'test': test, 'j': j ,'convolved_noise_image': convolved_noise_image, 'image_error': image_error, 'grism_spectrum_noise': grism_spectrum_noise, 'grism_error': grism_error, 'wave_space': wave_space, 'wavelength': wavelength, 'delta_wave_cutoff': delta_wave_cutoff, 'y_factor': y_factor, 'wave_factor': wave_factor, 'index_max': index_max, 'index_min': index_min, 'grism_object': grism_object}
	priors = {'PA': PA_image[j], 'i': i[j], 'Va': Va[j], 'r_t': r_t[j], 'sigma0': sigma0[j], 'n': n[j]}
	#run fitting
	inf_data, kin_model, grism_object, num_samples  = run_fit(mock_params,priors, parametric = parametric)
	#post process inference data

	inf_data, model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object, parametric = parametric)	
	#save the masks in a fit file
	#create list 
	hdul = fits.HDUList()
	primary_hdu = fits.PrimaryHDU(kin_model.mask)
	primary_hdu.name = '2D_MASK'
	hdul.append(primary_hdu)
	mask_hdu = fits.ImageHDU(kin_model.masked_indices)
	mask_hdu.name = 'MASKED_IND'
	hdul.append(mask_hdu)
	hdul.writeto('testing/' + str(test) + '/' + str(j)+ '_masks', overwrite=True)	
	#save everything
	if parametric:
		plotting.plot_pp_cornerplot(inf_data,  kin_model = kin_model, choice = 'real', save_to_folder = str(test), name = str(j) + '_cornerplot_real', PA = PA_grism[j], i = i[j], Va = Va[j], r_t = r_t[j], sigma0 = sigma0[j])

	v_re_med, v_re_truth , kin_model= save_results(config_path, inf_data, test, j, r_t[j], kin_model, grism_object, num_samples,parametric)
	#plot the posteriors of the tuning parameters
	# plotting.plot_tuning_parameters(inf_data, model = 'Disk', save_to_folder = str(test), name = str(j) + '_tuning_parameters', scaling = False, error_scaling = False, errors = False, reg = False)
	inf_data.posterior['v_sigma'] = inf_data.posterior['v_re'] / inf_data.posterior['sigma0']
	inf_data.prior['v_sigma'] = inf_data.prior['v_re'] / inf_data.prior['sigma0']
#compute Mdyn posterior and quantiles
	pressure_cor = 3.35 #= 2*re/rd
	inf_data.posterior['v_circ2'] = inf_data.posterior['v_re']**2 + inf_data.posterior['sigma0']**2*pressure_cor
	inf_data.prior['v_circ2'] = inf_data.prior['v_re']**2 + inf_data.prior['sigma0']**2*pressure_cor
	inf_data.posterior['v_circ'] = np.sqrt(inf_data.posterior['v_circ2'])
	inf_data.prior['v_circ'] = np.sqrt(inf_data.prior['v_circ2'])
	ktot = 1.8 #for q0 = 0.2
	G = 4.3009172706e-3 #gravitational constant in pc*M_sun^-1*(km/s)^2
	DA = cosmo.angular_diameter_distance(3.0).to('m')
	meters_to_pc = 3.086e16
	# Convert arcseconds to radians and calculate the physical size
	inf_data.posterior['r_eff_pc'] = np.deg2rad(inf_data.posterior['r_eff']*0.06/3600)*DA.value/meters_to_pc
	inf_data.prior['r_eff_pc'] = np.deg2rad(inf_data.prior['r_eff']*0.06/3600)*DA.value/meters_to_pc
	inf_data.posterior['M_dyn'] = np.log10(ktot*inf_data.posterior['v_circ2']*inf_data.posterior['r_eff_pc']/G)
	inf_data.prior['M_dyn'] = np.log10(ktot*inf_data.prior['v_circ2']*inf_data.prior['r_eff_pc']/G)
	az.plot_trace(inf_data, var_names =['PA', 'i', 'Va','r_t', 'sigma0'], divergences = True)
	plt.savefig('testing/' + str(test) + '/' + str(j)+ '_chains.png', dpi=500)
	plt.show()
	#save the summary plot
	plt.close('all')
	kin_model.plot_summary(grism_spectrum_noise, grism_error, inf_data, wave_space[index_min:index_max+1], save_to_folder = str(test), name = str(j) + '_summary', v_re = v_re_med,  PA = PA_grism[j], i = i[j], Va = Va[j], r_t = r_t[j], sigma0 = sigma0[j])
	#plot and save delta map of the fluxes
	plt.close('all')
	median = np.where(kin_model.mask == 1, kin_model.fluxes_mean, 0.0)
	truth = np.where(kin_model.mask == 1, intrinsic_image, 0.0)
	chi = (median - truth)/kin_model.flux_error
	plt.imshow(chi, origin = 'lower', cmap = 'coolwarm')
	plt.colorbar()
	plt.title('Flux Chi')
	plt.savefig('testing/' + str(test) + '/' + str(j)+ '_fluxchi.png', dpi=500)
	plt.show()
	plt.close()

	# #plot MAP:
	
	# grism_MAP,PA_map, inc_map, Va_map, r_t_map, sigma0_map = utils.compute_MAP(inf_data, grism_object, convolved_noise_image)
	# plt.imshow(grism_MAP, origin='lower')
	# plt.colorbar()
	# plt.title('MAP: ' + str(PA_map) + ', ' + str(inc_map) + ', ' + str(Va_map) + ', ' + str(r_t_map) + ', ' + str(sigma0_map))
	# plt.xlabel('Wavelength')
	# plt.ylabel('Spatial Position')
	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP.png', dpi=500)
	# plt.close()

	# plt.imshow((grism_spectrum_noise - grism_MAP)/grism_error, origin='lower')
	# plt.colorbar()
	# plt.title('Residuals')
	# plt.xlabel('Wavelength')
	# plt.ylabel('Spatial Position')
	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP_residuals.png', dpi=500)
	# plt.close()

	#save all of the results in one table
		#obtain quantiles for each parameter from the posterior distribution
	params = ['PA', 'i', 'Va', 'r_t', 'sigma0', 'v_re']
	quantiles = [0.16, 0.50, 0.84]
	#compute the azimuthally average velocity at the effective radius
	inf_data, v_re_16, v_re_med, v_re_84 = utils.add_v_re(inf_data, kin_model, grism_object, num_samples)

	for ii_p in params_single:
		res[ii_p + "_q16"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 16)
		res[ii_p + "_q50"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 50)
		res[ii_p + "_q84"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 84)
	res['v_re'][j] = v_re_truth







