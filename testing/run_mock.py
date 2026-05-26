# importing my own modules
from geko import grism
from geko import preprocess as pre
from geko import postprocess as post
from geko import utils
from geko import plotting
from geko import models

from geko.fitting import Fit_Numpyro
from geko.config import FitConfiguration, MCMCSettings
from geko.postprocess import compute_derived_posterior, summarize_posterior, DERIVED_QUANTITIES
from geko.param_spec import _apply_overrides_to_specs, all_param_specs
from geko.models import GalaxyModel
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
import corner


from jax import config
# config.update("jax_debug_nans", True)
# JAX_DEBUG_NANS = True
# JAX_TRACEBACK_FILTERING=False



def read_config_table(config_path, test):
	'''
		Read config table and load values of the parameters for every iteration of the
		test into arrays. Returns standard geom arrays plus a flat params_dict of ALL
		columns — rotation-model params are accessed via params_dict by spec name.
	'''
	config = Table.read(config_path, format='ascii')
	config_test = config[config['test'] == test]
	params_dict = {col: np.array(config_test[col]) for col in config_test.colnames}

	# Derive r_eff from r_t if not explicitly in the table (Arctan: r_eff = (1.676/0.4)*r_t)
	if 'r_eff' not in params_dict and 'r_t' in params_dict:
		params_dict['r_eff'] = (1.676 / 0.4) * params_dict['r_t']

	PA_image = params_dict['PA_image']
	PA_grism = params_dict['PA_grism']
	i        = params_dict['i']
	sigma0   = params_dict['sigma0']
	SN_image = params_dict['SN_image']
	SN_grism = params_dict['SN_grism']
	n        = params_dict['n']

	return PA_image, PA_grism, i, sigma0, SN_image, SN_grism, n, params_dict

def make_image(PA_image, i, r_eff, SN_image, n, psf, image_shape, xc_morph=None, yc_morph=None):
	'''
		Make mock image from inputs

		Parameters
		----------
		r_eff : float
			Effective radius in pixels (passed directly from config table).
		xc_morph, yc_morph : float, optional
			Morphological center coordinates. If None, uses image_shape//2 (image center)
		psf : array
			PSF array to convolve with (already idealized for ideal mode if needed)
	'''
	axis_ratio = utils.compute_axis_ratio(i, q0 = 0.2)
	ellip = 1 - axis_ratio
	print('Ellipticity: ' + str(ellip) + ', inclination: ' + str(i))
	print('Reff: ', r_eff)

	# Use provided centers or default to image center
	if xc_morph is None:
		xc_morph = image_shape // 2
	if yc_morph is None:
		yc_morph = image_shape // 2

	time_start = time.time()
	# galaxy_model = Sersic2D(amplitude=1/27**2, r_eff = r_eff*27, n =1, x_0 = image_shape//2*27 + 13 , y_0 = image_shape//2*27 +13, ellip = ellip, theta=(90 - PA_image)*np.pi/180) #function takes theta in rads
	ny = nx = image_shape
	# y, x = np.mgrid[0:ny*81, 0:nx*81]
	# image = jnp.array(galaxy_model(x, y))


	# galaxy_model = utils.sersic_profile(x,y,1, r_eff*81, 1, image_shape//2*81 + 40, image_shape//2*81 + 40, ellip, (90 - PA_image)*np.pi/180)/81**2
	# image = utils.resample(galaxy_model, 81, 81)
	# Generate flux using direct high-res grid (Gemini's suggestion)
	# Avoids interpolation artifacts from image.resize
	x_grid = jnp.linspace(0 - xc_morph, image_shape - xc_morph - 1, image_shape*5)
	y_grid = jnp.linspace(0 - yc_morph, image_shape - yc_morph - 1, image_shape*5)
	x_grid, y_grid = jnp.meshgrid(x_grid, y_grid)

	Ie = utils.flux_to_Ie(200, n, r_eff, ellip)
	# Convert PA for Sersic profile (restoring 90-PA conversion)
	mock_image_highres = utils.sersic_profile(x_grid, y_grid, Ie/5**2, r_eff, n, 0, 0, ellip, (90 - PA_image)*np.pi/180)
	mock_image = utils.resample(mock_image_highres, 5, 5)

	# image = image.at[15,15].set(4)
	max_image = jnp.max(mock_image)
	print('SN image: ' + str(SN_image) + ', max image: ' + str(max_image) + ', max_image/sn: ' + str(max_image/SN_image))
	# noise = max_image/SN_image*np.random.normal(0,1, (image_shape, image_shape))
	noise = make_noise_image((mock_image.shape[0], mock_image.shape[1]), distribution='gaussian', mean=0, stddev=max_image/SN_image)

	# Use the PSF passed from RunGekoTests (already idealized for ideal mode)
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

def initialize_grism(mock_image, psf, image_shape, factor=5):
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

	grism_object = grism.Grism(image_shape*factor, 0.0629/factor, icenter = y0_grism, jcenter = x0_grism, wavelength = wavelength, wave_space = wave_space_oversampled, index_min = (index_min)*wave_factor, index_max = (index_max+1)*wave_factor, 
					   grism_filter = grism_filter, grism_module = 'A', grism_pupil = 'R', PSF = PSF)

	return grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min 

def make_vel_fields(PA_grism, i, truth_rot_params, sigma0, image_shape, x0_vel=None, y0_vel=None, factor=5):
	'''
		Make velocity and velocity dispersion fields from inputs.
		Model-agnostic: truth_rot_params dict is passed directly to GalaxyModel.velocity_field,
		which calls rot_model.rotation_curve internally — works for any rotation component.

		Parameters
		----------
		truth_rot_params : dict
			Rotation model parameters, e.g. {'Va': 200.0, 'r_t': 1.0} for Arctan.
		x0_vel, y0_vel : float, optional
			Velocity field center coordinates. If None, uses image_shape//2 (image center)
	'''
	# Use provided centers or default to image center
	if x0_vel is None:
		x0_vel = image_shape // 2
	if y0_vel is None:
		y0_vel = image_shape // 2

	x_grid = jnp.linspace(0 - x0_vel, image_shape - x0_vel - 1, image_shape*factor)
	y_grid = jnp.linspace(0 - y0_vel, image_shape - y0_vel - 1, image_shape*factor)
	x_grid, y_grid = jnp.meshgrid(x_grid, y_grid)

	gm = GalaxyModel((image_shape, image_shape), factor)
	V = gm.velocity_field(x_grid, y_grid, PA_grism, i, truth_rot_params)
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

def make_mock_data(PA_image, PA_grism, i, truth_rot_params, sigma0, SN_image, SN_grism, n, psf, r_eff, image_shape = 31, factor = 5, ideal = False, x0_vel=None, y0_vel=None, xc_morph=None, yc_morph=None, psf_mode='2d'):
	'''
		Make mock images and grism spectra from inputs

		Parameters
		----------
		x0_vel, y0_vel : float, optional
			Velocity field center coordinates
		xc_morph, yc_morph : float, optional
			Morphological center coordinates
		psf_mode : str, optional
			'2d' for standard 2D PSF, '1d' for 1D PSF (only spatial y-axis)
	'''
	#make direct image
	image, image_highres, convolved_image, noise_image, convolved_noise_image = make_image(PA_image, i, r_eff, SN_image, n, psf, image_shape, xc_morph=xc_morph, yc_morph=yc_morph)
	max_image = jnp.max(image)
	image_error = (max_image/SN_image)*jnp.ones((image_shape, image_shape))
	#make grism object
	grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min = initialize_grism(convolved_image, psf, image_shape, factor=factor)

	# Configure PSF and LSF settings for ideal vs realistic mode
	if ideal:
		# Ideal mode: PSF already set from RunGekoTests (sigma=0.5 pix, 11x11 grid)
		# Only need to set idealized LSF: better than real instrument but not too sharp
		# Real instrument LSF sigma ~ 0.003 microns, use 0.001 microns (3x better)
		grism_object.sigma_lsf = 0.0002
		grism_object.use_psf = True
		grism_object.use_lsf = True
		print(f'Ideal mode: Using idealized PSF from RunGekoTests and LSF (sigma={grism_object.sigma_lsf} um)')
	else:
		# Realistic mode: use instrument PSF and LSF
		grism_object.use_psf = True
		grism_object.use_lsf = True
		print(f'Realistic mode: Using instrument PSF and LSF')

	# Modify PSF for 1D mode — applies to both mock and inference (self-consistent)
	if psf_mode == '1d':
		# Create 1D PSF: Gaussian in y (spatial perpendicular to dispersion), delta in x
		# For row dispersion (pupil='R'): dispersion is along x, spatial is y
		psf_2d = grism_object.PSF[:, :, 0]  # Get 2D PSF from first wavelength slice
		psf_size_y, psf_size_x = psf_2d.shape

		# Integrate 2D PSF along x to get y-profile, delta function in x
		psf_1d_profile = jnp.sum(psf_2d, axis=1)
		psf_1d_profile = psf_1d_profile / jnp.sum(psf_1d_profile)

		psf_1d = jnp.zeros((psf_size_y, psf_size_x))
		center_x = psf_size_x // 2
		psf_1d = psf_1d.at[:, center_x].set(psf_1d_profile)

		grism_object.PSF = psf_1d[:, :, jnp.newaxis]
		print(f'PSF mode: 1D (spatial y-axis only) — used for both mock and inference')
		print(f'  Original 2D PSF shape: {psf_2d.shape}, 1D PSF shape: {psf_1d.shape}')
	else:
		print(f'PSF mode: 2D (standard convolution in both axes) — used for both mock and inference')
	#make velocity and velocity dispersion fields
	# Use inclination from config (removed hardcoded i=60)
	print('Params for vel fields: PA = ' + str(PA_grism) + ', i = ' + str(i) + ', rot_params = ' + str(truth_rot_params) + ', sigma0 = ' + str(sigma0))

	V, D = make_vel_fields(PA_grism, i, truth_rot_params, sigma0, image_shape, x0_vel=x0_vel, y0_vel=y0_vel, factor=factor)

	grism_spectrum = grism_object.disperse(image_highres, V, D)
	print(f'DEBUG: PSF shape before disperse: {grism_object.PSF.shape}')
	print(f'DEBUG: PSF sum: {jnp.sum(grism_object.PSF):.6f}')
	print(f'DEBUG: PSF non-zero elements: {jnp.sum(grism_object.PSF > 1e-10)}')
	grism_spectrum = utils.resample(grism_spectrum, factor, wave_factor)
	# plt.imshow(grism_spectrum, origin='lower', cmap = 'inferno', vmin = 0.0, vmax = grism_spectrum.max())
	# plt.colorbar()
	# plt.title('Mock Grism Spectrum')
	# plt.show()

	# plt.imshow(grism_spectrum, origin='lower', cmap = 'inferno')
	# plt.colorbar()
	# plt.show()


	max_grism = jnp.max(grism_spectrum)
	grism_error = (max_grism/SN_grism)*jnp.ones((grism_spectrum.shape[0], grism_spectrum.shape[1]))
	if ideal:
		#ideal mode: no noise added; data is the perfect noiseless signal
		grism_spectrum_noise = grism_spectrum
	else:
		#add noise to the grism spectrum
		grism_noise = make_noise_image((grism_spectrum.shape[0], grism_spectrum.shape[1]), distribution='gaussian', mean=0, stddev=max_grism/SN_grism)
		grism_spectrum_noise = grism_spectrum + grism_noise

	mask = jnp.where(grism_spectrum_noise/grism_error < 1, 0, 1)
	# print(grism_spectrum_noise[mask.astype(bool)].sum())
	# print((grism_spectrum_noise*mask).sum())


	fig, axs = plt.subplots(2, 1, figsize=(5, 6))

	# Define the X and Y axes
	X = np.linspace(4.5 - delta_wave_cutoff, 4.5 + delta_wave_cutoff, grism_spectrum.shape[1] + 1)
	Y = np.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape + 1)*0.06

	# First subplot: Mock 2D Grism Spectrum
	pc1 = axs[0].pcolormesh(X, Y, grism_spectrum, cmap='PuBu', shading='auto')
	axs[0].set_title('Mock 2D Grism Spectrum')
	axs[0].text(0.5, 0.85, r'Dispersed image', transform=axs[0].transAxes, ha='center', fontsize=15, fontweight='bold')
	# axs[0].set_xlabel(r'$\lambda$ [microns]') #r'$$\lambda$$ [microns]
	axs[0].set_ylabel(r'$\Delta $DEC [arcsec]')
	fig.colorbar(pc1, ax=axs[0], orientation='vertical', label = r'Flux [a.u.]')

	# Second subplot: Mock 2D Grism Spectrum with Noise
	pc2 = axs[1].pcolormesh(X, Y, grism_spectrum_noise, cmap='PuBu', shading='auto')
	axs[1].set_title('Mock 2D Grism Spectrum with Noise')
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
	y_max = min(ny, center_y + box_height // 2 + 1) 
		
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
	x_axis = np.linspace(4.5 - delta_wave_cutoff, 4.5 + delta_wave_cutoff, grism_spectrum.shape[1])
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
	

	# In ideal mode return the unconvolved image; otherwise return PSF-convolved image
	observed_image = image if ideal else convolved_image
	return observed_image, image_error, image, grism_spectrum_noise, grism_error, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object

def run_fit(mock_params, fit_config, parametric=False):
	'''
		Run the fitting code.

		Parameters
		----------
		fit_config : FitConfiguration
			Priors and MCMC settings. Replaces the old priors dict.
	'''
	line = 'H_alpha'

	z_spec, wavelength, wave_space, obs_map, obs_error, kin_model, grism_object, delta_wave = \
		pre.run_full_preprocessing(None, None, line, mock_params=mock_params)

	# Apply rotation model from config — raises NotImplementedError for unregistered components
	kin_model.galaxy_model.rot_model = fit_config.build_rot_model(z_spec)

	# Apply FitConfiguration to model (mirrors fitting.py)
	kin_model.galaxy_model.morph_model.apply_prior_overrides(fit_config.morph_prior_overrides)
	_apply_overrides_to_specs(kin_model.galaxy_model.shared_kin_specs,
	                          fit_config.geom_prior_overrides)
	for comp in kin_model.galaxy_model.rot_model.components:
		comp.apply_prior_overrides(fit_config.rot_prior_overrides)
	kin_model.galaxy_model.apply_fixed_params(fit_config.fixed_params)

	num_samples      = fit_config.mcmc.num_samples
	num_warmup       = fit_config.mcmc.num_warmup
	target_accept_prob = fit_config.mcmc.target_accept_prob
	num_chains       = fit_config.mcmc.num_chains
	step_size        = 1.0

	mask = jnp.ones_like(obs_map, dtype=bool)
	print('Using no mask (all pixels included)')

	run_fit_obj = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object,
	                          kin_model=kin_model, inference_data=None, parametric=parametric)

	rng_key = random.PRNGKey(4)

	if parametric:
		inference_model = run_fit_obj.kin_model.inference_model_parametric
	else:
		inference_model = run_fit_obj.kin_model.inference_model
	prior_predictive = Predictive(inference_model, num_samples=num_samples)
	prior = prior_predictive(rng_key, grism_object=run_fit_obj.grism_object,
	                         obs_map=run_fit_obj.obs_map, obs_error=run_fit_obj.obs_error,
	                         mask=mask)

	run_fit_obj.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
	                          median=True, step_size=step_size, adapt_step_size=True,
	                          target_accept_prob=target_accept_prob, num_chains=num_chains,
	                          init_vals=None, mask=mask)

	inf_data = az.from_numpyro(run_fit_obj.mcmc, prior=prior)

	return inf_data, kin_model, grism_object, num_samples, z_spec


def save_results(config_path, inf_data, z_spec, test, j, truth_rot_params, r_eff_truth, i_true,
                 kin_model, grism_object, num_samples, parametric, save_folder=None):
	'''
		Save MCMC output and per-iteration results table.
		Model-agnostic: truth v_re is computed via rot_model.rotation_curve(r_eff, truth_rot_params),
		which works for any rotation component. Parameter names are derived from all_param_specs.
	'''
	if save_folder is None:
		save_folder = test

	try:
		output_path = 'testing/' + str(save_folder) + '/' + str(save_folder) + '_' + str(j) + '_output'
		inf_data.to_netcdf(output_path)
		print(f'Saved MCMC output to {output_path}')
	except Exception as e:
		print(f'ERROR: Failed to save netcdf output: {e}')
		raise

	# Add derived quantities (v_re, v_sigma, v_circ, M_dyn) to posterior
	inf_data = compute_derived_posterior(inf_data, kin_model, z_spec)

	# Derive param names from model specs — no hardcoded list
	gm = kin_model.galaxy_model
	specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
	sampled_names = [s.name for s in specs if not s.fixed]
	derived_names = [dq.name for dq in DERIVED_QUANTITIES]
	all_names = sampled_names + derived_names
	summary = summarize_posterior(inf_data, all_names)

	# Build per-iteration results table
	row = {}
	for name in all_names:
		if name in summary:
			row[name + '_q16'] = summary[name]['16']
			row[name + '_q50'] = summary[name]['50']
			row[name + '_q84'] = summary[name]['84']
	res = Table([row]) if row else Table()

	# Truth v_re: model-agnostic via rotation_curve(r_eff, truth_rot_params)
	v_circ_truth = float(gm.rot_model.rotation_curve(jnp.array([r_eff_truth]), truth_rot_params)[0])
	v_re_truth = v_circ_truth * np.sin(np.radians(i_true))
	if len(res) > 0:
		res['v_re_truth'] = v_re_truth

	results_path = 'testing/' + str(save_folder) + '/' + 'results_' + str(j)
	res.write(results_path, format='ascii', overwrite=True)
	print(f'Saved per-iteration results table to {results_path}')

	v_re_med = summary.get('v_re', {}).get('50', np.nan)
	return summary, v_re_truth, kin_model


def run_test(test, j, config_path, parametric, PA_image, PA_grism, i, sigma0,
             SN_image, SN_grism, n, psf, params_dict, params_single, res, save_folder,
             psf_mode='2d', num_chains=2, num_warmup=1000, num_samples=1000,
             fit_config=None):
	'''
		Wrapper function to run the test for the mock data.
		Model-agnostic: rotation params are read from params_dict by spec name,
		not hardcoded as Va/r_t.

		Parameters
		----------
		params_dict : dict
			All config table columns as numpy arrays (from read_config_table).
		psf_mode : str, optional
			'2d' for standard 2D PSF (default), '1d' for 1D PSF in mock (y-axis only)
		num_chains, num_warmup, num_samples : int
			MCMC settings — passed via CLI args from RunGekoTests.
		fit_config : FitConfiguration, optional
			Base configuration. rotation_components is taken from here, so passing a
			custom config is how you test non-default rotation models. Prior overrides
			and MCMC settings are derived from the config table / CLI args and will
			override whatever is in the passed config. Defaults to FitConfiguration().
	'''
	os.makedirs('testing/' + save_folder, exist_ok=True)

	ideal = '_ideal' in save_folder

	image_shape = 31
	factor = 5
	xc_morph_true = 15.0
	yc_morph_true = 15.0
	x0_vel_true   = 15.0
	y0_vel_true   = 15.0

	if fit_config is None:
		fit_config = FitConfiguration()

	# Use the input config's rotation_components to read parameter spec names
	rot_model_temp = fit_config.build_rot_model()
	gm_temp = GalaxyModel((image_shape, image_shape), factor, rot_model=rot_model_temp)
	rot_param_names = [s.name for s in gm_temp.rot_model.parameters]
	truth_rot_params = {name: float(params_dict[name][j])
	                    for name in rot_param_names if name in params_dict}
	r_eff_true = float(params_dict['r_eff'][j])

	convolved_noise_image, image_error, intrinsic_image, grism_spectrum_noise, grism_error, wave_space, \
	wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object \
	= make_mock_data(PA_image[j], PA_grism[j], i[j], truth_rot_params, sigma0[j],
	                 SN_image[j], SN_grism[j], n[j], psf, r_eff_true,
	                 image_shape=image_shape, ideal=ideal,
	                 x0_vel=x0_vel_true, y0_vel=y0_vel_true,
	                 xc_morph=xc_morph_true, yc_morph=yc_morph_true, psf_mode=psf_mode)

	print('Convolved mock image max pixel: ' + str(jnp.max(convolved_noise_image)))
	print(f'Mock morphology centers: xc_morph={xc_morph_true}, yc_morph={yc_morph_true}')
	print(f'Mock velocity field centers: x0_vel={x0_vel_true}, y0_vel={y0_vel_true}')

	mock_params = {'test': test, 'j': j, 'convolved_noise_image': convolved_noise_image,
	               'image_error': image_error, 'grism_spectrum_noise': grism_spectrum_noise,
	               'grism_error': grism_error, 'wave_space': wave_space, 'wavelength': wavelength,
	               'delta_wave_cutoff': delta_wave_cutoff, 'y_factor': y_factor,
	               'wave_factor': wave_factor, 'index_max': index_max, 'index_min': index_min,
	               'grism_object': grism_object, 'PSF': psf}

	fit_config = FitConfiguration(
	    rotation_components=fit_config.rotation_components,
	    mcmc=MCMCSettings(num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples),
	    morph_prior_overrides={
	        'PA_morph_mu': PA_image[j], 'PA_morph_std': 5.0,
	        'r_eff_mu': r_eff_true, 'r_eff_std': float(max(3.0, r_eff_true)),
	        'r_eff_min': 0.0, 'r_eff_max': 15.0,
	        'n_mu': float(n[j]), 'n_std': 1.0, 'n_min': 0.36, 'n_max': 8.0,
	        'amplitude_mu': 200.0, 'amplitude_std': 40.0, 'amplitude_min': 0.0,
	        'xc_morph_mu': 15.0, 'xc_morph_std': 1.0,
	        'yc_morph_mu': 15.0, 'yc_morph_std': 1.0,
	    },
	    geom_prior_overrides={
	        'PA_mu': PA_image[j], 'PA_std': 10.0,
	        'i_mu': float(i[j]), 'i_std': 5.0,
	        'sigma0_min': 0.0, 'sigma0_max': 600.0,
	        'v0_mu': 0.0, 'v0_std': 200.0,
	    },
	)

	inf_data, kin_model, grism_object, num_samples_out, z_spec = run_fit(mock_params, fit_config, parametric=parametric)
	num_samples = num_samples_out

	inf_data, model_map, model_flux, fluxes_mean, model_velocities, model_dispersions = \
	    kin_model.compute_model(inf_data, grism_object, parametric=parametric)

	# --- Cornerplots (non-critical): var_names derived from model — model-agnostic ---
	try:
		gm = kin_model.galaxy_model
		specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
		_var_names = [s.name for s in specs if not s.fixed and s.name in inf_data.posterior]
		_labels    = [s.label for s in specs if not s.fixed and s.name in inf_data.posterior]
		# Truth values: geom params known directly; rot params from truth_rot_params
		_truths = []
		truth_lookup = {
		    'PA': PA_grism[j], 'i': float(i[j]), 'sigma0': float(sigma0[j]),
		    **truth_rot_params,
		}
		for name in _var_names:
		    _truths.append(truth_lookup.get(name, None))
		_post  = np.column_stack([np.concatenate(inf_data.posterior[v].values) for v in _var_names])
		_prior = np.column_stack([np.concatenate(inf_data.prior[v].values)     for v in _var_names])
		_fig = corner.corner(_prior, labels=_labels, color='lightgray',
		    plot_datapoints=False, plot_density=False, fill_contours=False,
		    plot_contours=False, smooth=2, max_n_ticks=3)
		corner.corner(_post, labels=_labels, color='blue', truths=_truths, truth_color='crimson',
		    plot_datapoints=False, plot_density=False, fill_contours=True,
		    smooth=2, quantiles=[0.16, 0.5, 0.84], show_titles=True,
		    title_kwargs=dict(fontsize=12), max_n_ticks=3, fig=_fig)
		plt.savefig('testing/' + save_folder + '/' + str(j) + '_cornerplot_kin.png', dpi=300)
		plt.close()
		if parametric:
		    plotting.plot_pp_cornerplot(inf_data, kin_model=kin_model, choice='real',
		        save_to_folder=save_folder, name=str(j) + '_cornerplot_real',
		        PA=PA_grism[j], i=float(i[j]), sigma0=float(sigma0[j]), **truth_rot_params)
	except Exception as e:
		print(f'Warning: cornerplot failed with error: {e}')
		plt.close('all')

	# --- Critical: save output file and per-iteration results table ---
	summary, v_re_truth, kin_model = save_results(
	    config_path, inf_data, z_spec, test, j,
	    truth_rot_params, r_eff_true, float(i[j]),
	    kin_model, grism_object, num_samples, parametric,
	    save_folder=save_folder)

	v_re_med = summary.get('v_re', {}).get('50', np.nan)

	# --- Trace and summary plots (non-critical) ---
	# Derived quantities (v_sigma, v_circ, M_dyn) already added by save_results
	try:
		trace_var_names = [v for v in _var_names if v in inf_data.posterior]
		az.plot_trace(inf_data, var_names=trace_var_names, divergences=True)
		plt.savefig('testing/' + save_folder + '/' + str(j) + '_chains.png', dpi=500)
		plt.show()
		plt.close('all')
		kin_model.plot_summary(grism_spectrum_noise, grism_error, inf_data,
		    wave_space[index_min:index_max+1], save_to_folder=save_folder,
		    name=str(j) + '_summary', v_re=v_re_med, PA=PA_grism[j], i=float(i[j]),
		    sigma0=float(sigma0[j]), **truth_rot_params)
		plt.close('all')
	except Exception as e:
		print(f'Warning: trace/summary plots failed with error: {e}')
		plt.close('all')
	plt.close('all')

	# --- Fill aggregate results table from summary dict ---
	for ii_p in params_single:
		if ii_p + '_q50' in res.colnames and ii_p in summary:
		    res[ii_p + '_q16'][j] = summary[ii_p]['16']
		    res[ii_p + '_q50'][j] = summary[ii_p]['50']
		    res[ii_p + '_q84'][j] = summary[ii_p]['84']
	if 'v_re' in res.colnames:
		res['v_re'][j] = v_re_truth







