# importing my own modules
import grism
import preprocess as pre
import postprocess as post
import plotting
import utils
import models
import fitting

from fitting import Fit_Numpyro
from numpyro.infer import Predictive
from jax import random

import os

import math

import time

import jax
import jax.numpy as jnp
import numpy as np
import arviz as az

import argparse

import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.modeling.models import Gaussian2D, Sersic2D

from photutils.datasets import make_noise_image

from numpyro.infer.util import log_likelihood

from jax.scipy.signal import convolve
import numpyro
# numpyro.set_platform('gpu')
numpyro.set_host_device_count(2)
numpyro.enable_validation()
jax.config.update('jax_enable_x64', True)

# import faulthandler
# faulthandler.enable()


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

    return PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism

def make_image(PA_image, i, r_t, SN_image, psf, image_shape):
    '''
        Make mock image from inputs
    '''
    # from inclination infer the ellipticity
    axis_ratio = utils.compute_axis_ratio(i, q0 = 0.0)
    ellip = 1 - axis_ratio
    print('Ellipticity: ' + str(ellip) + ', inclination: ' + str(i))
    #infer r_eff from the turnaround radius
    r_eff = (1.676/0.4)*r_t
    print('Reff: ', r_eff)
    time_start = time.time()
    galaxy_model = Sersic2D(amplitude=1, r_eff = r_eff*27, n =1, x_0 = image_shape//2*27 + 13 , y_0 = image_shape//2*27 +13, ellip = ellip, theta=(90 - PA_image)*np.pi/180) #function takes theta in rads
    ny = nx = image_shape
    y, x = np.mgrid[0:ny*27, 0:nx*27]
    image = jnp.array(galaxy_model(x, y))
    image = utils.resample(image, 27, 27)/27**2
    time_end = time.time()
    # image = image.at[15,15].set(4)
    max_image = jnp.max(image)
    print('SN image: ' + str(SN_image) + ', max image: ' + str(max_image) + ', max_image/sn: ' + str(max_image/SN_image))
    # noise = max_image/SN_image*np.random.normal(0,1, (image_shape, image_shape))
    noise = make_noise_image((image.shape[0], image.shape[1]), distribution='gaussian', mean=0, stddev=max_image/SN_image)
    
    plt.imshow(image,origin='lower', cmap='PuRd')
    plt.title('Mock Image')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    
    time_start_manual = time.time()
    galaxy_model_manual = utils.sersic_profile(x,y,1, r_eff*27, 1, image_shape//2*27 + 13, image_shape//2*27 + 13, ellip, (90 - PA_image)*np.pi/180)
    galaxy_model_manual = utils.resample(galaxy_model_manual, 27, 27)/27**2
    time_end_manual = time.time()
    
    print('Time for model: ', time_end - time_start)
    print('Time for manual model: ', time_end_manual - time_start_manual)
    
    plt.imshow(galaxy_model_manual,origin='lower', cmap='PuRd')
    plt.title('Manual Sersic Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    plt.imshow(image-galaxy_model_manual,origin='lower')
    plt.colorbar()
    plt.title('Difference between two sersic functions')
    plt.show()
    
    print(utils.bn_approx(0.3), utils.bn_approx(5))
    
    image = galaxy_model_manual
    noise_image = image + noise
    convolved_image = convolve(image, psf, mode='same')
    convolved_noise_image = convolve(image, psf, mode='same') + noise
    plt.imshow(convolved_image,origin='lower', cmap='PuRd')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Convolved Mock Image')
    plt.show()
    
    plt.imshow(convolved_noise_image,origin='lower', cmap='PuRd')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Convolved Mock Image with Noise')
    plt.show()

    return image, convolved_image, noise_image, convolved_noise_image

def initialize_grism(image, psf, image_shape):
    #create wave space
	wave_factor = 9
	delta_wave = 0.001
	wavelength = 3.5
	delta_wave_cutoff = 0.02
	wave_space = jnp.linspace(3.0, 4.0, int(1/delta_wave)+1)
	# wave_space= jnp.arange(3.0, 4.0, delta_wave)
	print(wave_space)
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - 3.0)/delta_wave) #+10
	index_max = round((wave_max - 3.0)/delta_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - 3.0)/delta_wave)
    #set other free parameters
	y_factor = 1
	factor = 5
	x0_grism = y0_grism = image_shape//2
	xcenter_detector = ycenter_detector = 1024
	redshift = 3.0
	grism_filter = 'F356W'
	PSF = psf

	wave_space_oversampled = jnp.linspace(3.0, 4.0, int(wave_factor/delta_wave)) #+1)
    #initialize grism object
	grism_object = grism.Grism(direct=image, direct_scale=0.0629/y_factor, icenter=y0_grism, jcenter=x0_grism, segmentation=None, factor=factor, y_factor=y_factor,
                            xcenter_detector=xcenter_detector, ycenter_detector=ycenter_detector, wavelength=wavelength, redshift=redshift,
                            wave_space=wave_space_oversampled, wave_factor=wave_factor, wave_scale=delta_wave/wave_factor, index_min=(index_min)*wave_factor, index_max=(index_max)*wave_factor,
                            grism_filter=grism_filter, grism_module='A', grism_pupil='R', PSF = PSF)
	return grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min 

def make_vel_fields(PA_grism, i ,Va, r_t, sigma0, image_shape, factor =5):
	'''
		Make velocity and velocity dispersion fields from inputs
	'''
	x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
	y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
	x,y = jnp.meshgrid(x,y)
	
	# x,y= utils.oversample(x, factor, factor)*factor**2, utils.oversample(y, factor, factor)*factor**2
	# print(image_shape//2)
	kin_model = models.KinModels()
	# kin_model.compute_factors(jnp.radians(PA_grism), jnp.radians(i), x,y)
	V = kin_model.v( x, y, PA_grism, i, Va, r_t)
	D = sigma0*jnp.ones_like(V)

	# x_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor*10)
	# y_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor*10)
	# x_10,y_10 = jnp.meshgrid(x_10,y_10)

	# V_10 = kin_model.v( x_10, y_10, PA_grism, i, Va, r_t)
	# D_10 = sigma0*jnp.ones_like(V_10)

	# V_10 = utils.resample(V_10, 10, 10)/10**2
	# D_10 = utils.resample(D_10, 10, 10)/10**2

	print(jnp.argwhere(V==0))

	plt.imshow(V, origin='lower')
	plt.colorbar()
	plt.title('Mock Velocity Field')
	plt.show()

	return V, D

def make_mock_data(PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, psf,image_shape = 31, factor = 5):
	'''
        Make mock images and grism spectra from inputs
	'''
    #make direct image
	image, convolved_image, noise_image, convolved_noise_image = make_image(PA_image, i, r_t, SN_image, psf, image_shape)	
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

	oversample_image = utils.oversample(image, factor, factor, method='bicubic')

	# plt.imshow(oversample_image, origin='lower', cmap = 'BuGn')
	# plt.colorbar()
	# plt.title('Oversampled Mock Image')
	# plt.show()

	grism_spectrum = grism_object.disperse_mock(oversample_image, V, D)
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

	plt.imshow(grism_spectrum, origin='lower', cmap = 'PuRd')
	plt.title('Mock 2D Grism Spectrum')
	plt.xlabel('Wavelength')
	plt.ylabel('Spatial Position')
	plt.colorbar()
	plt.show()

	max_grism = jnp.max(grism_spectrum)
	#add noise to the grism spectrum
	grism_noise = make_noise_image((grism_spectrum.shape[0], grism_spectrum.shape[1]), distribution='gaussian', mean=0, stddev=max_grism/SN_grism) 
	grism_spectrum_noise = grism_spectrum + grism_noise
	grism_error = (max_grism/SN_grism)*jnp.ones((grism_spectrum.shape[0], grism_spectrum.shape[1]))

	mask = jnp.where(grism_spectrum_noise/grism_error < 3.0, 0, 1)
	print(grism_spectrum_noise[mask.astype(bool)].sum())
	print((grism_spectrum_noise*mask).sum())
	plt.imshow(grism_spectrum*mask, origin='lower', cmap = 'PuRd')
	plt.title('Mock 2D Grism Spectrum with Noise')
	plt.xlabel('Wavelength')
	plt.ylabel('Spatial Position')
	plt.colorbar()
	plt.show()

	return convolved_noise_image, image_error, image, grism_spectrum_noise, grism_error, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object

def run_fit(mock_params, priors):
	'''
		Run the fitting code
	'''

	line = 'H_alpha'
	#need to make a preprocessing function just for the mock data, probably add an entry to run_full_pre that defaults to none
	direct, direct_error, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism, x0_grism,\
	num_samples, num_warmup, step_size, target_accept_prob, \
	wave_space, delta_wave, index_min, index_max, factor = pre.run_full_preprocessing(None, None, line, mock_params, priors)
	
	mask = (jnp.where(obs_map/obs_error < 5.0, 0, 1)).astype(bool)
	# ----------------------------------------------------------running the inference------------------------------------------------------------------------

	run_fit = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object, kin_model=kin_model, inference_data=None)

	rng_key = random.PRNGKey(4)

	#check truth likelihood

	prior_predictive = Predictive(run_fit.kin_model.inference_model, num_samples=num_samples)

	prior = prior_predictive(rng_key, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error, mask = mask)

	log_l = run_fit.kin_model.log_posterior(grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error, values = {'PA':90, 'i':60, 'sigma0':80, 'Va':200, 'r_t':1, 'fluxes':direct, 'fluxes_error':direct_error})
	log_l_max = run_fit.kin_model.log_posterior(grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error, values = {'PA':90, 'i':57, 'sigma0':176, 'Va':143, 'r_t':0.83, 'fluxes':direct, 'fluxes_error':direct_error})

	print('Log-L of truth:',log_l)
	print('Log-L of max:',log_l_max)
	# total_L = []
	# Va_vals = np.linspace(0, 1, 100)
	# PA = 0.5
	# for Va in Va_vals:
	# 	init_vals = { 'unscaled_PA': PA, 'unscaled_i': 0.0, 'unscaled_sigma0': Va, 'unscaled_Va': 0.3, 'unscaled_r_t':0.0, 'unscaled_v0': 0.0,'unscaled_y0_vel': 0.0}
	# 	log_l_array = log_likelihood(run_fit.kin_model.inference_model, init_vals, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error )
	# 	sum_l = np.sum(np.array(log_l_array['obs']) - np.max(np.array(log_l_array['obs'])))
	# 	total_L.append(np.exp(sum_l))
	# 	# print('Log-L of truth:',np.sum(np.array(log_l_array['obs'])))
	# plt.plot(Va_vals,total_L, '+')
	# plt.title('PA = ' + str((PA*0.45+1.57)*180/np.pi))
	# plt.show()
	# sigma0_map = (Va_vals[np.argmax(total_L)])*200
	# PA_map = (PA*0.45+1.57)*180/np.pi
# --------------------------plot the MAP------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 	image_shape = 31
# 	print(image_shape//2)
# 	x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
# 	y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
# 	x,y = jnp.meshgrid(x,y)
# 	kin_model = models.KinModels()
# 	kin_model.compute_factors(jnp.radians((PA_map)), jnp.radians(60), x,y)
# 	V = kin_model.v( x, y, jnp.radians((PA_map)), jnp.radians(60), 300, 1)
# 	D = sigma0_map*jnp.ones((31*factor, 31*factor))

# 	oversampled_image = utils.oversample(direct, factor, factor)
# 	mock_grism_high = grism_object.disperse(oversampled_image, V, D)

# 	MAP_grism = utils.resample(mock_grism_high, y_factor*factor, wave_factor)
# 	plt.imshow(MAP_grism, origin='lower')
# 	plt.colorbar()
# 	# plt.title('MAP: ' + str(np.round(PA_map,2)) + ' ' + str(np.round(inc_map,2)) + ' ' + str(np.round(Va_map,2)) + ' ' + str(np.round(r_t_map,2)) + ' ' + str(np.round(sigma0_map,2)))
# 	plt.title('MAP')
# 	plt.xlabel('Wavelength')
# 	plt.ylabel('Spatial Position')
# 	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP.png', dpi=500)
# 	plt.show()

# 	plt.imshow((obs_map - MAP_grism)/obs_error, origin='lower')
# 	plt.colorbar()
# 	plt.title('OBS-MAP')
# 	plt.xlabel('Wavelength')
# 	plt.ylabel('Spatial Position')
# 	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP_residuals.png', dpi=500)
# 	plt.show()

# #now plot the truth
# 	image_shape = 31
# 	print(image_shape//2)
# 	x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
# 	y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*factor)
# 	x,y = jnp.meshgrid(x,y)
# 	kin_model = models.KinModels()
# 	kin_model.compute_factors(jnp.radians((PA_map)), jnp.radians(60), x,y)
# 	V = kin_model.v( x, y, jnp.radians((PA_map)), jnp.radians(60), 300, 1)
# 	D = 100*jnp.ones((31*factor, 31*factor))

# 	oversampled_image = utils.oversample(direct, factor, factor)
# 	mock_grism_high = grism_object.disperse(oversampled_image, V, D)

# 	mock_grism = utils.resample(mock_grism_high, y_factor*factor, wave_factor)
# 	plt.imshow(mock_grism, origin='lower')
# 	plt.colorbar()
# 	# plt.title('MAP: ' + str(np.round(PA_map,2)) + ' ' + str(np.round(inc_map,2)) + ' ' + str(np.round(Va_map,2)) + ' ' + str(np.round(r_t_map,2)) + ' ' + str(np.round(sigma0_map,2)))
# 	plt.title('TRUTH')
# 	plt.xlabel('Wavelength')
# 	plt.ylabel('Spatial Position')
# 	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP.png', dpi=500)
# 	plt.show()

# 	plt.imshow((MAP_grism - mock_grism)/obs_error, origin='lower')
# 	plt.colorbar()
# 	plt.title('MAP-TRUTH')
# 	plt.xlabel('Wavelength')
# 	plt.ylabel('Spatial Position')
# 	# plt.savefig('testing/' + str(test) + '/' + str(j)+ '_MAP_residuals.png', dpi=500)
# 	plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	# print('nan indices: ', np.argwhere(np.isnan(log_l_array['obs'])))
	# init_vals = { 'unscaled_i': 0.0, 'unscaled_sigma0': 0.5, 'unscaled_Va': 0.3,'unscaled_v0': 0.0,'unscaled_y0_vel': 0.0}

	#plot a dot on the nan indices in the obs_map array
	# plt.imshow(obs_map, origin='lower', cmap = 'PuRd')
	# plt.colorbar()
	# plt.scatter(np.argwhere(np.isnan(log_l_array['obs']))[:,1], np.argwhere(np.isnan(log_l_array['obs']))[:,0], c = 'r', s = 1)
	# plt.title('Obs Map')
	# plt.show()

	# print('Log-L of truth:',np.sum(np.array(log_l_array['obs'])))

	# best_fit_vals = {'unscaled_PA': 0.33, 'unscaled_i': -0.95, 'unscaled_sigma0': 0.92, 'Va': 0.41, 'unscaled_r_t':0.44, 'unscaled_v0': 0.01,'unscaled_y0_vel': 0.24}
	# log_l_array = log_likelihood(run_fit.kin_model.inference_model, best_fit_vals, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error )
	# print('Log-L of best fit:',np.sum(np.array(log_l_array['obs'])))

	# plt.imshow(obs_map, origin='lower', cmap = 'PuRd')
	# plt.colorbar()
	# plt.scatter(np.argwhere(np.isnan(log_l_array['obs']))[:,1], np.argwhere(np.isnan(log_l_array['obs']))[:,0], c = 'r', s = 1)
	# plt.title('Obs Map')
	# plt.show()

	# run_fit.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
	# 	                      median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2, init_vals = None)
	
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


def save_results(config_path, inf_data, test, j, r_t, kin_model, grism_object, num_samples):
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
	model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object)
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
	r_eff = kin_model.r_eff
	inf_data, v_re_16, v_re_med, v_re_84 = utils.add_v_re(inf_data, kin_model, grism_object, num_samples, r_eff)

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
	PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism = read_config_table(config_path, test)
	#initialize kin_model with the truth values
	PA = np.radians(PA_image[j])
	i = np.radians(i[j])
	Va = Va[j]
	r_t = r_t[j]
	x = np.linspace(0 - 31//2, 31 - 31//2 - 1, 31*grism_object.factor)
	y = np.linspace(0 - 31//2, 31 - 31//2 - 1, 31*grism_object.factor)
	x,y = np.meshgrid(x,y)
	r_eff = (1.676/0.4)*r_t
	v_re_truth = kin_model.v_rad(x,y, PA, i, Va, r_t, r_eff)
	res['v_re'] = v_re_truth
	res.write('testing/' + str(test) + '/' + 'results_' + str(j), format='ascii', overwrite=True)
	return v_re_med, v_re_truth, kin_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='testing/mock_results',
                    help='config table path') #config table name
parser.add_argument('--test', type=str, default='',
                    help='name of the test running') #test name
# parser.add_argument('--psf_path', type=str, default='gdn_mpsf_F356W_small.fits',
#                     help='psf file path') #psf file path --> add the default here

if __name__ == "__main__":
	
	args = parser.parse_args()
	config_path = args.config
	test = args.test
	# psf_path = args.psf_path

	PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism = read_config_table(config_path, test)
	psf = utils.load_psf(filter = 'F356W', y_factor = 1, size = 9)

	PA_image = np.array([90, 70,45,20,0])
	PA_grism = np.array([90, 70,45,20,0])

	#create table where the results will be saved
	params_single = [ 'PA', 'i', 'Va', 'r_t', 'sigma0', 'v_re']
	all_params_single = [[i + "_q16", i + "_q50", i + "_q84"] for i in params_single]
	config_params = ['PA_image', 'PA_grism', 'i', 'Va', 'r_t', 'sigma0', 'SN_image', 'SN_grism']
	cat_col = np.append(config_params, np.append(["v_re"], np.concatenate(all_params_single)))
	t_empty = np.zeros((len(cat_col), len(PA_image)))
	#fill the config columns with the values read from config file
	t_empty[0,:] = PA_image
	t_empty[1,:] = PA_grism
	t_empty[2,:] = i
	t_empty[3,:] = Va
	t_empty[4,:] = r_t
	t_empty[5,:] = sigma0
	t_empty[6,:] = SN_image
	t_empty[7,:] = SN_grism
	res = Table(t_empty.T, names=cat_col)


	for j in range(len(PA_image)) : #range(len(PA_image))
		# PA_grism[j] = 90
		# PA_image[j] = 90
		# SN_grism[j] = 50
		# sigma0[j] = 100
		print('Running test ' + str(test) + ' iteration ' + str(j))
		print('Parameters: PA_image = ' + str(PA_image[j]) + ', PA_grism = ' + str(PA_grism[j]) + ', i = ' + str(i[j]) + ', Va = ' + str(Va[j]) + ', r_t = ' + str(r_t[j]) + ', sigma0 = ' + str(sigma0[j]) + ', SN_image = ' + str(SN_image[j]) + ', SN_grism = ' + str(SN_grism[j]))
		convolved_noise_image, image_error, intrinsic_image, grism_spectrum_noise, grism_error, wave_space, \
		wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min, grism_object \
		= make_mock_data(PA_image[j], PA_grism[j], i[j], Va[j], r_t[j], sigma0[j],SN_image[j], SN_grism[j], psf, image_shape= 31)
		#summarize ouputs in one mock_params dictionary
		print('Convolved mock image max pixel: ' + str(jnp.max(convolved_noise_image)))
		mock_params = {'test': test, 'j': j ,'convolved_noise_image': convolved_noise_image, 'image_error': image_error, 'grism_spectrum_noise': grism_spectrum_noise, 'grism_error': grism_error, 'wave_space': wave_space, 'wavelength': wavelength, 'delta_wave_cutoff': delta_wave_cutoff, 'y_factor': y_factor, 'wave_factor': wave_factor, 'index_max': index_max, 'index_min': index_min, 'grism_object': grism_object}
		priors = {'PA': PA_image[j], 'i': i[j], 'Va': Va[j], 'r_t': r_t[j], 'sigma0': sigma0[j]}
		#run fitting
		inf_data, kin_model, grism_object, num_samples  = run_fit(mock_params,priors)
		#post process inference data
		# model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object)	
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
		plotting.plot_pp_cornerplot(inf_data,  kin_model = kin_model, choice = 'real', save_to_folder = str(test), name = str(j) + '_cornerplot_real', PA = PA_grism[j], i = i[j], Va = Va[j], r_t = r_t[j], sigma0 = sigma0[j])

		v_re_med, v_re_truth , kin_model= save_results(config_path, inf_data, test, j, r_t[j], kin_model, grism_object, num_samples)
    	#plot the posteriors of the tuning parameters
		# plotting.plot_tuning_parameters(inf_data, model = 'Disk', save_to_folder = str(test), name = str(j) + '_tuning_parameters', scaling = False, error_scaling = False, errors = False, reg = False)

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
		r_eff = kin_model.r_eff
		inf_data, v_re_16, v_re_med, v_re_84 = utils.add_v_re(inf_data, kin_model, grism_object, num_samples, r_eff)

		for ii_p in params_single:
			res[ii_p + "_q16"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 16)
			res[ii_p + "_q50"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 50)
			res[ii_p + "_q84"][j] = np.percentile(np.concatenate(inf_data['posterior'][ii_p][:]), 84)
		res['v_re'][j] = v_re_truth
	res.write('testing/' + str(test) + '/' + 'results', format='ascii', overwrite=True)



