"""
Put all of the necessary post-processing functions here

	Written by A L Danhaive: ald66@cam.ac.uk
"""

# imports
import preprocess as pre
import fitting as fit
import models
import grism
import plotting
import utils

from matplotlib import pyplot as plt

import jax.numpy as jnp
from jax import image
import numpyro

import arviz as az

import yaml

import numpy as np

from jax.scipy.signal import convolve



def process_results(output, master_cat, line):
    """
        Main function that automatically post-processes the inference data and saves all of the relevant plots
        Returns the main data products so that data can be analyzed separately
    """

    #pre-process the galaxy data
    direct,direct_error, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism, x0_grism, \
    num_samples, num_warmup, step_size, target_accept_prob,  \
    wave_space, delta_wave, index_min, index_max, factor  = pre.run_full_preprocessing(output, master_cat, line)

    #load inference data
    inf_data = az.InferenceData.from_netcdf('fitting_results/' + output + '/'+ 'output')
    data = fit.Fit_Numpyro(obs_map = obs_map, obs_error = obs_error, grism_object = grism_object, kin_model = kin_model, inference_data = inf_data )
    model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object)
    #define the wave_space
    len_wave = int((wave_space[len(wave_space)-1] - wave_space[0])/(delta_wave))
    wave_space = jnp.linspace(wave_space[0], wave_space[len(wave_space)-1], len_wave+1)
    wave_space = wave_space[index_min:index_max+1]
    # print(jnp.sum(jnp.abs(convolve(fluxes_mean, kin_model.Laplace_kernel, mode='same'))))
    # plt.imshow(jnp.abs(convolve(fluxes_mean, kin_model.Laplace_kernel, mode='same')), origin = 'lower')
    # plt.colorbar()
    # plt.show()
    # print(jnp.sqrt(jnp.sum(jnp.power(kin_model.std,2))))
    #plot the grism model
    plotting.plot_grism(obs_map, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'truth_map')

    plotting.plot_grism(model_map, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'model_map', limits= obs_map)

    plotting.plot_grism_residual(obs_map, model_map, obs_error, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'residual_map')

    #plot the flux model
    plotting.plot_image(direct,x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'truth_flux')

    plotting.plot_image(fluxes_mean, x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'model_flux') #, limits= direct)
    # Laplace_kernel = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # fluxes_mean_lap = convolve(fluxes_mean, Laplace_kernel, mode = 'same')
    # plotting.plot_image(jnp.abs(fluxes_mean_lap),x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'truth_flux')


    #save the fluxes image and posteriors in fits file
    utils.save_fits_image(fluxes_mean, kin_model.masked_indices, inf_data,'fitting_results/' + output + '/' + 'model_flux' + '.fits')

    # fluxes_errors = direct
    # Laplace_kernel = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # plotting.plot_image_residual(direct, fluxes_mean, fluxes_errors, x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'residual_flux')
    # laplace_fluxes = jnp.abs(convolve(direct, Laplace_kernel, mode='same'))
    # laplace_fluxes_err = jnp.maximum(0.000001, kin_model.flux_bounds[0]* laplace_fluxes)
    # plotting.plot_image(laplace_fluxes, x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'laplace_flux') #, limits= direct)
    # sum_reg = jnp.sum(jnp.abs(laplace_fluxes))/jnp.sum(jnp.abs(direct))
    # print('sum_reg: ', sum_reg)
    # # threshold_grism = 0.2*obs_map.max()
    # error_reg = jnp.sqrt(jnp.sum(obs_error**2))/obs_error.sum()
    # sum_reg_norm = sum_reg/obs_map.sum() #*model_map.shape[0]*model_map.shape[1]

    # print('direct sum: ', np.sum(direct[kin_model.masked_indices]))
    # print('fluxes_mean sum: ', np.sum(fluxes_mean[kin_model.masked_indices]))

    # print('obs_map sum: ', np.sum(obs_map))
    # # error_reg_norm = error_reg*model_map.shape[0]*model_map.shape[1]/model_map.sum()
    # laplace_fluxes_r = jnp.reshape(sum_reg_norm, (1,))
    # print(laplace_fluxes_r, error_reg)

    #define grid for velocity plots
    x = jnp.linspace(0 - x0_grism, direct.shape[1]-1 -x0_grism, direct.shape[1]*factor)
    y = jnp.linspace(0 - y0_grism, direct.shape[0]-1 -y0_grism, direct.shape[0]*factor)
    X,Y = jnp.meshgrid(x,y)
    
    #plot the velocity field
    if model_name == 'Disk':

        plotting.plot_velocity_profile(fluxes_mean, kin_model.x0, kin_model.y0, direct.shape[0], model_velocities, save_to_folder=output, name = 'velocity_profile')
        plotting.plot_velocity_map(model_velocities, kin_model.mask, kin_model.x0, kin_model.y0, direct.shape[0], save_to_folder=output, name = 'velocity_map')
        plotting.plot_velocity_map(model_dispersions, kin_model.mask, kin_model.x0, kin_model.y0, direct.shape[0], save_to_folder=output, name = 'dispersion_map')
    
    #plot the posteriors of the kinematic parameters
    plotting.plot_pp_cornerplot(inf_data,  kin_model = kin_model, choice = 'real', save_to_folder = output, name = 'cornerplot_real')
    #plot the posteriors of the tuning parameters
    plotting.plot_tuning_parameters(inf_data, model = model_name, save_to_folder = output, name = 'tuning_parameters_real')
    #plot some flux posteriors from the middle ish of the cube
    index_bright = np.argmax(kin_model.flux_prior[kin_model.mask ==1]) #inf_data.posterior['fluxes'].shape[2]//2
    print('index_mid: ', index_bright)
    print(inf_data.prior['fluxes'][:,:,index_bright].mean(),inf_data.prior['fluxes'][:,:,index_bright].std(),  inf_data.prior['fluxes'][:,:,index_bright].min(), inf_data.prior['fluxes'][:,:,index_bright].max())
    plotting.plot_flux_corner(inf_data, index_min= index_bright - 2, index_max = index_bright + 2, save_to_folder = output, name = 'flux_corner')
    #plot the full summary of the results
    kin_model.plot_summary(obs_map, obs_error, inf_data, wave_space, save_to_folder = output, name = 'summary')
    print('Fluxes scaling: ', kin_model.fluxes_scaling_mean)



# def match_to_IFU(F1, V1, D1, F2, V2, D2, full_PSF, FWHM_PSF, R_LSF, wavelength, IFU='kmos'):
# 	# if there is no second component than you should input zero arrays in the F2,V2,D2
# 	# create a 3D cube with the model best fit, basically with f[i,j,k]=F[i,j]*gaussian(k)
# 		# make the gaussian with stats.norm.pdf(velocity_space, v[i,j],sigma_v[i,j])
# 	velocity_space = jnp.linspace(-1000, 1000, 2001)
# 	broadcast_velocity_space = np.broadcast_to(
# 	    velocity_space[:, np.newaxis, np.newaxis], (velocity_space.size, F1.shape[0], F1.shape[0]))
# 	if F2 is not None:
# 		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1) + \
# 		                         F2*stats.norm.pdf(broadcast_velocity_space, V2, D2)
# 	else:
# 		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1)

# 	print('cube created')
# 	# if no full PSF given, use the FWHM to create a 2D gaussian kernel
# 	if full_PSF is None:
# 		FWHM_PSF_pixels = FWHM_PSF/0.0629
# 		sigma_PSF = FWHM_PSF_pixels/(2*np.sqrt(2*np.log(2)))
# 		full_PSF = np.array(Gaussian2DKernel(sigma_PSF))
# 	print('PSF created')
# 	# create the LSF kernel - assuming a constant one for now since only emission around the same wavelength
# 	# compute the std for the velocities from the spectral resolution
# 	sigma_LSF = wavelength/(2.355*R_LSF)
# 	sigma_LSF_v = (c/1000)*sigma_LSF/wavelength
# 	LSF = np.array(Gaussian1DKernel(sigma_LSF_v))
# 	print('LSF created')
# 	# create a 3D kernel with 2D PSF and 1D LSF
# 	full_kernel = np.array(full_PSF) * np.broadcast_to(np.array(LSF)[:, np.newaxis, np.newaxis], (np.array(
# 	    LSF).size, np.array(full_PSF).shape[0], np.array(full_PSF).shape[0]))
# 	print('kernel created')
# 	# convolve the cube with the kernel
# 	print('convolving cube')
# 	convolved_cube = signal.fftconvolve(cub     e, full_kernel, mode='same')
# 	print('convolution done')
# 	# from the convolved cube, obtain the flux, velocity and dispersion maps
# 	F_kmos = np.sum(convolved_cube, axis=0)
# 	# fit a gaussian to the convolved cube to obtain the velocity and dispersion maps
# 	V_kmos = np.zeros_like(F_kmos)
# 	D_kmos = np.zeros_like(F_kmos)

# 	def gauss(x, A, mu, sigma):
# 		return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)*A

# 	for i in range(V_kmos.shape[0]):
# 		for j in range(V_kmos.shape[1]):
# 			popt, pcov = curve_fit(gauss, velocity_space, convolved_cube[:, i, j], p0=[
# 			                       F_kmos[i, j], V1[i, j], D1[i, j]])
# 			V_kmos[i, j] = popt[1]
# 			D_kmos[i, j] = popt[2]
# 	# popt, pcov = curve_fit(gauss, velocity_space, convolved_cube, p0=[F_kmos,V1, D1])
# 	# V_kmos = popt[1]
# 	# D_kmos = popt[2]

# 	# finally resample to IFU pixel scale
# 	IFU_scale_arcseconds = 0.1799
# 	NC_LW_scale_arcseconds = 0.0629
# 	# rescale the maps to the IFU pixel scale

# 	if IFU == 'kmos':
# 		IFU_scale_pixels = int(IFU_scale_arcseconds/NC_LW_scale_arcseconds)
# 		# blocks = F_kmos.reshape((int(F_kmos.shape[0]/(IFU_scale_pixels)), IFU_scale_pixels, int(F_kmos.shape[1]/IFU_scale_pixels), IFU_scale_pixels))
# 		# F_kmos = jnp.sum(blocks, axis=(1,3))
# 		F_kmos = image.resize(F_kmos, (int(
# 		    F_kmos.shape[0]/IFU_scale_pixels), int(F_kmos.shape[1]/IFU_scale_pixels)), method='nearest')
# 		F_kmos *= IFU_scale_pixels**2
# 		V_kmos = image.resize(V_kmos, (int(
# 		    V_kmos.shape[0]/IFU_scale_pixels), int(V_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')
# 		D_kmos = image.resize(D_kmos, (int(
# 		    D_kmos.shape[0]/IFU_scale_pixels), int(D_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')

# 		print('rescaling done')
# 	return F_kmos, V_kmos, D_kmos