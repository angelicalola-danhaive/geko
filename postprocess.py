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

from matplotlib import pyplot as plt

import jax.numpy as jnp
import numpyro

import arviz as az

import yaml



def process_results(output):
    """
        Main function that automatically post-processes the inference data and saves all of the relevant plots
        Returns the main data products so that data can be analyzed separately
    """

    #pre-process the galaxy data
    direct, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism, x0_grism, \
    num_samples, num_warmup, step_size, target_accept_prob,  \
    wave_space, delta_wave, index_min, index_max, factor  = pre.run_full_preprocessing(output)

    #load inference data
    inf_data = az.InferenceData.from_netcdf('fitting_results/' + output + '/'+ 'output')
    data = fit.Fit_Numpyro(obs_map = obs_map, obs_error = obs_error, grism_object = grism_object, kin_model = kin_model, inference_data = inf_data )
    model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object)

    #define the wave_space
    len_wave = int((wave_space[len(wave_space)-1] - wave_space[0])/(delta_wave))
    wave_space = jnp.linspace(wave_space[0], wave_space[len(wave_space)-1], len_wave+1)
    wave_space = wave_space[index_min:index_max+1]#

    #plot the grism model
    plotting.plot_grism(obs_map, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'truth_map')

    plotting.plot_grism(model_map, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'model_map', limits= obs_map)

    plotting.plot_grism_residual(obs_map, model_map, obs_error, y0_grism, direct.shape[0], wave_space, save_to_folder=output, name = 'residual_map')

    #plot the flux model
    plotting.plot_image(direct,x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'truth_flux')

    plotting.plot_image(fluxes_mean, x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'model_flux', limits= direct)

    fluxes_errors = direct

    plotting.plot_image_residual(direct, fluxes_mean, fluxes_errors, x0_grism,  y0_grism, direct.shape[0], save_to_folder = output, name = 'residual_flux')

    #define grid for velocity plots
    x = jnp.linspace(0 - x0_grism, direct.shape[1]-1 -x0_grism, direct.shape[1]*factor)
    y = jnp.linspace(0 - y0_grism, direct.shape[0]-1 -y0_grism, direct.shape[0]*factor)
    X,Y = jnp.meshgrid(x,y)
    
    #plot the velocity field
    if model_name == 'Disk':

        plotting.plot_velocity_profile(fluxes_mean, kin_model.x0, kin_model.y0, direct.shape[0], model_velocities, save_to_folder=output, name = 'velocity_profile')
        plotting.plot_velocity_map(model_velocities, kin_model.mask, kin_model.x0, kin_model.y0, direct.shape[0], save_to_folder=output, name = 'velocity_map')
    
    #plot the posteriors of the kinematic parameters
    plotting.plot_pp_cornerplot(inf_data,  kin_model = kin_model, choice = 'real', save_to_folder = output, name = 'cornerplot_real')

    #plot the posteriors of the tuning parameters
    plotting.plot_tuning_parameters(inf_data, model = model_name, save_to_folder = output, name = 'tuning_parameters_real')

    #plot the full summary of the results
    kin_model.plot_summary(obs_map, obs_error, inf_data, wave_space, save_to_folder = output, name = 'summary')