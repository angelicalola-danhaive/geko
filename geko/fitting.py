__all__ = ["Fit_Numpyro", "run_geko_fit", "run_geko_fit_multi"]

# imports

# importing my own modules
# from . import grism_dev
from . import preprocess as pre
from . import postprocess as post
from . import grism

import os

import jax
import jax.numpy as jnp
import numpy as np


import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS, BarkerMH, SA
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer.initialization import init_to_median, init_to_sample, init_to_uniform, init_to_value, init_to_feasible

import statistics as st
import math

# useful for plotting
from numpyro.infer import Predictive
from jax import random

import arviz as az

import argparse
import corner

from astropy.table import Table

from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import convolve as convolve_astropy

from astropy.cosmology import Planck18 as cosmo



# plotting settings

# setup.configure_plots()


class Fit_Numpyro():
    def __init__(self, obs_map, obs_error, grism_object, kin_model, inference_data, parametric, config=None):
        """ Class to fit model to data

                        Parameters
                        ----------
                        obs_map : array-like
                            Observed 2D grism spectrum
                        obs_error : array-like
                            Error map for observations
                        grism_object : Grism
                            Grism dispersion object
                        kin_model : KinModels
                            Kinematic model object
                        inference_data : arviz.InferenceData or None
                            Previous inference results
                        parametric : bool
                            Whether to use parametric morphology
                        config : FitConfiguration, optional
                            Configuration object with priors and settings

                        Attributes
                        ----------
        """

        self.obs_map = obs_map
        self.obs_error = obs_error

        self.mask = (jnp.where(obs_map/obs_error < 5.0, 0, 1)).astype(bool)

        self.grism_object = grism_object
        self.kin_model = kin_model
        self.inference_data = inference_data
        self.parametric = parametric

        # Config is now applied in run_geko_fit before Fit_Numpyro initialization
        # Store config reference if provided (for potential future use)
        self.config = config


    def run_inference(self, num_samples=None, num_warmup=None, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=None, max_tree_depth=None, num_chains=None, init_vals = None, mask=None):
        """
        Run MCMC inference using the NUTS sampler.

        Performs Bayesian parameter estimation using Numpyro's No-U-Turn Sampler (NUTS).
        MCMC settings are taken from the config object if provided, otherwise defaults are used.

        Parameters
        ----------
        num_samples : int, optional
            Number of MCMC samples to draw (default: from config or 1000)
        num_warmup : int, optional
            Number of warmup/burn-in samples (default: from config or 500)
        high_res : bool, optional
            Use high resolution model (default: False)
        median : bool, optional
            Initialize from median values (default: True)
        step_size : float, optional
            NUTS step size (default: 1 or from config)
        adapt_step_size : bool, optional
            Adapt step size during warmup (default: True)
        target_accept_prob : float, optional
            Target acceptance probability (default: from config or 0.8)
        max_tree_depth : int, optional
            Maximum NUTS tree depth (default: from config or 10)
        num_chains : int, optional
            Number of MCMC chains (default: from config or 4)
        init_vals : dict, optional
            Initial parameter values (default: None)
        mask : jax.numpy.ndarray, optional
            Custom mask to use (default: None, will create mask using photutils segmentation)

        Notes
        -----
        Results are stored in self.mcmc and printed to console.
        If mask is not provided, creates a source mask automatically using photutils segmentation.
        """

        # Always use config for defaults - create default config if none provided
        from .config import MCMCSettings
        if self.config is not None:
            mcmc_config = self.config.mcmc
        else:
            # Use default MCMCSettings if no config provided
            mcmc_config = MCMCSettings()

        # Use config values if parameters not explicitly provided
        if num_samples is None:
            num_samples = mcmc_config.num_samples
        if num_warmup is None:
            num_warmup = mcmc_config.num_warmup
        if target_accept_prob is None:
            target_accept_prob = mcmc_config.target_accept_prob
        if max_tree_depth is None:
            max_tree_depth = mcmc_config.max_tree_depth
        if num_chains is None:
            num_chains = mcmc_config.num_chains
        if mcmc_config.step_size is not None:
            step_size = mcmc_config.step_size

        if self.parametric:
            inference_model = self.kin_model.inference_model_parametric
        else:
            inference_model = self.kin_model.inference_model
        self.nuts_kernel = NUTS(inference_model,  step_size=step_size, adapt_step_size=adapt_step_size, init_strategy=init_to_median(num_samples=2000),
                                target_accept_prob=target_accept_prob, find_heuristic_step_size=True, max_tree_depth=max_tree_depth, dense_mass=False, adapt_mass_matrix=True) 
        
        print(f'MCMC settings: {num_chains} chains, {num_samples} samples, {num_warmup} warmup, max_tree_depth={max_tree_depth}, target_accept={target_accept_prob}')
        print('step size: ', step_size)
        print('warmup: ', num_warmup)
        print('samples: ', num_samples)


        self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples,
                         num_warmup=num_warmup, num_chains=num_chains)
        self.rng_key = random.PRNGKey(100)

        # Use provided mask if given, otherwise create one using photutils segmentation
        if mask is None:
            new_mask = self.create_mask()
            print('Using photutils segmentation mask')
        else:
            new_mask = mask
            print('Using provided custom mask')

        self.mcmc.run(self.rng_key, grism_object = self.grism_object, obs_map = self.obs_map, obs_error = self.obs_error, mask =new_mask) #, extra_fields=("potential_energy", "accept_prob"))

        print('done')

        self.mcmc.print_summary()

    def run_inference_multi(self, observations, masks=None, num_samples=None, num_warmup=None,
                           num_chains=None, step_size=1, adapt_step_size=True,
                           target_accept_prob=None, max_tree_depth=None):
        """
        Run MCMC inference with multiple grism observations.

        Parameters
        ----------
        observations : list of GrismObservation
            List of observations to fit jointly
        masks : list of jax.numpy.ndarray, optional
            Source masks for each observation (default: None, will auto-generate)
            If provided, must be same length as observations
        num_samples : int, optional
            Number of MCMC samples (default: from config or 1000)
        num_warmup : int, optional
            Number of warmup iterations (default: from config or 500)
        num_chains : int, optional
            Number of MCMC chains (default: from config or 4)
        step_size : float, optional
            NUTS step size (default: 1 or from config)
        adapt_step_size : bool, optional
            Adapt step size during warmup (default: True)
        target_accept_prob : float, optional
            Target acceptance probability (default: from config or 0.8)
        max_tree_depth : int, optional
            Maximum NUTS tree depth (default: from config or 10)

        Notes
        -----
        Results are stored in self.mcmc and printed to console.
        If masks not provided, they will be auto-generated for each observation.
        """
        from .grism import GrismObservation

        if not isinstance(observations, list):
            observations = [observations]

        # Get MCMC settings from config
        from .config import MCMCSettings
        if self.config is not None:
            mcmc_config = self.config.mcmc
        else:
            mcmc_config = MCMCSettings()

        # Use config values if parameters not explicitly provided
        if num_samples is None:
            num_samples = mcmc_config.num_samples
        if num_warmup is None:
            num_warmup = mcmc_config.num_warmup
        if target_accept_prob is None:
            target_accept_prob = mcmc_config.target_accept_prob
        if max_tree_depth is None:
            max_tree_depth = mcmc_config.max_tree_depth
        if num_chains is None:
            num_chains = mcmc_config.num_chains
        if mcmc_config.step_size is not None:
            step_size = mcmc_config.step_size

        # Auto-generate masks if not provided
        if masks is None:
            print("\nAuto-generating masks for each observation...")
            masks = []
            for obs in observations:
                # Temporarily set obs_map/obs_error to generate mask
                old_obs_map = self.obs_map
                old_obs_error = self.obs_error
                self.obs_map = obs.obs_map
                self.obs_error = obs.obs_error
                mask = self.create_mask()
                masks.append(mask)
                # Restore
                self.obs_map = old_obs_map
                self.obs_error = old_obs_error

        # Check parametric mode
        if not self.parametric:
            raise ValueError("Multi-observation fitting only supported for parametric mode")

        # Create inference model that captures observations and masks
        def inference_model():
            return self.kin_model.inference_model_parametric_multi(observations, masks)

        # Setup NUTS kernel
        self.nuts_kernel = NUTS(
            inference_model,
            step_size=step_size,
            adapt_step_size=adapt_step_size,
            init_strategy=init_to_median(num_samples=2000),
            target_accept_prob=target_accept_prob,
            find_heuristic_step_size=True,
            max_tree_depth=max_tree_depth,
            dense_mass=False,
            adapt_mass_matrix=True
        )

        print(f'\n{"="*60}')
        print(f'Multi-Observation MCMC Fitting')
        print(f'{"="*60}')
        print(f'Number of observations: {len(observations)}')
        for obs in observations:
            print(f'  - {obs}')
        print(f'MCMC settings: {num_chains} chains, {num_samples} samples, {num_warmup} warmup')
        print(f'               max_tree_depth={max_tree_depth}, target_accept={target_accept_prob}')
        print(f'{"="*60}\n')

        # Setup and run MCMC
        self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples,
                        num_warmup=num_warmup, num_chains=num_chains)
        self.rng_key = random.PRNGKey(100)

        # Run MCMC (observations and masks are captured in the closure)
        self.mcmc.run(self.rng_key)

        print('\nMCMC complete!')
        self.mcmc.print_summary()

    def run_inference_ns(self, num_samples=2000, num_warmup=2000, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=0.8, max_tree_depth=10, num_chains=5, init_vals = None):
        """
        Run nested sampling inference (experimental).

        Uses Numpyro's NestedSampler for Bayesian inference. This is an
        alternative to MCMC that may be more efficient for certain problems.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw (default: 2000)
        num_warmup : int, optional
            Number of warmup iterations (default: 2000)
        high_res : bool, optional
            Use high resolution model (default: False)
        median : bool, optional
            Initialize from median (default: True)
        step_size : float, optional
            Step size (default: 1)
        adapt_step_size : bool, optional
            Adapt step size (default: True)
        target_accept_prob : float, optional
            Target acceptance probability (default: 0.8)
        max_tree_depth : int, optional
            Maximum tree depth (default: 10)
        num_chains : int, optional
            Number of chains (default: 5)
        init_vals : dict, optional
            Initial values (default: None)

        Notes
        -----
        This method is experimental and may not work for all models.
        Results are stored in self.ns.
        """

        constructor_kwargs = {"max_samples": 1000}
        self.ns = NestedSampler(model = self.kin_model.inference_model, constructor_kwargs= constructor_kwargs)

        self.ns.run(random.PRNGKey(0), grism_object = self.grism_object, obs_map = self.obs_map, obs_error = self.obs_error, mask =self.mask)
        self.ns.print_summary()

        ns_samples = self.ns.get_samples(random.PRNGKey(1), num_samples=num_samples)
        print(ns_samples)

        print('done')

    def diverging_parameters(self, chain_number, divergence_number):
        """
        Extract parameter values at divergent MCMC transitions.

        Retrieves the parameter values for a specific divergent transition,
        useful for diagnosing MCMC sampling problems.

        Parameters
        ----------
        chain_number : int
            Which MCMC chain to examine
        divergence_number : int
            Index of the divergent transition within the chain

        Returns
        -------
        fluxes : jax.numpy.ndarray
            Flux values at divergent point
        PA : float
            Position angle at divergent point
        i : float
            Inclination at divergent point
        Va : float
            Asymptotic velocity at divergent point
        r_t : float
            Turnover radius at divergent point
        sigma0 : float
            Velocity dispersion at divergent point

        Notes
        -----
        Requires self.inference_data to be set with MCMC results.
        Used for debugging pathological MCMC behavior.
        """
        divergences = az.convert_to_dataset(
            self.inference_data, group="sample_stats").diverging.transpose("chain", "draw")
        PA_div = self.inference_data.posterior['PA'][chain_number,
            :][divergences[chain_number, :]]
        i_div = self.inference_data.posterior['i'][chain_number,
            :][divergences[chain_number, :]]
        Va_div = self.inference_data.posterior['Va'][chain_number,
            :][divergences[chain_number, :]]
        sigma0_div = self.inference_data.posterior['sigma0'][chain_number,
            :][divergences[chain_number, :]]
        r_t_div = self.inference_data.posterior['r_t'][chain_number,
            :][divergences[chain_number, :]]
        fluxes_div = self.inference_data.posterior['fluxes'][chain_number,
            :][divergences[chain_number, :]]

        return jnp.array(fluxes_div[divergence_number].data), jnp.array(PA_div[divergence_number]), jnp.array(i_div[divergence_number]), jnp.array(Va_div[divergence_number]), jnp.array(r_t_div[divergence_number]), jnp.array(sigma0_div[divergence_number])


    def create_mask(self):
        """
        Create a source mask for the grism spectrum using photutils segmentation.

        Detects sources in the observed grism spectrum and creates a mask
        that isolates the central source, excluding contaminating sources.

        Returns
        -------
        numpy.ndarray
            Binary mask array (1 for source, 0 for background) with same
            shape as obs_map

        Notes
        -----
        Uses photutils source detection with:
        - 2D Gaussian smoothing kernel (sigma=3.0, size=5x5)
        - Background estimation on 15x15 pixel boxes
        - Detection threshold of 5-sigma above background
        - Minimum source size of 10 pixels
        The mask selects only the source at the central pixel position.
        """
        sigma_rms = jnp.minimum((self.obs_map/self.obs_error).max(),5)
        im_conv = convolve_astropy(self.obs_map, make_2dgaussian_kernel(3.0, size=5))

        bkg = Background2D(self.obs_map, (15, 15), filter_size=(5, 5), exclude_percentile=99.0)


        segment_map = detect_sources(im_conv, sigma_rms*np.abs(bkg.background_median), npixels=10)

        main_label = segment_map.data[int(0.5*self.obs_map.shape[0]), int(0.5*self.obs_map.shape[1])]

        # construct mask
        mask = segment_map.data
        new_mask = np.zeros_like(mask)
        new_mask[mask == main_label] = 1.0
        return new_mask
# -----------------------------------------------------------running the inference-----------------------------------------------------------------------------------

def run_geko_fit(output, master_cat, line, parametric, save_runs_path, num_chains, num_warmup, num_samples,
                 source_id, field, grism_filter='F444W', delta_wave_cutoff=0.02, factor=5, wave_factor=9,
                 model_name='Disk', config=None,
                 manual_psf_name=None, manual_theta_rot=None, manual_pysersic_file=None,
                 manual_grism_file=None):
    """
    Run geko fitting without requiring a YAML config file.

    Parameters
    ----------
    output : str
        Name of output subfolder
    master_cat : str
        Path to master catalog file
    line : int
        Emission line wavelength in Angstroms (e.g., 6562 for H-alpha)
    parametric : bool
        Use parametric morphology fitting
    save_runs_path : str
        Base directory containing data files
    num_chains : int
        Number of MCMC chains
    num_warmup : int
        Number of warmup iterations
    num_samples : int
        Number of MCMC samples
    source_id : int
        Source ID number
    field : str
        Field name: 'GOODS-N', 'GOODS-N-CONGRESS', 'GOODS-S-FRESCO', or 'manual'
    grism_filter : str, optional
        Grism filter name (default: 'F444W')
    delta_wave_cutoff : float, optional
        Wavelength bin size cutoff in microns (default: 0.005)
    factor : int, optional
        Spatial oversampling factor (default: 5)
    wave_factor : int, optional
        Wavelength oversampling factor (default: 10)
    model_name : str, optional
        Kinematic model type (default: 'Disk')
    config : FitConfiguration, optional
        Optional configuration object to override priors
    manual_psf_name : str, optional
        PSF filename (required if field='manual').
        Should be in save_runs_path/psfs/ directory
    manual_theta_rot : float, optional
        Rotation angle in degrees (required if field='manual')
        Angle to rotate morphology to match grism orientation
    manual_pysersic_file : str, optional
        PySersic results filename (required if field='manual' and parametric=True)
        Should be in save_runs_path/morph_fits/ directory
    manual_grism_file : str, optional
        Grism spectrum filename (required if field='manual')
        Should be in save_runs_path/output/ directory

    Returns
    -------
    arviz.InferenceData
        MCMC inference results
    """

    # ----------------------------------------------------------preprocessing the data------------------------------------------------------------------------
    z_spec, wavelength, wave_space, obs_map, obs_error, kin_model, grism_object,\
    delta_wave = pre.run_full_preprocessing(output, master_cat, line, save_runs_path=save_runs_path,
                                            source_id=source_id, field=field, grism_filter=grism_filter,
                                            delta_wave_cutoff=delta_wave_cutoff, factor=factor,
                                            wave_factor=wave_factor, model_name=model_name,
                                            manual_psf_name=manual_psf_name, manual_grism_file=manual_grism_file)

    if parametric:
        # Try to load PySersic morphology file
        pysersic_available = False

        # Handle manual field option
        if field == 'manual':
            if manual_pysersic_file is None:
                if config is None:
                    raise ValueError(
                        "When field='manual', you must provide either:\n"
                        "  1. manual_pysersic_file parameter, or\n"
                        "  2. Complete morphological priors via the config parameter"
                    )
                print(f"WARNING: No manual_pysersic_file provided for field='manual'. Will use config priors.")
            else:
                try:
                    pysersic_summary = Table.read(save_runs_path + 'morph_fits/' + manual_pysersic_file, format='ascii')
                    pysersic_available = True
                except:
                    if config is None:
                        raise FileNotFoundError(
                            f"PySersic file not found at {save_runs_path}morph_fits/{manual_pysersic_file}\n"
                            f"To run without PySersic, you must provide morphological priors via the config parameter."
                        )
                    print(f"WARNING: PySersic file not found at {save_runs_path}morph_fits/{manual_pysersic_file}. Will use config priors.")
        else:
            # Standard field-based loading
            try:
                pysersic_summary = Table.read(save_runs_path + 'morph_fits/summary_' + str(source_id) + '_image_F150W_svi.cat', format='ascii')
                pysersic_available = True
            except:
                try:
                    pysersic_summary = Table.read(save_runs_path + 'morph_fits/summary_' + str(source_id) + '_image_F182M_svi.cat', format='ascii')
                    pysersic_available = True
                except:
                    # No PySersic file found
                    if config is None:
                        raise FileNotFoundError(
                            f"No PySersic morphology file found for source {source_id} at:\n"
                            f"  {save_runs_path}morph_fits/summary_{source_id}_image_F150W_svi.cat\n"
                            f"  {save_runs_path}morph_fits/summary_{source_id}_image_F182M_svi.cat\n\n"
                            f"To run without PySersic, you must provide morphological priors via the config parameter.\n"
                            f"See the demo notebook for examples of setting custom priors."
                        )
                    print(f"WARNING: No PySersic file found for source {source_id}. Will use config priors.")

        # Load emission line flux from master catalog
        master_cat_table = Table.read(master_cat, format="ascii")
        log_int_flux = master_cat_table['fit_flux_cgs'][master_cat_table['ID'] == source_id][0] #in log(ergs/s/cm2)
        int_flux = 10**log_int_flux #in ergs/s/cm2
        log_int_flux_err = master_cat_table['fit_flux_cgs_e'][master_cat_table['ID'] == source_id][0] #in log(ergs/s/cm2)
        int_flux_err_high = 10**(log_int_flux + log_int_flux_err) - 10**log_int_flux #in ergs/s/cm2
        int_flux_err_low = 10**log_int_flux - 10**(log_int_flux - log_int_flux_err) #in ergs/s/cm2
        int_flux_err = np.mean([int_flux_err_high, int_flux_err_low]) #in ergs/s/cm2

        # Set field-specific rotation to match JADES to grism survey
        if field == 'manual':
            if manual_theta_rot is None:
                raise ValueError("manual_theta_rot must be provided when field='manual'")
            theta_rot = jnp.radians(manual_theta_rot)
        elif field == 'GOODS-S-FRESCO':
            theta_rot = jnp.radians(0)
        elif field == 'GOODS-N': #fresco
            theta_rot = jnp.radians(230.5098)
        elif field == 'GOODS-N-CONGRESS':
            theta_rot = jnp.radians(228.22379)
        else:
            raise ValueError("Field not recognized. Please check the field name.")

        # Assemble model components from config
        from .morph_models import MORPH_REGISTRY
        from .rotation_models import COMPONENT_REGISTRY, CompositeRotationCurve
        from .config import FitConfiguration

        cfg = config if config is not None else FitConfiguration()

        # Set morphology model
        kin_model.galaxy_model.morph_model = MORPH_REGISTRY[cfg.morphology_model]()

        # Build rotation components (compute kpc_per_px for mass-based components)
        pixel_scale_arcsec = 0.0629
        kpc_per_px = (pixel_scale_arcsec
                      * cosmo.angular_diameter_distance(z_spec).to('kpc').value
                      * np.pi / (180.0 * 3600.0))
        components = []
        for comp_name in cfg.rotation_components:
            cls = COMPONENT_REGISTRY[comp_name]
            comp = cls()
            if cls.NEEDS_PHYSICAL_SCALE:
                comp.kpc_per_px = kpc_per_px
            components.append(comp)
        kin_model.galaxy_model.rot_model = CompositeRotationCurve(components)

        # Set priors: PySersic first (if available), then config overrides always applied on top
        if pysersic_available:
            kin_model.galaxy_model.set_parametric_priors(
                pysersic_summary, [int_flux, int_flux_err], z_spec, wavelength,
                delta_wave, theta_rot=theta_rot, shape=obs_map.shape[0]
            )
        else:
            print("\nUsing config priors (no PySersic file available)...")
        kin_model.galaxy_model.apply_config_overrides(cfg)
    else:
        #raise non-parametric fitting not implemented error
        raise ValueError("Non-parametric fitting is not implemented yet. Please set --parametric to True to use the parametric fitting.")

    # ----------------------------------------------------------running the inference------------------------------------------------------------------------

    run_fit = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object, kin_model=kin_model, inference_data=None, parametric=parametric, config=config)

    rng_key = random.PRNGKey(4)
    inference_model = run_fit.kin_model.inference_model_parametric
    num_samples_prior = np.max([1000, num_samples])
    prior_predictive = Predictive(inference_model, num_samples=num_samples_prior)

    prior = prior_predictive(rng_key, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error)

    # Run inference - config parameters will be used automatically if provided
    run_fit.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
                              median=True, adapt_step_size=True, num_chains=num_chains)

    inf_data = az.from_numpyro(run_fit.mcmc, prior=prior)

    # Save results
    inf_data.to_netcdf(save_runs_path + output + '/' + str(source_id) + '_output')

    # Process results
    v_re_16, v_re_med, v_re_84, kin_model, inf_data = post.process_results(
        output, master_cat, line, parametric=parametric, ID=source_id, save_runs_path=save_runs_path,
        field=field, grism_filter=grism_filter, delta_wave_cutoff=delta_wave_cutoff,
        factor=factor, wave_factor=wave_factor, model_name=model_name,
        manual_psf_name=manual_psf_name, manual_grism_file=manual_grism_file)

    return inf_data


def run_geko_fit_multi(observations_config, output, master_cat, line, parametric, save_runs_path,
                       num_chains, num_warmup, num_samples, source_id, field, grism_filter='F444W',
                       delta_wave_cutoff=0.02, factor=5, wave_factor=9, model_name='Disk', config=None,
                       manual_psf_name=None, manual_pysersic_file=None, step_size=0.1,
                       adapt_step_size=True, target_accept_prob=0.8):
    """
    Run geko multi-observation fitting for multiple grism observations.

    This function jointly fits multiple grism observations (e.g., at different position angles
    or dispersion directions) by sharing galaxy parameters across observations while computing
    separate likelihoods for each.

    Parameters
    ----------
    observations_config : list of dict
        List of observation configurations. Each dict should contain:
            - 'grism_file': str, grism spectrum filename
            - 'theta_rot': float, rotation angle in degrees (for morphology alignment)
            - 'dispersion': str, 'R' (row) or 'C' (column) dispersion direction
            - 'name': str, optional name for this observation (default: 'obsN')
        Example:
            [
                {'grism_file': 'spec_2d_PA0.fits', 'theta_rot': 0.0, 'dispersion': 'R', 'name': 'PA0'},
                {'grism_file': 'spec_2d_PA90.fits', 'theta_rot': 90.0, 'dispersion': 'R', 'name': 'PA90'}
            ]
    output : str
        Name of output subfolder
    master_cat : str
        Path to master catalog file
    line : int
        Emission line wavelength in Angstroms (e.g., 6562 for H-alpha)
    parametric : bool
        Use parametric morphology fitting
    save_runs_path : str
        Base directory containing data files
    num_chains : int
        Number of MCMC chains
    num_warmup : int
        Number of warmup iterations
    num_samples : int
        Number of MCMC samples
    source_id : int
        Source ID number
    field : str
        Field name: 'GOODS-N', 'GOODS-N-CONGRESS', 'GOODS-S-FRESCO', or 'manual'
    grism_filter : str, optional
        Grism filter name (default: 'F444W')
    delta_wave_cutoff : float, optional
        Wavelength bin size cutoff in microns (default: 0.005)
    factor : int, optional
        Spatial oversampling factor (default: 5)
    wave_factor : int, optional
        Wavelength oversampling factor (default: 10)
    model_name : str, optional
        Kinematic model type (default: 'Disk')
    config : FitConfiguration, optional
        Optional configuration object to override priors
    manual_psf_name : str, optional
        PSF filename (required if field='manual')
    manual_pysersic_file : str, optional
        PySersic results filename (required if field='manual' and parametric=True)
    step_size : float, optional
        MCMC step size (default: 0.1)
    adapt_step_size : bool, optional
        Adapt step size during warmup (default: True)
    target_accept_prob : float, optional
        Target acceptance probability (default: 0.8)

    Returns
    -------
    arviz.InferenceData
        MCMC inference results with posterior samples
    dict
        Dictionary mapping observation names to their model predictions, including:
            - 'model_map': median model prediction
            - 'model_map_16': 16th percentile prediction
            - 'model_map_84': 84th percentile prediction
    """

    # Run preprocessing on the first observation to initialize model
    first_obs_file = observations_config[0]['grism_file']

    print("Running preprocessing on first observation...")
    z_spec, wavelength, wave_space, obs_map_ref, obs_error_ref, kin_model, grism_object_ref, delta_wave = \
        pre.run_full_preprocessing(
            output=output,
            master_cat=master_cat,
            line=line,
            save_runs_path=save_runs_path,
            source_id=source_id,
            field=field,
            grism_filter=grism_filter,
            delta_wave_cutoff=delta_wave_cutoff,
            factor=factor,
            wave_factor=wave_factor,
            model_name=model_name,
            manual_psf_name=manual_psf_name,
            manual_grism_file=first_obs_file
        )

    # Set up parametric priors if needed
    if parametric:
        # Load PySersic morphology file
        pysersic_available = False

        if field == 'manual':
            if manual_pysersic_file is None:
                if config is None:
                    raise ValueError(
                        "When field='manual', you must provide either:\n"
                        "  1. manual_pysersic_file parameter, or\n"
                        "  2. Complete morphological priors via the config parameter"
                    )
                print(f"WARNING: No manual_pysersic_file provided for field='manual'. Will use config priors.")
            else:
                try:
                    pysersic_summary = Table.read(save_runs_path + 'morph_fits/' + manual_pysersic_file, format='ascii')
                    pysersic_available = True
                except:
                    if config is None:
                        raise FileNotFoundError(
                            f"PySersic file not found at {save_runs_path}morph_fits/{manual_pysersic_file}\n"
                            f"To run without PySersic, you must provide morphological priors via the config parameter."
                        )
                    print(f"WARNING: PySersic file not found. Will use config priors.")
        else:
            # Standard field-based loading
            try:
                pysersic_summary = Table.read(save_runs_path + 'morph_fits/summary_' + str(source_id) + '_image_F150W_svi.cat', format='ascii')
                pysersic_available = True
            except:
                try:
                    pysersic_summary = Table.read(save_runs_path + 'morph_fits/summary_' + str(source_id) + '_image_F182M_svi.cat', format='ascii')
                    pysersic_available = True
                except:
                    if config is None:
                        raise FileNotFoundError(
                            f"No PySersic morphology file found for source {source_id}\n"
                            f"To run without PySersic, you must provide morphological priors via the config parameter."
                        )
                    print(f"WARNING: No PySersic file found for source {source_id}. Will use config priors.")

        # Load emission line flux from master catalog
        master_cat_table = Table.read(master_cat, format="ascii")
        log_int_flux = master_cat_table['fit_flux_cgs'][master_cat_table['ID'] == source_id][0]
        int_flux = 10**log_int_flux
        log_int_flux_err = master_cat_table['fit_flux_cgs_e'][master_cat_table['ID'] == source_id][0]
        int_flux_err_high = 10**(log_int_flux + log_int_flux_err) - 10**log_int_flux
        int_flux_err_low = 10**log_int_flux - 10**(log_int_flux - log_int_flux_err)
        int_flux_err = np.mean([int_flux_err_high, int_flux_err_low])

        # For multi-observation fitting, we use the reference frame (first observation)
        # and rotation angles are applied per-observation
        theta_rot_ref = jnp.radians(observations_config[0]['theta_rot'])

        # Assemble model components from config
        from .morph_models import MORPH_REGISTRY
        from .rotation_models import COMPONENT_REGISTRY, CompositeRotationCurve
        from .config import FitConfiguration

        cfg = config if config is not None else FitConfiguration()

        # Set morphology model
        kin_model.galaxy_model.morph_model = MORPH_REGISTRY[cfg.morphology_model]()

        # Build rotation components (compute kpc_per_px for mass-based components)
        pixel_scale_arcsec = 0.0629
        kpc_per_px = (pixel_scale_arcsec
                      * cosmo.angular_diameter_distance(z_spec).to('kpc').value
                      * np.pi / (180.0 * 3600.0))
        components = []
        for comp_name in cfg.rotation_components:
            cls = COMPONENT_REGISTRY[comp_name]
            comp = cls()
            if cls.NEEDS_PHYSICAL_SCALE:
                comp.kpc_per_px = kpc_per_px
            components.append(comp)
        kin_model.galaxy_model.rot_model = CompositeRotationCurve(components)

        # Set priors: PySersic first (if available), then config overrides always applied on top
        if pysersic_available:
            kin_model.galaxy_model.set_parametric_priors(
                pysersic_summary, [int_flux, int_flux_err], z_spec, wavelength,
                delta_wave, theta_rot=theta_rot_ref, shape=obs_map_ref.shape[0]
            )
        else:
            print("\nUsing config priors (no PySersic file available)...")
        kin_model.galaxy_model.apply_config_overrides(cfg)
    else:
        raise ValueError("Non-parametric fitting is not implemented yet. Please set parametric=True.")

    # Create GrismObservation objects for all observations
    print(f"\nCreating {len(observations_config)} GrismObservation objects...")
    observations = []

    for i, obs_config in enumerate(observations_config):
        obs_name = obs_config.get('name', f'obs{i}')
        grism_file = obs_config['grism_file']
        theta_rot = obs_config['theta_rot']
        dispersion = obs_config['dispersion']

        # Load this observation's data
        if i == 0:
            # Use already loaded reference observation
            obs_map = obs_map_ref
            obs_error = obs_error_ref
            grism_obj = grism_object_ref
        else:
            # Load additional observations
            print(f"  Loading observation {obs_name}...")
            _, _, _, obs_map, obs_error, _, grism_obj, _ = pre.run_full_preprocessing(
                output=output,
                master_cat=master_cat,
                line=line,
                save_runs_path=save_runs_path,
                source_id=source_id,
                field=field,
                grism_filter=grism_filter,
                delta_wave_cutoff=delta_wave_cutoff,
                factor=factor,
                wave_factor=wave_factor,
                model_name=model_name,
                manual_psf_name=manual_psf_name,
                manual_grism_file=grism_file
            )

        # Create GrismObservation
        obs = grism.GrismObservation(
            grism=grism_obj,
            obs_map=obs_map,
            obs_error=obs_error,
            theta_rot=theta_rot,
            dispersion=dispersion,
            name=obs_name
        )
        observations.append(obs)
        print(f"  Created: {obs}")

    # Initialize Fit_Numpyro
    print("\nInitializing Fit_Numpyro...")
    run_fit = Fit_Numpyro(
        obs_map=obs_map_ref,  # Use reference observation
        obs_error=obs_error_ref,
        grism_object=grism_object_ref,
        kin_model=kin_model,
        inference_data=None,
        parametric=parametric,
        config=config
    )

    # Generate prior predictive samples
    rng_key = random.PRNGKey(4)
    inference_model = run_fit.kin_model.inference_model_parametric_multi
    num_samples_prior = np.max([1000, num_samples])
    prior_predictive = Predictive(inference_model, num_samples=num_samples_prior)

    print("\nGenerating prior predictive samples...")
    prior = prior_predictive(rng_key, observations=observations, masks=None)

    # Run multi-observation inference
    print(f"\nRunning multi-observation MCMC inference...")
    print(f"  Observations: {len(observations)}")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples: {num_samples}")

    run_fit.run_inference_multi(
        observations=observations,
        masks=None,  # Auto-generate masks
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        step_size=step_size,
        adapt_step_size=adapt_step_size,
        target_accept_prob=target_accept_prob
    )

    # Convert to arviz InferenceData
    inf_data = az.from_numpyro(run_fit.mcmc, prior=prior)

    # Save results
    output_file = save_runs_path + output + '/' + str(source_id) + '_output_multi'
    print(f"\nSaving results to: {output_file}")
    inf_data.to_netcdf(output_file)

    # Compute model results for each observation
    print("\nComputing model results for each observation...")
    inf_data, results = kin_model.compute_model_parametric_multi(inf_data, observations)

    # Process results: compute v_re, v/sigma, save fits, and generate plots
    print("\nPost-processing results...")
    v_re_16, v_re_med, v_re_84, kin_model, inf_data = post.process_results_multi(
        observations, results, output, master_cat, line, parametric, source_id, save_runs_path,
        field, grism_filter, delta_wave_cutoff, factor, wave_factor, model_name,
        manual_psf_name=manual_psf_name, manual_grism_file=observations_config[0]['grism_file'])

    print("\nMulti-observation fitting complete!")
    print(f"  v_re: {v_re_med:.2f} (+{v_re_84-v_re_med:.2f}/-{v_re_med-v_re_16:.2f}) km/s")
    print(f"  Results available for: {list(results.keys())}")

    return inf_data, results


