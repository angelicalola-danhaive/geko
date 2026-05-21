"""
Multi-observation mock fitting tests

Tests how additional grism observations constrain:
1. Velocity field parameters (PA, i, Va, r_t, sigma0)
2. Flux model (parametric vs non-parametric)
3. Model degeneracies and parameter correlations

Scenarios to test:
- single_R: Single R observation (baseline)
- R+C: R and C at same angle (orthogonal dispersions - maximal constraint)
- R+R_90: Two R observations at 90° (tests PA-inclination degeneracy breaking)
"""

from geko import grism
from geko import preprocess as pre
from geko import postprocess as post
from geko import utils
from geko import plotting
from geko import models
from geko import config as geko_config

from geko.fitting import Fit_Numpyro
from geko.grism import GrismObservation
from numpyro.infer import Predictive
from jax import random

import os
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

from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import convolve as convolve_astropy
from photutils.datasets import make_noise_image

from jax.scipy.signal import convolve
import numpyro
from astropy.cosmology import Planck18 as cosmo

if 'gpu' in jax.devices()[0].device_kind.lower():
    print('Using GPU')
    numpyro.set_platform('gpu')
numpyro.set_host_device_count(2)
numpyro.enable_validation()
jax.config.update('jax_enable_x64', True)

XLA_FLAGS='--xla_gpu_deterministic_ops=true'


def make_grism_observation(morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
                           SN_grism, psf, image_shape, pupil='R', theta_rot=0.0, name=None):
    """
    Create a single mock grism observation (R or C dispersion).

    Regenerates morphology for each observation to match the inference model,
    which applies theta_rot rotation to both morphology and kinematics.

    Parameters
    ----------
    morphology_params : dict
        Morphology parameters: 'amplitude', 'r_eff', 'n', 'ellip', 'xc', 'yc'
    PA_image : float
        Position angle of image morphology (degrees)
    PA_grism : float
        Position angle for velocity field (degrees)
    i, Va, r_t, sigma0 : float
        Kinematic parameters
    SN_grism : float
        Signal-to-noise ratio for grism
    psf : array
        Point spread function
    image_shape : int
        Size of image
    pupil : str
        'R' for row or 'C' for column dispersion
    theta_rot : float
        Rotation angle for this observation (degrees)
    name : str
        Name for this observation

    Returns
    -------
    grism_obs : GrismObservation
        Observation object containing grism, data, and metadata
    grism_spectrum : array
        Mock grism spectrum (for visualization)
    """
    # Create wave space
    wave_factor = 9
    delta_wave = 0.001
    wavelength = 4.5
    delta_wave_cutoff = 0.02
    wave_first = 4.0
    wave_space = jnp.linspace(wave_first, 5.0, int(1/delta_wave)+1)

    wave_min = wavelength - delta_wave_cutoff
    wave_max = wavelength + delta_wave_cutoff
    index_min = round((wave_min - wave_first)/delta_wave)
    index_max = round((wave_max - wave_first)/delta_wave)

    # Create oversampled wave space
    half_step = (delta_wave / wave_factor)*(wave_factor//2)
    wave_space_oversampled = np.arange(wave_space[0]- half_step, wave_space[-1] + delta_wave + half_step,
                                       delta_wave / wave_factor)

    # Initialize grism object
    factor = 5
    x0_grism = y0_grism = image_shape//2

    grism_object = grism.Grism(
        image_shape*factor, 0.0629/factor,
        icenter=y0_grism, jcenter=x0_grism,
        wavelength=wavelength, wave_space=wave_space_oversampled,
        index_min=(index_min)*wave_factor, index_max=(index_max+1)*wave_factor,
        grism_filter='F444W', grism_module='A', grism_pupil=pupil, PSF=psf
    )

    # Generate morphology for this observation
    # Extract morphology parameters
    amplitude = morphology_params['amplitude']
    r_eff = morphology_params['r_eff']
    n = morphology_params['n']
    ellip = morphology_params['ellip']
    xc = morphology_params.get('xc', image_shape // 2)
    yc = morphology_params.get('yc', image_shape // 2)

    # Build coordinate grids
    x = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
    y = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape)
    x_mesh, y_mesh = jnp.meshgrid(x, y)
    x_grid = image.resize(x_mesh, (image_shape*factor, image_shape*factor), method='linear')
    y_grid = image.resize(y_mesh, (image_shape*factor, image_shape*factor), method='linear')

    # Generate flux map (morphology) using utils.compute_adaptive_sersic_profile
    # Convert PA to the convention used by sersic (90 - PA)
    Ie = utils.flux_to_Ie(amplitude, n, r_eff, ellip)
    mock_image_highres = utils.compute_adaptive_sersic_profile(
        x_grid, y_grid, Ie/(factor)**2, r_eff, n,
        xc - image_shape//2, yc - image_shape//2, ellip,
        (90 - PA_image) * np.pi / 180
    )

    # Make velocity field
    kin_model = models.GrismFitter()
    V = kin_model.v(x_grid, y_grid, PA_grism, i, Va, r_t)
    D = sigma0*jnp.ones_like(V)

    # Make grism spectrum
    grism_spectrum = grism_object.disperse(mock_image_highres, V, D)

    # Resample to grism resolution
    grism_spectrum = utils.resample(grism_spectrum, factor, wave_factor)

    # Add noise
    max_grism = jnp.max(grism_spectrum)
    grism_noise = make_noise_image((grism_spectrum.shape[0], grism_spectrum.shape[1]),
                                   distribution='gaussian', mean=0, stddev=max_grism/SN_grism)
    grism_spectrum_noise = grism_spectrum + grism_noise
    grism_error = (max_grism/SN_grism)*jnp.ones((grism_spectrum.shape[0], grism_spectrum.shape[1]))

    # Create GrismObservation object
    if name is None:
        name = f"{pupil}_{theta_rot:.0f}deg"

    grism_obs = GrismObservation(
        grism=grism_object,
        obs_map=grism_spectrum_noise,
        obs_error=grism_error,
        theta_rot=theta_rot,
        dispersion=pupil,
        name=name
    )

    return grism_obs, grism_spectrum_noise, grism_error, wave_space, index_min, index_max


def make_multi_observation_data(PA_image, PA_grism, i, Va, r_t, sigma0,
                                SN_image, SN_grism, n, psf,
                                image_shape=31, scenario='R+C'):
    """
    Create multiple mock observations for testing.

    Parameters
    ----------
    scenario : str
        'single_R' : Single R observation (baseline)
        'single_C' : Single C observation (baseline)
        'R+C' : R and C at same angle (orthogonal dispersions)
        'R+R_90' : Two R observations 90° apart

    Returns
    -------
    observations : list of GrismObservation
        List of observation objects
    mock_image : array
        True underlying image
    """
    # Make the underlying image (for visualization reference only)
    from run_mock import make_image

    image, image_highres, convolved_image, noise_image, convolved_noise_image = \
        make_image(PA_image, i, r_t, SN_image, n, psf, image_shape)

    # Create morphology parameters dict (will regenerate morphology for each observation)
    axis_ratio = utils.compute_axis_ratio(i, q0=0.2)
    ellip = 1 - axis_ratio
    r_eff = (1.676/0.4) * 1  # Same as in run_mock.py
    amplitude = 200.0  # Same as in run_mock.py

    morphology_params = {
        'amplitude': amplitude,
        'r_eff': r_eff,
        'n': n,
        'ellip': ellip,
        'xc': image_shape // 2,
        'yc': image_shape // 2
    }

    observations = []
    spectra = []

    if scenario == 'single_R':
        # Baseline: single R observation
        obs, spec, err, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='R', theta_rot=0.0, name='R_0deg'
        )
        observations.append(obs)
        spectra.append(spec)

    elif scenario == 'single_C':
        # Baseline: single C observation
        obs, spec, err, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='C', theta_rot=0.0, name='C_0deg'
        )
        observations.append(obs)
        spectra.append(spec)

    elif scenario == 'R+C':
        # R and C at same angle - orthogonal dispersions (maximal constraint)
        obs_R, spec_R, err_R, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='R', theta_rot=0.0, name='R_0deg'
        )
        obs_C, spec_C, err_C, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='C', theta_rot=0.0, name='C_0deg'
        )
        observations.extend([obs_R, obs_C])
        spectra.extend([spec_R, spec_C])

    elif scenario == 'R+R_90':
        # Two R observations 90° apart - tests PA-inclination degeneracy
        obs_R1, spec_R1, err_R1, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image, PA_grism, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='R', theta_rot=0.0, name='R_0deg'
        )
        obs_R2, spec_R2, err_R2, wave_space, idx_min, idx_max = make_grism_observation(
            morphology_params, PA_image-90, PA_grism-90, i, Va, r_t, sigma0,
            SN_grism, psf, image_shape, pupil='R', theta_rot=90.0, name='R_90deg'
        )
        observations.extend([obs_R1, obs_R2])
        spectra.extend([spec_R1, spec_R2])

    return observations, spectra, convolved_image, wave_space, idx_min, idx_max


def run_multi_obs_fit(observations, parametric=False, num_samples=1000, num_warmup=500, PA_prior_mean=45.0):
    """
    Run multi-observation fitting.

    Parameters
    ----------
    observations : list of GrismObservation
        Observations to fit jointly
    parametric : bool
        Use parametric flux model
    num_samples, num_warmup : int
        MCMC parameters
    PA_prior_mean : float
        Mean of PA prior in degrees (default: 45.0)

    Returns
    -------
    inf_data : arviz.InferenceData
        Posterior samples
    kin_model : KinModels
        Kinematic model
    """
    # Initialize kinematic model (GrismFitter has multi-obs inference methods)
    kin_model = models.GrismFitter()

    # Use first observation to initialize Fit_Numpyro
    obs_first = observations[0]

    # Initialize the disk model with bounds and priors
    grism = obs_first.grism
    im_shape_val = grism.im_shape // grism.factor
    im_shape = (im_shape_val, im_shape_val)  # Needs to be a tuple
    factor = grism.factor
    wave_factor = 9  # From the test setup
    x0 = y0 = im_shape_val // 2
    x0_vel = y0_vel = im_shape_val // 2

    kin_model.set_bounds(im_shape, factor, wave_factor, x0, x0_vel, y0, y0_vel)

    # Set parametric priors using new FitConfiguration
    test_config = geko_config.FitConfiguration(
        morph_prior_overrides={
            'PA_morph_mu': PA_prior_mean, 'PA_morph_std': 5.0,
            'r_eff_mu': 2.0, 'r_eff_std': 2.0, 'r_eff_min': 0.0, 'r_eff_max': 15.0,
            'n_mu': 1.0, 'n_std': 1.0, 'n_min': 0.5, 'n_max': 8.0,
            'amplitude_mu': 200.0, 'amplitude_std': 40.0, 'amplitude_min': 10.0, 'amplitude_max': 1000.0,
            'xc_morph_mu': 15.0, 'xc_morph_std': 1.0,
            'yc_morph_mu': 15.0, 'yc_morph_std': 1.0,
        },
        geom_prior_overrides={
            'PA_mu': PA_prior_mean, 'PA_std': 10.0,
            'i_mu': 60.0, 'i_std': 12.0,
            'sigma0_min': 0.0, 'sigma0_max': 600.0,
        },
        rot_prior_overrides={
            'Va_min': -1000.0, 'Va_max': 1000.0,
        },
    )
    kin_model.galaxy_model.apply_config_overrides(test_config)

    # Create fitting object
    fit = Fit_Numpyro(
        obs_map=obs_first.obs_map,
        obs_error=obs_first.obs_error,
        grism_object=obs_first.grism,
        kin_model=kin_model,
        inference_data=None,
        parametric=parametric
    )

    # Generate prior predictive samples
    print("\nGenerating prior predictive samples...")
    rng_key = random.PRNGKey(4)
    inference_model = kin_model.inference_model_parametric_multi
    num_samples_prior = np.max([1000, num_samples])
    prior_predictive = Predictive(inference_model, num_samples=num_samples_prior)

    prior = prior_predictive(rng_key, observations=observations, masks=None)

    # Run multi-observation inference
    print(f"\nRunning multi-observation fit with {len(observations)} observations:")
    for obs in observations:
        print(f"  - {obs.name} ({obs.dispersion} dispersion, theta_rot={obs.theta_rot:.1f}°)")

    fit.run_inference_multi(
        observations=observations,
        masks=None,  # Auto-generate masks
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=2,
        step_size=1,
        adapt_step_size=True,
        target_accept_prob=0.8,
        max_tree_depth=10
    )

    # Convert to arviz with prior samples
    inf_data = az.from_numpyro(fit.mcmc, prior=prior)

    return inf_data, kin_model, fit


def compare_scenarios(PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n, psf,
                     scenarios=['single_R', 'single_C', 'R+C', 'R+R_90'], parametric=False,
                     num_samples=500, num_warmup=250):
    """
    Compare different multi-observation scenarios.

    Tests how additional observations constrain the model:
    1. Posterior widths (parameter uncertainties)
    2. Correlations between parameters
    3. Flux model freedom (if non-parametric)

    Returns
    -------
    results : dict
        Dictionary with results for each scenario
    """
    results = {}

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Testing scenario: {scenario}")
        print(f"{'='*70}")

        # Generate mock data
        observations, spectra, true_image, wave_space, idx_min, idx_max = \
            make_multi_observation_data(
                PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n, psf,
                scenario=scenario
            )

        # Run fitting
        inf_data, kin_model, fit = run_multi_obs_fit(
            observations, parametric=parametric,
            num_samples=num_samples, num_warmup=num_warmup,
            PA_prior_mean=PA_grism
        )

        # Compute statistics
        params = ['PA', 'i', 'Va', 'r_t', 'sigma0']
        stats = {}

        for param in params:
            posterior = np.concatenate(inf_data.posterior[param].values)
            stats[param] = {
                'mean': np.mean(posterior),
                'std': np.std(posterior),
                'q16': np.percentile(posterior, 16),
                'q50': np.percentile(posterior, 50),
                'q84': np.percentile(posterior, 84),
            }

        # Compute model predictions using fit's method
        print(f"\nGenerating model predictions for {scenario}...")
        inf_data_with_models, model_results = kin_model.compute_model_parametric_multi(inf_data, observations)

        # Add v_re and v_sigma to inference data (needed for plotting)
        import xarray as xr

        # Compute v_re for posterior
        num_chains = len(inf_data_with_models.posterior.chain)
        num_samples_post = len(inf_data_with_models.posterior.draw)

        v_re_samples = []
        for chain in range(num_chains):
            for sample in range(num_samples_post):
                Va_val = float(inf_data_with_models.posterior['Va'][chain, sample])
                r_t_val = float(inf_data_with_models.posterior['r_t'][chain, sample])
                v_re_val = Va_val * (2/np.pi) * np.arctan(2/r_t_val)
                v_re_samples.append(v_re_val)

        v_re_array = np.array(v_re_samples).reshape(num_chains, num_samples_post)
        inf_data_with_models.posterior['v_re'] = xr.DataArray(v_re_array, dims=('chain', 'draw'))

        # Compute v_sigma for posterior
        v_sigma_array = v_re_array / inf_data_with_models.posterior['sigma0'].values
        inf_data_with_models.posterior['v_sigma'] = xr.DataArray(v_sigma_array, dims=('chain', 'draw'))

        # Compute v_re for prior (if it exists)
        if hasattr(inf_data_with_models, 'prior') and inf_data_with_models.prior is not None:
            num_prior_samples = len(inf_data_with_models.prior.draw)
            v_re_prior = []
            for sample in range(num_prior_samples):
                Va_val = float(inf_data_with_models.prior['Va'][0, sample])
                r_t_val = float(inf_data_with_models.prior['r_t'][0, sample])
                v_re_val = Va_val * (2/np.pi) * np.arctan(2/r_t_val)
                v_re_prior.append(v_re_val)

            v_re_prior_array = np.array(v_re_prior).reshape(1, num_prior_samples)
            inf_data_with_models.prior['v_re'] = xr.DataArray(v_re_prior_array, dims=('chain', 'draw'))

        # Store results
        results[scenario] = {
            'inf_data': inf_data_with_models,
            'kin_model': kin_model,
            'observations': observations,
            'spectra': spectra,
            'stats': stats,
            'true_image': true_image,
            'model_results': model_results,  # Model predictions for plotting
        }

        # Generate detailed diagnostic plots
        print(f"\nGenerating diagnostic plots for {scenario}...")
        # Create filename suffix with PA
        pa_suffix = f"_PA{int(PA_grism)}"
        os.makedirs(f'geko/testing/{scenario}', exist_ok=True)

        # Plot multi-observation summary
        try:
            # Create unique ID with PA
            scenario_id = f"{scenario}{pa_suffix}"

            plotting.plot_disk_summary_multi(
                observations=observations,
                results=model_results,
                inf_data=inf_data_with_models,
                wave_space=wave_space,
                x0=kin_model.x0_vel_mean,
                y0=kin_model.y0_vel_mean,
                factor=1,
                direct_image_size=kin_model.im_shape[0],
                save_to_folder=scenario,
                name='summary',  # Use 'summary' so it respects save_runs_path
                obs_radius=kin_model.r_eff_mean,
                ellip=kin_model.ellip_mean,
                theta_Ha=kin_model.PA_morph_mean/(180/np.pi) + np.pi/2,
                n=kin_model.n_mean,
                save_runs_path='geko/testing/',
                ID=scenario_id  # Use scenario_id with PA suffix
            )
            print(f"  ✓ Saved summary plot: geko/testing/{scenario}/{scenario_id}_summary_multi.png")
        except Exception as e:
            print(f"  ✗ Error creating summary plot: {e}")
            import traceback
            traceback.print_exc()

        # Plot corner plot with truth values
        try:
            import corner
            # Define corner args with correct var_names for our case
            CORNER_KWARGS = plotting.define_corner_args(
                divergences=False,
                var_names=['PA', 'Va', 'i', 'r_t', 'sigma0', 'v_re'],
                labels=[r'$PA$', r'$V_a$', r'$i$', r'$r_t$', r'$\sigma_0$', r'$V_{re}$']
            )

            v_re_true = Va * (2/np.pi) * np.arctan(2/r_t)
            truths = {'PA': PA_grism, 'Va': Va, 'i': i, 'r_t': r_t, 'sigma0': sigma0, 'v_re': v_re_true}

            fig = corner.corner(inf_data_with_models, group='posterior',
                              truths=truths, truth_color='crimson',
                              color='blue', **CORNER_KWARGS)

            # Add prior for comparison if available
            if hasattr(inf_data_with_models, 'prior') and inf_data_with_models.prior is not None:
                CORNER_KWARGS_prior = plotting.define_corner_args(
                    divergences=False, fill_contours=False,
                    plot_contours=False, show_titles=False,
                    var_names=['PA', 'Va', 'i', 'r_t', 'sigma0', 'v_re'],
                    labels=[r'$PA$', r'$V_a$', r'$i$', r'$r_t$', r'$\sigma_0$', r'$V_{re}$']
                )
                fig = corner.corner(inf_data_with_models, group='prior',
                                  fig=fig, color='lightgray', **CORNER_KWARGS_prior)

            plt.savefig(f'geko/testing/{scenario}/{scenario_id}_corner.png', dpi=300)
            plt.close()
            print(f"  ✓ Saved corner plot: geko/testing/{scenario}/{scenario_id}_corner.png")
        except Exception as e:
            print(f"  ✗ Error creating corner plot: {e}")
            import traceback
            traceback.print_exc()

        # Print summary
        print(f"\nParameter uncertainties for {scenario}:")
        print(f"{'Parameter':<10} {'Truth':<10} {'Median':<10} {'Std':<10} {'68% CI width':<15}")
        print("-"*60)
        true_vals = {'PA': PA_grism, 'i': i, 'Va': Va, 'r_t': r_t, 'sigma0': sigma0}
        for param in params:
            truth = true_vals[param]
            median = stats[param]['q50']
            std = stats[param]['std']
            ci_width = stats[param]['q84'] - stats[param]['q16']
            print(f"{param:<10} {truth:<10.2f} {median:<10.2f} {std:<10.3f} {ci_width:<15.3f}")

    return results


def plot_scenario_comparison(results, save_path=None):
    """
    Create comparison plots showing how constraints improve with more observations.

    Plots:
    1. Parameter uncertainty comparison (bar chart)
    2. Corner plots for each scenario
    3. Correlation matrices
    """
    scenarios = list(results.keys())
    params = ['PA', 'i', 'Va', 'r_t', 'sigma0']

    # Plot 1: Uncertainty comparison
    fig, axes = plt.subplots(1, len(params), figsize=(15, 4))

    for i, param in enumerate(params):
        uncertainties = [results[s]['stats'][param]['std'] for s in scenarios]
        axes[i].bar(range(len(scenarios)), uncertainties)
        axes[i].set_xticks(range(len(scenarios)))
        axes[i].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[i].set_ylabel('Posterior Std')
        axes[i].set_title(param)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_uncertainty_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: CI width comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(params))
    width = 0.8 / len(scenarios)

    for i, scenario in enumerate(scenarios):
        ci_widths = [results[scenario]['stats'][p]['q84'] - results[scenario]['stats'][p]['q16']
                     for p in params]
        ax.bar(x + i*width, ci_widths, width, label=scenario)

    ax.set_xlabel('Parameter')
    ax.set_ylabel('68% Credible Interval Width')
    ax.set_title('Parameter Constraint Comparison Across Scenarios')
    ax.set_xticks(x + width * (len(scenarios)-1) / 2)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_CI_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Test multi-observation fitting')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['single_R', 'single_C', 'R+C', 'R+R_90', 'all'],
                       help='Which scenario to test')
    parser.add_argument('--parametric', action='store_true',
                       help='Use parametric flux model')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=500,
                       help='Number of warmup iterations')
    parser.add_argument('--PA', type=float, default=0.0,
                       help='Position angle in degrees (default: 0)')

    args = parser.parse_args()

    # Load PSF (create simple Gaussian for now)
    psf_size = 15
    center = psf_size // 2
    y, x = np.meshgrid(np.arange(psf_size), np.arange(psf_size), indexing='ij')
    psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2))
    psf = psf / np.sum(psf)

    # Set true parameters
    PA_image = args.PA
    PA_grism = args.PA  # Same as PA_image
    i = 60.0
    Va = 200.0
    r_t = 2.0
    sigma0 = 50.0
    SN_image = 50.0
    SN_grism = 30.0
    n = 1.0

    print("\n" + "="*70)
    print("Multi-observation Mock Testing")
    print("="*70)
    print(f"Position Angle (PA): {PA_grism:.1f}°")
    print(f"Inclination: {i:.1f}°")
    print(f"Va: {Va:.1f} km/s")
    print(f"r_t: {r_t:.1f} pixels")
    print(f"sigma0: {sigma0:.1f} km/s")
    print("="*70)

    # Determine scenarios to run
    if args.scenario == 'all':
        scenarios = ['single_R', 'single_C', 'R+C', 'R+R_90']
    else:
        scenarios = [args.scenario]

    # Run comparison
    results = compare_scenarios(
        PA_image, PA_grism, i, Va, r_t, sigma0, SN_image, SN_grism, n, psf,
        scenarios=scenarios,
        parametric=args.parametric,
        num_samples=args.num_samples,
        num_warmup=args.num_warmup
    )

    # Plot comparison with PA in filename
    plot_scenario_comparison(results, save_path=f'geko/testing/multi_obs_comparison_PA{int(PA_grism)}')

    print("\n" + "="*70)
    print("Multi-observation testing complete!")
    print("="*70)
