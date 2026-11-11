"""
Multi-observation mock recovery tests: run_mock.py for several grism observations.

Tests how additional grism observations constrain the galaxy parameters, with the
same mock and fitting conventions as the single-observation run_mock.py:

- each observation is generated with the inference model's own per-observation
  transform (inference_model_parametric_multi): morphology rotated by theta_rot
  (adjust_for_observation), velocity centre rotated with rotate_coords, PA turned
  by +theta_rot, rendered with GalaxyModel.generate_flux_map / velocity_field on the
  subpixel-centred model grid, dispersed with Grism.disperse and binned;
- noise, the grism set-up (run_mock.initialize_grism) and preprocessing
  (pre.run_full_preprocessing with mock_params: error scaling, S/N cap) are those
  of run_mock.py;
- priors are run_mock.run_test's truth-centred defaults, plus FitConfiguration
  overrides; the joint fit is Fit_Numpyro.run_inference_multi, as in
  run_geko_fit_multi;
- derived quantities (v_re = rotation_curve(r_eff), v/sigma with the instrumental
  sigma0 floor, ...) come from geko.postprocess, and the v_re truth is the
  intrinsic rotation_curve(r_eff), as in run_mock.save_results.

Scenarios:
- single_R : one R observation (baseline)
- single_C : one C observation (baseline)
- R+C      : R and C at the same angle (orthogonal dispersions)
- R+R_90   : two R observations 90 deg apart (PA-inclination degeneracy)

Run from the geko repository root:
    python testing/run_mock_multi_obs.py --scenario R+C --PA 30
Outputs go to testing/multi_obs/<scenario>_PA<PA>/.
"""
import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
import numpyro
import matplotlib.pyplot as plt
from astropy.table import Table
from photutils.datasets import make_noise_image
from numpyro.infer import Predictive
from jax import random

from geko import preprocess as pre
from geko import utils
from geko import plotting
from geko.fitting import Fit_Numpyro
from geko.grism import GrismObservation
from geko.models import GalaxyModel
from geko.config import FitConfiguration, MCMCSettings
from geko.postprocess import (compute_derived_posterior, summarize_posterior,
                              DERIVED_QUANTITIES, compute_information_gain)
from geko.param_spec import all_param_specs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mock import make_image, initialize_grism   # same mock conventions as run_mock

if 'gpu' in jax.devices()[0].device_kind.lower():
    print('Using GPU')
    numpyro.set_platform('gpu')
numpyro.enable_validation()
jax.config.update('jax_enable_x64', True)

IMAGE_SHAPE = 31
FACTOR = 5
CENTER = 15.0           # morphological and kinematic centre (as in run_mock.run_test)
LINE = 'H_alpha'

# scenario -> list of (dispersion, theta_rot [deg], observation name)
SCENARIOS = {
    'single_R': [('R', 0.0, 'R_0deg')],
    'single_C': [('C', 0.0, 'C_0deg')],
    'R+C':      [('R', 0.0, 'R_0deg'), ('C', 0.0, 'C_0deg')],
    'R+R_90':   [('R', 0.0, 'R_0deg'), ('R', 90.0, 'R_90deg')],
}


def default_truth(PA=0.0):
    """Truth parameters in the sky (reference) frame; 'rot' holds the rotation-curve parameters."""
    return {
        'PA_image': PA, 'PA_grism': PA, 'i': 60.0, 'sigma0': 50.0,
        'r_eff': (1.676 / 0.4) * 1.0, 'n': 1.0, 'amplitude': 200.0,
        'xc_morph': CENTER, 'yc_morph': CENTER, 'x0_vel': CENTER, 'y0_vel': CENTER,
        'v0': 0.0,
        'rot': {'Va': 200.0, 'r_t': 2.0},
        'SN_image': 50.0, 'SN_grism': 30.0,
    }


def make_observation(truth, gm, psf, pupil, theta_rot, name, j, test, ideal=False):
    """
    One mock grism observation at theta_rot, rendered with the inference model's
    per-observation transform and preprocessed exactly like run_mock.

    gm : GalaxyModel whose morph_model / rot_model render the mock (same model
         classes and rotation curve as the fit).
    Returns the GrismObservation (preprocessed obs_map, obs_error, grism) and the
    run_full_preprocessing outputs of this observation.
    """
    image, image_highres, convolved_image, noise_image, convolved_noise_image = make_image(
        truth['PA_image'], truth['i'], truth['r_eff'], truth['SN_image'], truth['n'], psf,
        IMAGE_SHAPE, xc_morph=truth['xc_morph'], yc_morph=truth['yc_morph'])
    image_error = (jnp.max(image) / truth['SN_image']) * jnp.ones((IMAGE_SHAPE, IMAGE_SHAPE))

    grism_object, wave_space, wavelength, delta_wave_cutoff, y_factor, wave_factor, index_max, index_min = \
        initialize_grism(convolved_image, psf, IMAGE_SHAPE, factor=FACTOR, pupil=pupil)
    grism_object.use_psf = True
    grism_object.use_lsf = False      # mock is always LSF-free (as in run_mock)

    # --- the per-observation transform of inference_model_parametric_multi ---
    center = (IMAGE_SHAPE - 1) / 2
    theta_rad = np.radians(theta_rot)
    morph_sky = {'r_eff': truth['r_eff'], 'n': truth['n'], 'PA_morph': truth['PA_image'],
                 'xc_morph': truth['xc_morph'], 'yc_morph': truth['yc_morph']}
    morph_obs = gm.morph_model.adjust_for_observation(morph_sky, theta_rot, center)
    morph_obs['amplitude'] = truth['amplitude']
    flux_high = gm.generate_flux_map(morph_obs, {'i': truth['i']})

    x0_obs, y0_obs = utils.rotate_coords(truth['x0_vel'], truth['y0_vel'], center, center, theta_rad)
    X, Y = jnp.meshgrid(utils.subpixel_grid(IMAGE_SHAPE, FACTOR, x0_obs),
                        utils.subpixel_grid(IMAGE_SHAPE, FACTOR, y0_obs))
    V = jnp.asarray(gm.velocity_field(X, Y, truth['PA_grism'] + theta_rot, truth['i'],
                                      {**morph_obs, **truth['rot']})) + truth['v0']
    D = truth['sigma0'] * jnp.ones_like(V)

    spectrum = utils.resample(grism_object.disperse(flux_high, V, D), FACTOR, wave_factor)

    # --- noise as in run_mock.make_mock_data ---
    max_grism = jnp.max(spectrum)
    grism_error = (max_grism / truth['SN_grism']) * jnp.ones(spectrum.shape)
    if ideal:
        spectrum_noise = spectrum
    else:
        spectrum_noise = spectrum + make_noise_image(spectrum.shape, distribution='gaussian', mean=0,
                                                     stddev=max_grism / truth['SN_grism'])

    # --- preprocessing as in run_mock.run_fit ---
    mock_params = {'test': test, 'j': j, 'convolved_noise_image': convolved_noise_image,
                   'image_error': image_error, 'grism_spectrum_noise': spectrum_noise,
                   'grism_error': grism_error, 'wave_space': wave_space, 'wavelength': wavelength,
                   'delta_wave_cutoff': delta_wave_cutoff, 'y_factor': y_factor,
                   'wave_factor': wave_factor, 'index_max': index_max, 'index_min': index_min,
                   'grism_object': grism_object, 'PSF': psf}
    pre_out = pre.run_full_preprocessing(None, None, LINE, mock_params=mock_params)
    z_spec, wavelength, wave_space, obs_map, obs_error, kin_model, grism_object, delta_wave = pre_out

    obs = GrismObservation(grism=grism_object, obs_map=obs_map, obs_error=obs_error,
                           theta_rot=theta_rot, dispersion=pupil, name=name)
    return obs, pre_out


def build_fit_config(truth, fit_overrides=None, num_chains=2, num_warmup=500, num_samples=500):
    """run_mock.run_test's truth-centred default priors, with FitConfiguration overrides on top."""
    fit_overrides = fit_overrides or {}
    morph = {
        'PA_morph_mu': truth['PA_image'], 'PA_morph_std': 5.0,
        'r_eff_mu': truth['r_eff'], 'r_eff_std': float(max(3.0, truth['r_eff'])),
        'r_eff_min': 0.0, 'r_eff_max': 15.0,
        'n_mu': float(truth['n']), 'n_std': 1.0, 'n_min': utils.SERSIC_N_MIN, 'n_max': 8.0,
        'amplitude_mu': 200.0, 'amplitude_std': 40.0, 'amplitude_min': 0.0,
        'xc_morph_mu': CENTER, 'xc_morph_std': 1.0,
        'yc_morph_mu': CENTER, 'yc_morph_std': 1.0,
    }
    geom = {
        'PA_mu': truth['PA_grism'], 'PA_std': 10.0,
        'i_mu': float(truth['i']), 'i_std': 5.0,
        'sigma0_min': 0.0, 'sigma0_max': 600.0,
        'v0_mu': 0.0, 'v0_std': 200.0,
    }
    morph.update(fit_overrides.get('morph_prior_overrides', {}))
    geom.update(fit_overrides.get('geom_prior_overrides', {}))
    return FitConfiguration(
        mcmc=MCMCSettings(num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples),
        morph_prior_overrides=morph, geom_prior_overrides=geom,
        rot_prior_overrides=fit_overrides.get('rot_prior_overrides', {}),
        fixed_params=fit_overrides.get('fixed_params', {}),
    )


def run_scenario(scenario, truth, psf, fit_overrides=None, num_chains=2, num_warmup=500,
                 num_samples=500, ideal=False, out_dir='testing/multi_obs'):
    """Make the mock observations of one scenario, fit them jointly and save the results."""
    tag = f"{scenario}_PA{int(round(truth['PA_grism']))}"
    folder = os.path.join(out_dir, tag)
    os.makedirs(folder, exist_ok=True)
    cfg = build_fit_config(truth, fit_overrides, num_chains, num_warmup, num_samples)

    # mocks are rendered by a GalaxyModel with the configured rotation curve (priors
    # don't matter for rendering); the fitted model is the first observation's
    # preprocessing model, as in run_geko_fit_multi
    render_gm = GalaxyModel((IMAGE_SHAPE, IMAGE_SHAPE), FACTOR, rot_model=cfg.build_rot_model())
    observations, kin_model, z_spec, wave_space = [], None, None, None
    for k, (pupil, theta_rot, name) in enumerate(SCENARIOS[scenario]):
        obs, pre_out = make_observation(truth, render_gm, psf, pupil, theta_rot, name, k, tag, ideal)
        observations.append(obs)
        if kin_model is None:
            z_spec, wave_space, kin_model = pre_out[0], pre_out[2], pre_out[5]
    kin_model.galaxy_model.rot_model = cfg.build_rot_model(z_spec)
    kin_model.galaxy_model.apply_config_overrides(cfg)

    fit = Fit_Numpyro(obs_map=observations[0].obs_map, obs_error=observations[0].obs_error,
                      grism_object=observations[0].grism, kin_model=kin_model,
                      inference_data=None, parametric=True, config=cfg)
    masks = [jnp.ones_like(o.obs_map, dtype=bool) for o in observations]   # no mask, as run_mock
    model = kin_model.inference_model_parametric_multi
    prior = Predictive(model, num_samples=max(1000, num_samples))(random.PRNGKey(4),
                                                                  observations=observations, masks=masks)
    fit.run_inference_multi(observations=observations, masks=masks, num_samples=num_samples,
                            num_warmup=num_warmup, num_chains=num_chains, step_size=1.0,
                            adapt_step_size=True, target_accept_prob=cfg.mcmc.target_accept_prob)
    inf_data = az.from_numpyro(fit.mcmc, prior=prior)
    inf_data.to_netcdf(os.path.join(folder, f'{tag}_output'))

    inf_data, model_results = kin_model.compute_model_parametric_multi(inf_data, observations)
    summary = save_results(inf_data, kin_model, z_spec, truth, observations, folder, tag)

    try:
        plotting.plot_disk_summary_multi(
            observations=observations, results=model_results, inf_data=inf_data,
            wave_space=wave_space, x0=kin_model.x0_vel_mean, y0=kin_model.y0_vel_mean, factor=1,
            direct_image_size=kin_model.im_shape[0], save_to_folder=tag, name='summary',
            obs_radius=kin_model.r_eff_mean, ellip=kin_model.ellip_mean,
            theta_Ha=kin_model.PA_morph_mean / (180 / np.pi) + np.pi / 2, n=kin_model.n_mean,
            save_runs_path=out_dir + '/', ID=tag)
    except Exception as e:
        print(f'Warning: summary plot failed: {e}')
    return {'inf_data': inf_data, 'summary': summary, 'observations': observations}


def save_results(inf_data, kin_model, z_spec, truth, observations, folder, tag):
    """Results table (q16/q50/q84 of every sampled + derived quantity, plus truth columns)
    and information gain, as run_mock.save_results; sigma0 floor from the fitted grisms."""
    inf_data = compute_derived_posterior(inf_data, kin_model, z_spec,
                                         grism_objects=[o.grism for o in observations])
    gm = kin_model.galaxy_model
    specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
    sampled = [s.name for s in specs if not s.fixed]
    names = sampled + [dq.name for dq in DERIVED_QUANTITIES]
    summary = summarize_posterior(inf_data, names)
    row = {}
    for name in names:
        if name in summary:
            for q in ('16', '50', '84'):
                row[f'{name}_q{q}'] = summary[name][q]
    truth_cols = {'PA': truth['PA_grism'], 'i': truth['i'], 'sigma0': truth['sigma0'],
                  'r_eff': truth['r_eff'], 'n': truth['n'], **truth['rot'],
                  'v_re': float(gm.rot_model.rotation_curve(jnp.array([truth['r_eff']]), truth['rot'])[0])}
    for k, v in truth_cols.items():
        row[f'{k}_truth'] = v
    Table([row]).write(os.path.join(folder, f'{tag}_results'), format='ascii', overwrite=True)
    try:
        compute_information_gain(inf_data, sampled).write(os.path.join(folder, f'{tag}_info_gain'),
                                                          format='ascii', overwrite=True)
    except Exception as e:
        print(f'Warning: failed to compute information gain: {e}')
    print(f'\n{tag}: parameter, truth, median [16th, 84th]')
    for k, v in truth_cols.items():
        if k in summary:
            s = summary[k]
            print(f"  {k:8s} {v:9.2f}  {s['50']:9.2f}  [{s['16']:.2f}, {s['84']:.2f}]")
    return summary


def plot_scenario_comparison(results, save_path):
    """68% credible-interval width per parameter for each scenario."""
    params = ['PA', 'i', 'Va', 'r_t', 'sigma0', 'v_re']
    scenarios = list(results)
    x = np.arange(len(params))
    width = 0.8 / len(scenarios)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, sc in enumerate(scenarios):
        s = results[sc]['summary']
        ax.bar(x + k * width, [s[p]['84'] - s[p]['16'] if p in s else np.nan for p in params],
               width, label=sc)
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(params)
    ax.set_ylabel('68% credible interval width')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-observation mock recovery tests')
    parser.add_argument('--scenario', default='all', choices=list(SCENARIOS) + ['all'])
    parser.add_argument('--PA', type=float, default=0.0, help='PA of morphology and kinematics (deg)')
    parser.add_argument('--num_chains', type=int, default=2)
    parser.add_argument('--num_warmup', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--ideal', action='store_true', help='no noise in the mock grism spectra')
    args = parser.parse_args()

    # Gaussian PSF at detector resolution (as before); replace with an STPSF/mpsf as needed
    yy, xx = np.mgrid[:15, :15] - 7
    psf = np.exp(-(xx**2 + yy**2) / (2 * 2.0**2))
    psf /= psf.sum()

    truth = default_truth(args.PA)
    # Fit overrides, same keys as FIT_OVERRIDES in examples/RunGekoTests.py
    fit_overrides = {'fixed_params': {}, 'morph_prior_overrides': {},
                     'geom_prior_overrides': {}, 'rot_prior_overrides': {}}

    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    results = {sc: run_scenario(sc, truth, psf, fit_overrides, args.num_chains, args.num_warmup,
                                args.num_samples, ideal=args.ideal) for sc in scenarios}
    if len(results) > 1:
        plot_scenario_comparison(results, f'testing/multi_obs/comparison_PA{int(round(args.PA))}.png')
