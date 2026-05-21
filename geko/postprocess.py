"""
Post-processing functions for geko grism fits.

	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['process_results', 'process_results_multi',
           'compute_derived_posterior', 'DERIVED_QUANTITIES']

from dataclasses import dataclass
from typing import Callable

from . import preprocess as pre
from . import fitting as fit
from . import utils

from matplotlib import pyplot as plt

import jax.numpy as jnp
from jax import image
import numpyro

import arviz as az

import numpy as np

from jax.scipy.signal import convolve

import argparse

from astropy.cosmology import Planck18 as cosmo

import xarray as xr

from astropy.table import Table

import corner

# ============================================================================
# Derived-quantity registry
# ============================================================================

_G_PC_MSUN_KMS2 = 4.3009172706e-3   # pc M_sun^-1 (km/s)^2
_METERS_TO_PC   = 3.086e16
_PIXEL_SCALE    = 0.063              # NIRCam pixel scale in arcsec


def _r_eff_to_pc(r_eff_px, z_spec, pixel_scale):
	DA = cosmo.angular_diameter_distance(z_spec).to('m')
	return np.deg2rad(r_eff_px * pixel_scale / 3600) * DA.value / _METERS_TO_PC


@dataclass
class DerivedQuantity:
	name:             str
	label:            str       # axis label for corner plots
	compute:          Callable  # compute(posterior_dataset, context) -> DataArray
	include_in_corner: bool = True   # False for intermediate quantities


# Sampled parameters to include alongside derived quantities in the corner plot.
_CORNER_SAMPLED_PARAMS = ['sigma0']

# Ordered by dependency: each entry may use results from earlier entries.
DERIVED_QUANTITIES = [
	DerivedQuantity(
		name='r_eff_pc',
		label=r'$r_e$ [pc]',
		compute=lambda post, ctx: _r_eff_to_pc(
			post['r_eff'], ctx['z_spec'], ctx['pixel_scale']),
		include_in_corner=False,   # intermediate used by M_dyn
	),
	# v_sigma: divide only by samples where sigma0 > floor so the ratio
	# doesn't diverge. Unresolved samples (sigma0 <= floor) become NaN and
	# are skipped in quantile computation. No pile-up at the floor.
	DerivedQuantity(
		name='v_sigma',
		label=r'$v_{re}/\sigma_0$',
		compute=lambda post, ctx: (
			post['v_re'] / post['sigma0'].where(post['sigma0'] > ctx['sigma0_floor'])
		),
	),
	DerivedQuantity(
		name='v_circ',
		label=r'$v_{circ}$ [km/s]',
		compute=lambda post, ctx: np.sqrt(
			post['v_re']**2 + 3.35 * post['sigma0']**2),
	),
	DerivedQuantity(
		name='M_dyn',
		label=r'$\log M_{dyn}$ [$M_\odot$]',
		compute=lambda post, ctx: np.log10(
			1.8 * post['v_circ']**2 * post['r_eff_pc'] / _G_PC_MSUN_KMS2),
	),
]

# ============================================================================
# Pipeline helpers
# ============================================================================

def compute_derived_posterior(inf_data, kin_model, z_spec,
                              sigma0_floor=20.0, pixel_scale=_PIXEL_SCALE):
	"""Add all derived quantities to inf_data.posterior.

	Parameters
	----------
	sigma0_floor : float
		Samples with sigma0 <= this value (km/s) are excluded from the
		v_sigma computation to avoid ratio divergence. Does not affect
		the sigma0 posterior itself.
	pixel_scale : float
		Detector pixel scale in arcsec (default NIRCam 0.063 arcsec/px).
	"""
	utils.add_v_re(inf_data, kin_model, grism_object=None,
	               num_samples=inf_data.posterior['sigma0'].shape[1])

	context = {'z_spec': z_spec, 'pixel_scale': pixel_scale,
	           'sigma0_floor': sigma0_floor}
	for dq in DERIVED_QUANTITIES:
		inf_data.posterior[dq.name] = dq.compute(inf_data.posterior, context)

	return inf_data


def summarize_posterior(inf_data, names):
	"""Return {name: {'16': val, '50': val, '84': val}} for each name in posterior."""
	summary = {}
	for name in names:
		if name not in inf_data.posterior:
			continue
		post = inf_data.posterior[name]
		summary[name] = {
			'16': float(post.quantile(0.16, skipna=True)),
			'50': float(post.median(skipna=True)),
			'84': float(post.quantile(0.84, skipna=True)),
		}
	return summary


def build_results_table(ID, kin_model, summary):
	"""Build an astropy Table from a posterior summary dict.

	Columns are driven by the model's actual parameter specs plus the
	derived-quantity registry — no hardcoded parameter names.
	"""
	from .param_spec import all_param_specs

	gm = kin_model.galaxy_model
	specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
	sampled_names = [s.name for s in specs if not s.fixed]
	derived_names = [dq.name for dq in DERIVED_QUANTITIES]
	# ellip is not in param_specs (it's derived from i inside compute_model)
	all_names = sampled_names + ['ellip'] + derived_names

	row = {'ID': ID}
	for name in all_names:
		if name in summary:
			row[f'{name}_16'] = summary[name]['16']
			row[f'{name}_50'] = summary[name]['50']
			row[f'{name}_84'] = summary[name]['84']

	# Per-observation v0 and amplitude (multi-obs fits only)
	if hasattr(kin_model, 'v0_per_obs') and kin_model.v0_per_obs:
		for obs_name, stats in kin_model.v0_per_obs.items():
			row[f'v0_{obs_name}_16'] = float(stats['16'])
			row[f'v0_{obs_name}_50'] = float(stats['mean'])
			row[f'v0_{obs_name}_84'] = float(stats['84'])

	if hasattr(kin_model, 'amplitude_per_obs') and kin_model.amplitude_per_obs:
		for obs_name, stats in kin_model.amplitude_per_obs.items():
			row[f'amplitude_{obs_name}_16'] = float(stats['16'])
			row[f'amplitude_{obs_name}_50'] = float(stats['mean'])
			row[f'amplitude_{obs_name}_84'] = float(stats['84'])

	return Table([row])


# ============================================================================
# Saving results
# ============================================================================

def save_fit_results(output, inf_data, kin_model, z_spec, ID, save_runs_path,
                     sigma0_floor=20.0):
	"""Compute derived posteriors, write results table, and save corner plot."""
	inf_data = compute_derived_posterior(inf_data, kin_model, z_spec,
	                                     sigma0_floor=sigma0_floor)

	gm = kin_model.galaxy_model
	from .param_spec import all_param_specs
	specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
	sampled_names = [s.name for s in specs if not s.fixed]
	derived_names = [dq.name for dq in DERIVED_QUANTITIES]
	all_names = sampled_names + ['ellip'] + derived_names

	summary = summarize_posterior(inf_data, all_names)
	res = build_results_table(ID, kin_model, summary)
	res.write(save_runs_path + output + '/' + str(ID) + '_results',
	          format='ascii', overwrite=True)

	# Corner plot: sampled params + all derived quantities flagged include_in_corner
	all_labels    = {dq.name: dq.label for dq in DERIVED_QUANTITIES}
	corner_vars   = _CORNER_SAMPLED_PARAMS + [
	                    dq.name for dq in DERIVED_QUANTITIES if dq.include_in_corner]
	corner_labels = [all_labels.get(n, n) for n in corner_vars]

	fig = plt.figure(figsize=(10, 10))
	CORNER_KWARGS = dict(
		smooth=4,
		label_kwargs=dict(fontsize=20),
		title_kwargs=dict(fontsize=20),
		quantiles=[0.16, 0.5, 0.84],
		plot_density=False,
		plot_datapoints=False,
		fill_contours=True,
		plot_contours=True,
		show_titles=True,
		labels=corner_labels,
		titles=corner_labels,
		max_n_ticks=3,
		divergences=False,
	)
	corner.corner(inf_data, group='posterior', var_names=corner_vars,
	              color='royalblue', range=[0.99] * len(corner_vars),
	              **CORNER_KWARGS)
	plt.tight_layout()
	plt.savefig(save_runs_path + output + '/' + str(ID) + '_v_sigma_corner.png',
	            dpi=300)
	plt.close()


# ============================================================================
# Main postprocessing entry points
# ============================================================================

def process_results(output, master_cat, line, mock_params=None, test=None,
                    j=None, parametric=False, ID=None, save_runs_path=None,
                    field=None, grism_filter='F444W', delta_wave_cutoff=0.02,
                    factor=5, wave_factor=10, model_name='Disk',
                    manual_psf_name=None, manual_grism_file=None,
                    sigma0_floor=20.0):
	"""Post-process single-observation inference data and save all outputs."""
	z_spec, wavelength, wave_space, obs_map, obs_error, kin_model, grism_object, delta_wave = \
		pre.run_full_preprocessing(
			output, master_cat, line, mock_params=mock_params,
			save_runs_path=save_runs_path, source_id=ID, field=field,
			grism_filter=grism_filter, delta_wave_cutoff=delta_wave_cutoff,
			factor=factor, wave_factor=wave_factor, model_name=model_name,
			manual_psf_name=manual_psf_name, manual_grism_file=manual_grism_file)

	if mock_params is None:
		inf_data = az.InferenceData.from_netcdf(
			save_runs_path + output + '/' + str(ID) + '_output')
		j = 0
	else:
		inf_data = az.InferenceData.from_netcdf(
			'testing/' + str(test) + '/' + str(test) + '_' + str(j) + '_output')

	data = fit.Fit_Numpyro(obs_map=obs_map, obs_error=obs_error,
	                        grism_object=grism_object, kin_model=kin_model,
	                        inference_data=inf_data, parametric=parametric)
	inf_data, model_map, model_flux, fluxes_mean, model_velocities, model_dispersions = \
		kin_model.compute_model(inf_data, grism_object, parametric)

	index_min = grism_object.index_min
	index_max = grism_object.index_max
	len_wave = int((wave_space[-1] - wave_space[0]) / delta_wave)
	wave_space = jnp.linspace(wave_space[0], wave_space[-1], len_wave + 1)
	wave_space = wave_space[index_min:index_max]

	save_fit_results(output, inf_data, kin_model, z_spec, ID,
	                 save_runs_path=save_runs_path, sigma0_floor=sigma0_floor)

	summary = summarize_posterior(inf_data, [dq.name for dq in DERIVED_QUANTITIES])
	kin_model.plot_summary(obs_map, obs_error, inf_data, wave_space,
	                       save_to_folder=output, name='summary',
	                       v_re=summary.get('v_re', {}).get('50'),
	                       save_runs_path=save_runs_path, ID=ID)

	return summary, kin_model, inf_data


def process_results_multi(observations, results, output, master_cat, line,
                          parametric, ID, save_runs_path, field,
                          grism_filter='F444W', delta_wave_cutoff=0.02,
                          factor=5, wave_factor=10, model_name='Disk',
                          manual_psf_name=None, manual_grism_file=None,
                          sigma0_floor=20.0):
	"""Post-process multi-observation inference data and save all outputs."""
	from . import plotting

	inf_data = az.InferenceData.from_netcdf(
		save_runs_path + output + '/' + str(ID) + '_output_multi')

	z_spec, wavelength, wave_space_ref, obs_map_ref, obs_error_ref, kin_model, \
		grism_object_ref, delta_wave = pre.run_full_preprocessing(
			output=output, master_cat=master_cat, line=line,
			save_runs_path=save_runs_path, source_id=ID, field=field,
			grism_filter=grism_filter, delta_wave_cutoff=delta_wave_cutoff,
			factor=factor, wave_factor=wave_factor, model_name=model_name,
			manual_psf_name=manual_psf_name, manual_grism_file=manual_grism_file)

	inf_data, _ = kin_model.compute_model_parametric_multi(inf_data, observations)

	index_min = grism_object_ref.index_min
	index_max = grism_object_ref.index_max
	len_wave = int((wave_space_ref[-1] - wave_space_ref[0]) / delta_wave)
	wave_space = jnp.linspace(wave_space_ref[0], wave_space_ref[-1], len_wave + 1)
	wave_space = wave_space[index_min:index_max]

	save_fit_results(output, inf_data, kin_model, z_spec, ID,
	                 save_runs_path=save_runs_path, sigma0_floor=sigma0_floor)

	summary = summarize_posterior(inf_data, [dq.name for dq in DERIVED_QUANTITIES])

	obs_radius = kin_model.r_eff_mean
	ellip      = kin_model.ellip_mean
	theta_Ha   = kin_model.PA_morph_mean / (180 / jnp.pi) + jnp.pi / 2
	n          = kin_model.n_mean

	plotting.plot_disk_summary_multi(
		observations=observations,
		results=results,
		inf_data=inf_data,
		wave_space=wave_space,
		x0=kin_model.x0_vel_mean,
		y0=kin_model.y0_vel_mean,
		factor=1,
		direct_image_size=kin_model.im_shape[0],
		save_to_folder=output,
		name='summary',
		obs_radius=obs_radius,
		ellip=ellip,
		theta_Ha=theta_Ha,
		n=n,
		save_runs_path=save_runs_path,
		ID=ID,
		galaxy_model=kin_model.galaxy_model,
	)

	try:
		individual_inf_data = []
		obs_labels = []
		for obs in observations:
			obs_path = save_runs_path + str(ID) + '/' + obs.name + '/' + str(ID) + '_output'
			individual_inf_data.append(az.InferenceData.from_netcdf(obs_path))
			obs_labels.append(obs.name)

		all_inf_data = individual_inf_data + [inf_data]
		all_labels   = obs_labels + ['Joint']
		colors = ['royalblue', 'crimson', 'grey'][:len(all_inf_data)]

		comparison_path = save_runs_path + output + str(ID) + '_comparison_cornerplot.png'
		plotting.plot_multi_obs_comparison_cornerplot(
			inf_data_list=all_inf_data,
			run_labels=all_labels,
			colors=colors,
			save_path=comparison_path,
			ID=ID)
		print(f'  Saved comparison cornerplot: {comparison_path}')
	except Exception as e:
		print(f'  WARNING: Could not generate comparison cornerplot: {e}')

	return summary, kin_model, inf_data


# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='',
                    help='folder of the galaxy you want to postprocess')
parser.add_argument('--line', type=str, default='H_alpha',
                    help='line to fit')
parser.add_argument('--master_cat', type=str,
                    default='CONGRESS_FRESCO/master_catalog.cat',
                    help='master catalog file to use for the post-processing')

if __name__ == "__main__":
	args = parser.parse_args()
	inf_data = az.InferenceData.from_netcdf(
		'fitting_results/' + args.output + '/output')
	process_results(args.output, args.master_cat, args.line)
