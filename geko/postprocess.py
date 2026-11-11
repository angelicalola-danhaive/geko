"""
Post-processing functions for geko grism fits.

	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['process_results', 'process_results_multi',
           'compute_derived_posterior', 'DERIVED_QUANTITIES',
           'instrumental_sigma0_floor', 'sigma0_limits',
           'compute_information_gain']

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
_CORNER_SAMPLED_PARAMS  = ['sigma0']
_CORNER_SAMPLED_LABELS  = {'sigma0': r'$\sigma_0$ [km/s]'}

# Ordered by dependency: each entry may use results from earlier entries.
# v_re is computed by utils.add_v_re before this loop runs; the entry here is
# a pass-through so that v_re is included in summarize_posterior automatically.
DERIVED_QUANTITIES = [
	DerivedQuantity(
		name='v_re',
		label=r'$v_{re}$ [km/s]',
		compute=lambda post, ctx: post['v_re'],
	),
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

_FALLBACK_SIGMA0_FLOOR = 20.0   # km/s, only when no grism object is available


def instrumental_sigma0_floor(grism_objects, fraction=0.5):
	"""
	sigma0 below which the intrinsic dispersion is treated as unresolved:
	`fraction` times the instrumental line width, averaged over the observations
	(Grism.sigma_inst_kms, measured from each observation's dispersed PSF and
	detector pixel, so it follows the line wavelength, filter and module).
	At fraction = 0.5 the intrinsic sigma0 broadens the observed line core by ~12%.
	"""
	if not isinstance(grism_objects, (list, tuple)):
		grism_objects = [grism_objects]
	return fraction * float(np.mean([g.sigma_inst_kms for g in grism_objects]))


def compute_derived_posterior(inf_data, kin_model, z_spec,
                              sigma0_floor=None, pixel_scale=_PIXEL_SCALE,
                              grism_objects=None):
	"""Add all derived quantities to inf_data.posterior.

	Parameters
	----------
	sigma0_floor : float, optional
		sigma0 (km/s) below which it is treated as unresolved. Samples below it
		are excluded from the v_sigma posterior (so the ratio doesn't diverge),
		and sigma0 is reported as an upper limit when its 16th percentile is
		below it (see sigma0_limits). Default: instrumental_sigma0_floor of
		grism_objects; 20 km/s if none are given.
	pixel_scale : float
		Detector pixel scale in arcsec (default NIRCam 0.063 arcsec/px).
	grism_objects : Grism or list of Grism, optional
		The fitted observation(s), used for the default sigma0_floor.

	The floor used is stored in inf_data.posterior.attrs['sigma0_floor'].
	"""
	if sigma0_floor is None:
		if grism_objects is not None:
			sigma0_floor = instrumental_sigma0_floor(grism_objects)
		else:
			sigma0_floor = _FALLBACK_SIGMA0_FLOOR
			print(f'compute_derived_posterior: no grism object given, using sigma0_floor = {sigma0_floor} km/s')
	utils.add_v_re(inf_data, kin_model, grism_object=None,
	               num_samples=inf_data.posterior['sigma0'].shape[1])

	context = {'z_spec': z_spec, 'pixel_scale': pixel_scale,
	           'sigma0_floor': sigma0_floor}
	for dq in DERIVED_QUANTITIES:
		inf_data.posterior[dq.name] = dq.compute(inf_data.posterior, context)
	inf_data.posterior.attrs['sigma0_floor'] = float(sigma0_floor)

	return inf_data


def sigma0_limits(inf_data, sigma0_floor):
	"""
	Decide whether sigma0 is resolved, and the limits to quote if it is not.

	sigma0 is an upper limit when its 16th percentile is below sigma0_floor. Then
	sigma0 < its 84th percentile, and v/sigma > the 16th percentile of |v_re|/sigma0
	over all samples (|v_re|: the sign of Va only encodes the rotation direction;
	all samples: the v_sigma posterior drops samples below the floor, which would
	bias the limit low exactly when sigma0 is unresolved).

	Returns a dict: sigma0_floor, sigma0_is_upper_limit, sigma0_limit and
	v_sigma_limit (the last two NaN when sigma0 is resolved).
	"""
	s0 = inf_data.posterior['sigma0'].values.ravel()
	q16, q84 = np.percentile(s0, [16, 84])
	is_ul = bool(q16 < sigma0_floor)
	out = {'sigma0_floor': float(sigma0_floor), 'sigma0_is_upper_limit': is_ul,
	       'sigma0_limit': np.nan, 'v_sigma_limit': np.nan}
	if is_ul:
		vre = np.abs(inf_data.posterior['v_re'].values.ravel())
		with np.errstate(divide='ignore', invalid='ignore'):
			ratio = np.where(s0 > 0, vre / s0, np.inf)
		out['sigma0_limit'] = float(q84)
		out['v_sigma_limit'] = float(np.percentile(ratio, 16))
	return out


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


# np.trapz was removed in numpy 2 (np.trapezoid since numpy 2.0)
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def _kl_kde(post, prior, n_grid=1000):
	"""KL(posterior || prior) via KDE — works for any prior shape."""
	from scipy.stats import gaussian_kde
	lo = min(post.min(), prior.min())
	hi = max(post.max(), prior.max())
	margin = 0.05 * (hi - lo) if hi > lo else 1.0
	x = np.linspace(lo - margin, hi + margin, n_grid)
	q = gaussian_kde(post)(x);  q /= _trapezoid(q, x)
	p = gaussian_kde(prior)(x); p /= _trapezoid(p, x)
	mask = q > 1e-10 * q.max()
	return float(_trapezoid(
		np.where(mask, q * np.log(np.clip(q, 1e-300, None) / np.clip(p, 1e-300, None)), 0.0), x))


def compute_information_gain(inf_data, param_names=None):
	"""Compute per-parameter KL divergence and posterior contraction.

	KL divergence is KL(posterior || prior) estimated via KDE (no Gaussian assumption).
	Posterior contraction is 1 - Var(posterior) / Var(prior).

	Parameters
	----------
	inf_data : arviz.InferenceData
		Must contain both 'posterior' and 'prior' groups.
	param_names : list of str, optional
		Parameters to compute. Defaults to all variables present in both groups.

	Returns
	-------
	astropy.table.Table
		Columns: param, kl_divergence, posterior_contraction,
		prior_mean, prior_std, post_mean, post_std.
	"""
	if not hasattr(inf_data, 'prior'):
		raise ValueError("inf_data has no 'prior' group.")

	avail = {p for p in set(inf_data.posterior.data_vars) & set(inf_data.prior.data_vars)
	         if not p.startswith('unscaled_')}
	if param_names is None:
		param_names = sorted(avail)
	else:
		param_names = [p for p in param_names if p in avail]

	rows = []
	for name in param_names:
		post = inf_data.posterior[name].values.ravel()
		prior = inf_data.prior[name].values.ravel()
		post  = post[np.isfinite(post)]
		prior = prior[np.isfinite(prior)]
		if len(post) < 2 or len(prior) < 2:
			continue

		mu_prior    = float(np.mean(prior))
		sigma_prior = float(np.std(prior))
		mu_post     = float(np.mean(post))
		sigma_post  = float(np.std(post))

		if sigma_prior <= 0 or sigma_post <= 0:
			continue

		var_ratio = (sigma_post / sigma_prior) ** 2
		kl = _kl_kde(post, prior)
		pc = 1.0 - var_ratio

		rows.append({
			'param':                 name,
			'kl_divergence':         float(kl),
			'posterior_contraction': float(pc),
			'prior_mean':            mu_prior,
			'prior_std':             sigma_prior,
			'post_mean':             mu_post,
			'post_std':              sigma_post,
		})

	_empty_cols = ['param', 'kl_divergence', 'posterior_contraction',
	               'prior_mean', 'prior_std', 'post_mean', 'post_std']
	return Table(rows) if rows else Table(names=_empty_cols)


def build_results_table(ID, kin_model, summary, limits=None):
	"""Build an astropy Table from a posterior summary dict.

	Columns are driven by the model's actual parameter specs plus the
	derived-quantity registry — no hardcoded parameter names. `limits` (from
	sigma0_limits) adds sigma0_floor, sigma0_is_upper_limit, sigma0_limit and
	v_sigma_limit columns.
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

	if limits is not None:
		# booleans as 1/0 so the ascii table reads back as numbers
		row.update({k: (int(v) if isinstance(v, bool) else v) for k, v in limits.items()})

	return Table([row])


# ============================================================================
# Saving results
# ============================================================================

def save_fit_results(output, inf_data, kin_model, z_spec, ID, save_runs_path,
                     sigma0_floor=None, grism_objects=None):
	"""Compute derived posteriors, write results table, and save corner plot.

	sigma0_floor defaults to the instrumental floor of grism_objects (see
	compute_derived_posterior). When sigma0 is unresolved (sigma0_limits), the
	results table flags it and the corner plot quotes sigma0 < X and v/sigma > Y.
	"""
	inf_data = compute_derived_posterior(inf_data, kin_model, z_spec,
	                                     sigma0_floor=sigma0_floor,
	                                     grism_objects=grism_objects)
	limits = sigma0_limits(inf_data, inf_data.posterior.attrs['sigma0_floor'])
	if limits['sigma0_is_upper_limit']:
		print(f"sigma0 unresolved (16th percentile < floor {limits['sigma0_floor']:.1f} km/s): "
		      f"sigma0 < {limits['sigma0_limit']:.1f} km/s, v/sigma > {limits['v_sigma_limit']:.2f}")

	gm = kin_model.galaxy_model
	from .param_spec import all_param_specs
	specs = all_param_specs(gm.morph_model, gm.shared_kin_specs, gm.rot_model)
	sampled_names = [s.name for s in specs if not s.fixed]
	derived_names = [dq.name for dq in DERIVED_QUANTITIES]
	all_names = sampled_names + ['ellip'] + derived_names

	summary = summarize_posterior(inf_data, all_names)
	res = build_results_table(ID, kin_model, summary, limits=limits)
	res.write(save_runs_path + output + '/' + str(ID) + '_results',
	          format='ascii', overwrite=True)

	try:
		info_table = compute_information_gain(inf_data, sampled_names)
		if len(info_table) > 0:
			info_path = save_runs_path + output + '/' + str(ID) + '_info_gain'
			info_table.write(info_path, format='ascii', overwrite=True)
			print(f'Saved information gain table to {info_path}')
	except Exception as e:
		print(f'Warning: failed to compute information gain: {e}')

	# Corner plot: sampled params + all derived quantities flagged include_in_corner
	all_labels  = {**_CORNER_SAMPLED_LABELS, **{dq.name: dq.label for dq in DERIVED_QUANTITIES}}
	candidate_vars   = _CORNER_SAMPLED_PARAMS + [
	                       dq.name for dq in DERIVED_QUANTITIES if dq.include_in_corner]

	# Filter out variables whose finite samples don't span a valid range (e.g. v_sigma
	# when sigma0 is always below the floor → all NaN → matplotlib rejects NaN limits).
	# When sigma0 is unresolved, the v/sigma panel shows |v_re|/sigma0 over all
	# samples (the distribution its lower limit is taken from), cut at its 90th
	# percentile since it extends to infinity as sigma0 -> 0.
	is_ul = limits['sigma0_is_upper_limit']
	def _corner_column(var):
		if var == 'v_sigma' and is_ul:
			s0 = inf_data.posterior['sigma0'].values.ravel()
			vre = np.abs(inf_data.posterior['v_re'].values.ravel())
			with np.errstate(divide='ignore', invalid='ignore'):
				return np.where(s0 > 0, vre / s0, np.nan)
		return inf_data.posterior[var].values.ravel()

	corner_vars, corner_labels, corner_ranges, columns = [], [], [], []
	for var in candidate_vars:
		if var not in inf_data.posterior:
			continue
		flat = _corner_column(var)
		finite = flat[np.isfinite(flat)]
		if len(finite) < 10:
			print(f'  Corner plot: skipping {var!r} (fewer than 10 finite samples)')
			continue
		lo = np.percentile(finite, 0.5)
		hi = np.percentile(finite, 90 if (var == 'v_sigma' and is_ul) else 99.5)
		if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
			print(f'  Corner plot: skipping {var!r} (degenerate range [{lo}, {hi}])')
			continue
		corner_vars.append(var)
		corner_labels.append(r'$|v_{re}|/\sigma_0$' if (var == 'v_sigma' and is_ul) else all_labels.get(var, var))
		corner_ranges.append((lo, hi))
		columns.append(np.where(np.isfinite(flat), flat, np.nan))

	data_matrix = np.column_stack(columns)

	# Pass the full data matrix (NaN rows kept so sigma0 is uncut).
	# Quantile lines and titles are drawn manually with nanpercentile so columns
	# with NaN (v_sigma) get correct lines and no NaN in the title text.
	fig = plt.figure(figsize=(10, 10))
	CORNER_KWARGS = dict(
		smooth=4,
		label_kwargs=dict(fontsize=20),
		plot_density=False,
		plot_datapoints=False,
		fill_contours=True,
		plot_contours=True,
		show_titles=False,
		labels=corner_labels,
		max_n_ticks=3,
	)
	corner.corner(data_matrix, fig=fig, color='royalblue',
	              range=corner_ranges, **CORNER_KWARGS)

	def _fmt(val):
		if abs(val) >= 100:
			return f'{val:.0f}'
		elif abs(val) >= 10:
			return f'{val:.1f}'
		else:
			return f'{val:.2f}'

	ndim = len(corner_vars)
	axes = np.array(fig.get_axes()).reshape(ndim, ndim)
	for i, (var, label) in enumerate(zip(corner_vars, corner_labels)):
		col = data_matrix[:, i]
		q16, q50, q84 = np.nanpercentile(col, [16, 50, 84])
		ax = axes[i, i]
		if is_ul and var == 'sigma0':
			lines = (limits['sigma0_limit'],)          # upper limit only
		elif is_ul and var == 'v_sigma':
			lines = (limits['v_sigma_limit'],)         # lower limit only
		else:
			lines = (q16, q50, q84)
		for qval in lines:
			ax.axvline(qval, color='royalblue', linestyle='dashed', lw=1.5)
		if var == 'sigma0' and limits['sigma0_is_upper_limit']:
			title = f'{label}\n$<{_fmt(limits["sigma0_limit"])}$'
		elif var == 'v_sigma' and limits['sigma0_is_upper_limit']:
			title = f'{label}\n$>{_fmt(limits["v_sigma_limit"])}$'
		else:
			title = (f'{label}\n'
			         f'${_fmt(q50)}' + r'^{+' + _fmt(q84 - q50) + r'}_{-' + _fmt(q50 - q16) + r'}$')
		ax.set_title(title, fontsize=12)

	plt.savefig(save_runs_path + output + '/' + str(ID) + '_derived_corner.png',
	            dpi=300, bbox_inches='tight')
	plt.close()


# ============================================================================
# Main postprocessing entry points
# ============================================================================

def process_results(output, master_cat, line, mock_params=None, test=None,
                    j=None, parametric=False, ID=None, save_runs_path=None,
                    field=None, grism_filter='F444W', delta_wave_cutoff=0.02,
                    factor=5, wave_factor=10, model_name='Disk',
                    manual_psf_name=None, manual_grism_file=None,
                    sigma0_floor=None, flux_scaling=None):
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

	# Save S/N mask image (once, for diagnostics)
	if flux_scaling is not None and flux_scaling.mode == 'row_wise':
		import numpy as np
		sn_mask = np.abs(np.array(obs_map)) / np.array(obs_error) > 1.0
		fig, ax = plt.subplots(figsize=(5, 4))
		ax.imshow(sn_mask, origin='lower', cmap='gray_r', vmin=0, vmax=1)
		ax.set_title('S/N > 1 mask (row-wise scaling)')
		ax.set_xlabel('wavelength'); ax.set_ylabel('spatial')
		plt.tight_layout()
		plt.savefig(save_runs_path + output + '/' + str(ID) + '_sn_mask.png', dpi=150)
		plt.close()

	data = fit.Fit_Numpyro(obs_map=obs_map, obs_error=obs_error,
	                        grism_object=grism_object, kin_model=kin_model,
	                        inference_data=inf_data, parametric=parametric)
	kin_model.flux_scaling = flux_scaling
	inf_data, model_map, model_flux, fluxes_mean, model_velocities, model_dispersions = \
		kin_model.compute_model(inf_data, grism_object, parametric,
		                        obs_map=obs_map, obs_error=obs_error)

	# Save posterior-mean pixel-wise scale map
	if flux_scaling is not None and flux_scaling.mode == 'pixel_wise':
		if 'log_intrinsic_scale' in inf_data.posterior:
			import numpy as np
			log_s = np.array(inf_data.posterior['log_intrinsic_scale'].median(dim=['chain', 'draw']))
			scale_map = np.exp(log_s)
			vmax = np.percentile(np.abs(log_s), 99)
			fig, axes = plt.subplots(1, 2, figsize=(10, 4))
			im0 = axes[0].imshow(log_s, origin='lower', cmap='RdBu_r',
			                     vmin=-vmax, vmax=vmax)
			axes[0].set_title('log scale map (posterior median)')
			plt.colorbar(im0, ax=axes[0])
			im1 = axes[1].imshow(scale_map, origin='lower', cmap='RdBu_r',
			                     vmin=1/np.exp(vmax), vmax=np.exp(vmax))
			axes[1].set_title('scale map exp(log s)')
			plt.colorbar(im1, ax=axes[1])
			for ax in axes:
				ax.set_xlabel('x'); ax.set_ylabel('y')
			plt.tight_layout()
			plt.savefig(save_runs_path + output + '/' + str(ID) + '_scale_map.png', dpi=150)
			plt.close()
			np.save(save_runs_path + output + '/' + str(ID) + '_scale_map.npy', scale_map)

	index_min = grism_object.index_min
	index_max = grism_object.index_max
	len_wave = int((wave_space[-1] - wave_space[0]) / delta_wave)
	wave_space = jnp.linspace(wave_space[0], wave_space[-1], len_wave + 1)
	wave_space = wave_space[index_min:index_max]

	save_fit_results(output, inf_data, kin_model, z_spec, ID,
	                 save_runs_path=save_runs_path, sigma0_floor=sigma0_floor,
	                 grism_objects=grism_object)

	summary = summarize_posterior(inf_data, [dq.name for dq in DERIVED_QUANTITIES])
	kin_model.plot_summary(obs_map, obs_error, inf_data, wave_space,
	                       save_to_folder=output, name='summary',
	                       v_re=summary.get('v_re', {}).get('50'),
	                       save_runs_path=save_runs_path, ID=ID, line=line)

	return summary, kin_model, inf_data


def process_results_multi(observations, results, output, master_cat, line,
                          parametric, ID, save_runs_path, field,
                          grism_filter='F444W', delta_wave_cutoff=0.02,
                          factor=5, wave_factor=10, model_name='Disk',
                          manual_psf_name=None, manual_grism_file=None,
                          sigma0_floor=None):
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
	                 save_runs_path=save_runs_path, sigma0_floor=sigma0_floor,
	                 grism_objects=[obs.grism for obs in observations])

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
		line=line,
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
