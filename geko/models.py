"""

	Module holding all of the kinematic models used in the fitting process.

	Written by A L Danhaive: ald66@cam.ac.uk
"""
__all__ = ["KinModels"]

# imports
import numpy as np
# geko related imports
from . import  utils
from . import  plotting

# jax and its functions
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.signal import convolve
# from scipy.signal import convolve
from jax import image

# from skimage.morphology import dilation, disk

from astropy.modeling.models import GeneralSersic2D, Sersic2D

# scipy and its functions
from scipy.constants import pi
from scipy.ndimage import measurements

# numpyro and its functions
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam, CircularReparam, LocScaleReparam

from matplotlib import pyplot as plt

from photutils import centroids

from astropy.cosmology import Planck18 as cosmo

import time

from scipy.constants import c

import math
import xarray as xr
jax.config.update('jax_enable_x64', True)
# numpyro.enable_validation()  # Disabled for performance - only enable for debugging

# ============================================================================
# STANDALONE JIT-COMPILED FUNCTIONS (extracted from class methods)
# ============================================================================

@jax.jit
def _v_rad_core(x, y, PA, i, Va, r_t, r):
	"""Core computation for radial velocity (extracted from KinModels.v_rad)"""
	return (2/pi)*Va*jnp.arctan(r/r_t)*jnp.sin(i)


@jax.jit  
def _vel1d_core(r, Va, r_t):
	"""Core computation for 1D velocity profile (extracted from KinModels.vel1d)"""
	r_t_safe = jnp.where(r_t != 0.0, r_t, 1.0)
	r_safe = jnp.where(r != 0.0, r, 1.0)
	v_out = jnp.where((r_t!=0.0) & (r!=0.0), (2.0/jnp.pi)*Va*jnp.arctan2(r_safe,r_t_safe), 0.0)
	return jnp.array(v_out)


@jax.jit
def _v_core(x, y, PA, i, Va, r_t):
	"""Core computation for velocity field (extracted from KinModels.v)"""
	i_rad = i / 180 * jnp.pi
	PA_rad = PA / 180 * jnp.pi
	
	# Precompute trigonometric values
	sini = jnp.sin(i_rad)
	cosi = jnp.cos(i_rad)
	
	# Rotate coordinates
	x_rot = x * jnp.cos(PA_rad) - y * jnp.sin(PA_rad)
	y_rot = x * jnp.sin(PA_rad) + y * jnp.cos(PA_rad)
	
	# Safeguard for cases where cosi is zero
	i_rad_safe = jnp.where(cosi != 0, i_rad, 0)
	cosi_safe = jnp.where(cosi != 0, cosi, 1e-6)  # Use a small epsilon to avoid division by zero
	
	# Calculate r, handling x_rot = 0 and y_rot = 0 cases separately
	r_squared = x_rot**2 / cosi_safe**2 + y_rot**2
	r_safe_squared = jnp.where(r_squared != 0.0, r_squared, 1e-12)  # Small epsilon to avoid sqrt(0)
	r = jnp.sqrt(r_safe_squared)
	
	# Handle the special case where r = 0 or where both x_rot and y_rot are 0 explicitly
	r_safe = jnp.where((x_rot != 0) | (y_rot != 0), r, 1e-6)  # Use a small epsilon for r when both x_rot and y_rot are zero
	
	# Calculate observed velocity using the standalone vel1d function
	# For edge-on (cos(i)=0), use y_rot (major axis) as the visible rotation radius
	vel_obs = jnp.where(cosi != 0, _vel1d_core(r_safe, Va, r_t) * sini, _vel1d_core(y_rot, Va, r_t))
	
	# Final velocity computation, handling r = 0 or x_rot = y_rot = 0 case
	vel_obs_final = jnp.where(r_safe != 0.0, vel_obs * (y_rot / r_safe), 0.0)
	
	return vel_obs_final


@jax.jit
def _v_int_core(x, y, PA, i, Va, r_t):
	"""Core computation for integrated velocity field (extracted from KinModels.v_int)"""
	i_rad = i / 180 * jnp.pi
	PA_rad = PA / 180 * jnp.pi
	
	# Precompute trigonometric values
	sini = jnp.sin(i_rad)
	cosi = jnp.cos(i_rad)
	
	# Rotate coordinates
	x_rot = x * jnp.cos(PA_rad) - y * jnp.sin(PA_rad)
	y_rot = x * jnp.sin(PA_rad) + y * jnp.cos(PA_rad)
	
	# Safeguard for cases where cosi is zero
	i_rad_safe = jnp.where(cosi != 0, i_rad, 0)
	cosi_safe = jnp.where(cosi != 0, cosi, 1e-6)  # Use a small epsilon to avoid division by zero
	
	# Calculate r, handling x_rot = 0 and y_rot = 0 cases separately
	r_squared = x_rot**2 / cosi_safe**2 + y_rot**2
	r_safe_squared = jnp.where(r_squared != 0.0, r_squared, 1e-12)  # Small epsilon to avoid sqrt(0)
	r = jnp.sqrt(r_safe_squared)
	
	# Handle the special case where r = 0 or where both x_rot and y_rot are 0 explicitly
	r_safe = jnp.where((x_rot != 0) | (y_rot != 0), r, 1e-6)  # Use a small epsilon for r when both x_rot and y_rot are zero
	
	# Calculate observed velocity using the standalone vel1d function
	# For edge-on (cos(i)=0), use y_rot (major axis) as the visible rotation radius
	vel_obs = jnp.where(cosi != 0, _vel1d_core(r_safe, Va, r_t) * sini, _vel1d_core(y_rot, Va, r_t))
	
	# Final velocity computation, handling r = 0 or x_rot = y_rot = 0 case
	vel_obs_final = jnp.where(r_safe != 0.0, vel_obs * (y_rot / r_safe), 0.0)
	
	return vel_obs_final

# ============================================================================

class KinModels:
	'''
		This top level class only contains the functions to make the velocity maps. The rest will be specific to each sub class.
	'''


	def __init__(self):
		"""
		Initialize a new kinematic model.

		Creates a base kinematic model object with velocity field calculations
		using arctangent rotation curve parameterization.
		"""
		print('New kinematic model created')
	
	def v_rad(self, x, y, PA, i, Va, r_t, r):
		"""Radial velocity component"""
		return _v_rad_core(x, y, PA, i, Va, r_t, r)


	def v(self, x, y, PA, i, Va, r_t):
		"""2D velocity field calculation"""
		return _v_core(x, y, PA, i, Va, r_t)
	
	def v_int(self, x, y, PA, i, Va, r_t):
		"""Integrated velocity field calculation"""
		return _v_int_core(x, y, PA, i, Va, r_t)
	


	def vel1d(self, r, Va, r_t):
		"""1D velocity profile"""
		return _vel1d_core(r, Va, r_t)


		# return dispersions

		

	def set_main_bounds(self, factor, wave_factor, x0, x0_vel, y0, y0_vel):
		"""
		Set basic configuration parameters for the kinematic model.

		Parameters
		----------
		factor : int
			Spatial oversampling factor
		wave_factor : int
			Wavelength oversampling factor
		x0, y0 : float
			Morphological centroid positions (pixels)
		x0_vel, y0_vel : float
			Velocity centroid positions (pixels)

		Notes
		-----
		All priors (morphological and kinematic) are set separately via
		set_priors_from_config() or set_parametric_priors().
		"""
		self.factor = factor
		self.wave_factor = wave_factor

		# Centroid positions
		self.x0 = x0
		self.y0 = y0
		self.x0_vel = x0_vel
		self.mu_y0_vel = y0_vel

		# Default velocity centroid to morphological centroid if not provided
		if self.mu_y0_vel is None:
			self.x0_vel = x0
			self.mu_y0_vel = y0

	def rescale_to_mask(self, array, mask):
		"""
			Rescale the bounds to the mask
		"""
		rescaled_array = []
		for a in array:
			a = a[jnp.where(mask == 1)]
			rescaled_array.append(a)
		return rescaled_array

class GalaxyModel:
	"""
	Galaxy morphological and kinematic model for parametric fitting.

	Wraps a MorphologyModel and CompositeRotationCurve with shared kinematic
	parameters. Handles prior setting, parameter sampling, and model evaluation
	for MCMC fitting.

	Parameters
	----------
	direct_shape : tuple or int
		Shape of the direct image
	factor : int
		Spatial oversampling factor
	morph_model : MorphologyModel, optional
		Morphology model (default: SersicMorphology)
	rot_model : CompositeRotationCurve, optional
		Rotation curve model (default: CompositeRotationCurve([ArctanComponent()]))
	r_eff : float
		Effective radius in pixels

	Attributes
	----------
	im_shape : tuple
		Image dimensions
	factor : int
		Oversampling factor
	"""
	def __init__(self, direct_shape, factor, morph_model=None, rot_model=None):
		from .morph_models import SersicMorphology
		from .rotation_models import CompositeRotationCurve, ArctanComponent
		from .param_spec import SHARED_KINEMATIC_SPEC, _apply_fixed_to_specs
		from copy import deepcopy

		self.direct_shape = direct_shape
		self.factor = factor
		self.morph_model = morph_model or SersicMorphology()
		self.rot_model = rot_model or CompositeRotationCurve([ArctanComponent()])
		self.shared_kin_specs = deepcopy(SHARED_KINEMATIC_SPEC)

		# Link kinematic center to morphological center by default
		_apply_fixed_to_specs(self.shared_kin_specs, {
			'x0_vel': 'xc_morph',
			'y0_vel': 'yc_morph',
		})

	@property
	def amplitude_mu(self):
		for spec in self.morph_model.parameters:
			if spec.name == 'amplitude':
				return spec.prior_mu
		raise AttributeError("amplitude ParameterSpec not found")

	@property
	def amplitude_std(self):
		for spec in self.morph_model.parameters:
			if spec.name == 'amplitude':
				return spec.prior_std
		raise AttributeError("amplitude ParameterSpec not found")

	def set_parametric_priors(self, py_table, flux_measurements, redshift, wavelength, delta_wave, theta_rot=0.0, shape=31):
		"""Set morphological and kinematic priors from PySersic fitting results."""
		from .param_spec import _apply_overrides_to_specs

		arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift).value
		kpc_per_pixel = 0.063 / arcsec_per_kpc

		ellip = py_table['ellip_q50'][0]
		inclination = utils.compute_inclination(ellip=ellip, q0=0.2)
		inclination_err = ((utils.compute_inclination(ellip=py_table['ellip_q84'][0], q0=0.2) - inclination) +
		                   (inclination - utils.compute_inclination(ellip=py_table['ellip_q16'][0], q0=0.2))) / 2
		inclination_std = inclination_err

		r_eff_UV = py_table['r_eff_q50'][0] / 2
		r_eff_Ha = r_eff_UV
		r_eff_std = np.maximum(3, r_eff_Ha)

		r_eff_min = 0.1 / kpc_per_pixel
		r_eff_max = 10.0 / kpc_per_pixel

		n = py_table['n_q50'][0]
		n_std = 1

		int_flux, int_flux_err = flux_measurements
		amplitude = utils.int_flux_to_flux_density(int_flux, wavelength, delta_wave)
		amplitude_std = utils.int_flux_to_flux_density(int_flux, wavelength, delta_wave)

		xc_morph_py = py_table['xc_q50'][0] / 2
		xc_morph = xc_morph_py + (shape - 20) / 2
		xc_std = 0.25 * r_eff_Ha

		yc_morph_py = py_table['yc_q50'][0] / 2
		yc_morph = yc_morph_py + (shape - 20) / 2
		yc_std = 0.25 * r_eff_Ha

		xc_center, yc_center = (shape - 1) / 2, (shape - 1) / 2
		xc_morph_rot, yc_morph_rot = utils.rotate_coords(xc_morph, yc_morph, xc_center, yc_center, theta_rot)

		theta = py_table['theta_q50'][0]
		print('Rotating the prior by', theta_rot, 'radians, from', theta, 'radians to', theta - theta_rot, 'radians')
		theta_rot_adj = (theta - theta_rot) % (2 * jnp.pi)

		PA = (theta_rot_adj - jnp.pi / 2) * (180 / jnp.pi)
		if PA < 0:
			print('Converting pysersic PA from', PA, 'to', PA + 180, 'degrees')
			PA += 180
		elif PA > 180:
			print('Converting pysersic PA from', PA, 'to', PA - 180, 'degrees')
			PA -= 180
		PA = 90 - PA
		if PA < 0:
			PA += 180
		PA_mean_err = ((py_table['theta_q84'][0] - py_table['theta_q50'][0]) +
		               (py_table['theta_q50'][0] - py_table['theta_q16'][0])) / 2
		PA_std = PA_mean_err * (180 / jnp.pi)
		print('Setting parametric priors:', PA, inclination, r_eff_Ha, n, amplitude, xc_morph, yc_morph)

		self.morph_model.apply_prior_overrides({
			'PA_morph_mu': PA, 'PA_morph_std': PA_std,
			'r_eff_mu': r_eff_Ha, 'r_eff_std': r_eff_std,
			'r_eff_min': r_eff_min, 'r_eff_max': r_eff_max,
			'n_mu': n, 'n_std': n_std, 'n_min': 0.36, 'n_max': 8.0,
			'amplitude_mu': amplitude, 'amplitude_std': amplitude_std, 'amplitude_min': 0.0,
			'xc_morph_mu': xc_morph_rot, 'xc_morph_std': xc_std,
			'yc_morph_mu': yc_morph_rot, 'yc_morph_std': yc_std,
		})

		_apply_overrides_to_specs(self.shared_kin_specs, {
			'i_mu': inclination, 'i_std': inclination_std,
			'PA_mu': PA, 'PA_std': PA_std * 2,
			'sigma0_min': 0.0, 'sigma0_max': 500.0,
			'v0_mu': 0.0, 'v0_std': 200.0,
		})

		for comp in self.rot_model.components:
			comp.apply_prior_overrides({'Va_min': -1000.0, 'Va_max': 1000.0})

	def set_parametric_priors_test(self, priors):
		"""Set priors from a test dict (backward compat)."""
		from .param_spec import _apply_overrides_to_specs

		r_eff_Ha = (1.676 / 0.4) * priors['r_t']

		self.morph_model.apply_prior_overrides({
			'PA_morph_mu': priors['PA'], 'PA_morph_std': 5,
			'r_eff_mu': r_eff_Ha, 'r_eff_std': np.maximum(3, r_eff_Ha),
			'r_eff_min': 0, 'r_eff_max': 15,
			'n_mu': priors['n'], 'n_std': 1, 'n_min': 0.36, 'n_max': 8.0,
			'amplitude_mu': 200, 'amplitude_std': 40, 'amplitude_min': 0.0,
			'xc_morph_mu': 15, 'xc_morph_std': 1,
			'yc_morph_mu': 15, 'yc_morph_std': 1,
		})

		_apply_overrides_to_specs(self.shared_kin_specs, {
			'i_mu': priors['i'], 'i_std': 5,
			'PA_mu': priors['PA'], 'PA_std': 10,
			'sigma0_min': 0, 'sigma0_max': 600,
			'v0_mu': 0.0, 'v0_std': 200.0,
		})

		for comp in self.rot_model.components:
			comp.apply_prior_overrides({'Va_min': -1000, 'Va_max': 1000})

		print('Set mock kinematic priors:', priors['PA'], priors['i'], r_eff_Ha, 200, priors['n'], 15, 15)

	def apply_config_overrides(self, config):
		"""Apply config override dicts to morph, shared kinematics, and rotation components."""
		from .param_spec import _apply_overrides_to_specs

		if config.morph_prior_overrides:
			self.morph_model.apply_prior_overrides(config.morph_prior_overrides)
		if config.geom_prior_overrides:
			_apply_overrides_to_specs(self.shared_kin_specs, config.geom_prior_overrides)
		if config.rot_prior_overrides:
			for comp in self.rot_model.components:
				comp.apply_prior_overrides(config.rot_prior_overrides)
		if config.fixed_params:
			self.apply_fixed_params(config.fixed_params)

	def apply_fixed_params(self, fixed_params: dict):
		"""Pin or link parameters by name. Dispatches to the owning model group."""
		from .param_spec import _apply_fixed_to_specs

		morph_names = {s.name for s in self.morph_model.parameters}
		kin_names   = {s.name for s in self.shared_kin_specs}
		rot_names   = {s.name for c in self.rot_model.components for s in c.parameters}

		morph_fp = {k: v for k, v in fixed_params.items() if k in morph_names}
		kin_fp   = {k: v for k, v in fixed_params.items() if k in kin_names}

		if morph_fp:
			self.morph_model.apply_fixed_params(morph_fp)
		if kin_fp:
			_apply_fixed_to_specs(self.shared_kin_specs, kin_fp)
		for comp in self.rot_model.components:
			comp_fp = {k: v for k, v in fixed_params.items()
			           if k in {s.name for s in comp.parameters}}
			if comp_fp:
				comp.apply_fixed_params(comp_fp)

		unknown = set(fixed_params) - morph_names - kin_names - rot_names
		if unknown:
			raise ValueError(f"fixed_params references unknown parameter(s): {sorted(unknown)}")


	def sample_morphology_params(self, include_amplitude=True):
		"""
		Sample morphological parameters. Returns a dict.

		Parameters
		----------
		include_amplitude : bool, optional
			If True (default), sample amplitude. Set False for multi-obs.

		Returns
		-------
		amplitude : float or None
			Flux normalization (None if include_amplitude=False)
		r_eff : float
			Effective radius in pixels
		n : float
			Sersic index
		i : float
			Inclination in degrees
		ellip : float
			Ellipticity
		PA_morph : float
			Morphological position angle in degrees
		xc_morph : float
			X-centroid in pixels
		yc_morph : float
			Y-centroid in pixels
		dict
			Morphological parameters dict
		"""
		if include_amplitude:
			return self.morph_model.sample()
		else:
			return self.morph_model.sample_without_amplitude()

	def generate_flux_map(self, morph_params, shared_params):
		"""Generate flux map via morph_model. Both args are dicts."""
		return self.morph_model.generate_flux_map(
			morph_params, shared_params, self.direct_shape[0], self.factor
		)

	def _sample_shared_kinematics(self, morph_params, include_v0=True):
		"""Sample shared kinematic parameters with morph_params as context."""
		from .param_spec import sample_specs
		specs = self.shared_kin_specs if include_v0 else [s for s in self.shared_kin_specs if s.name != 'v0']
		return sample_specs(specs, context=morph_params)

	def sample_rot_params(self, morph_params):
		"""Sample rotation curve parameters."""
		return self.rot_model.sample(morph_params)

	def velocity_field(self, X, Y, PA, i, all_params):
		"""Line-of-sight velocity field for any CompositeRotationCurve.
		Deprojection is identical to _v_core; only the 1D velocity profile is replaced
		by rot_model.rotation_curve(), so any rotation component works without changes here.
		"""
		i_rad  = i  / 180.0 * jnp.pi
		PA_rad = PA / 180.0 * jnp.pi
		sini = jnp.sin(i_rad)
		cosi = jnp.cos(i_rad)

		x_rot = X * jnp.cos(PA_rad) - Y * jnp.sin(PA_rad)
		y_rot = X * jnp.sin(PA_rad) + Y * jnp.cos(PA_rad)

		cosi_safe = jnp.where(cosi != 0, cosi, 1e-6)
		r_squared = x_rot**2 / cosi_safe**2 + y_rot**2
		r = jnp.sqrt(jnp.where(r_squared != 0, r_squared, 1e-12))
		r_safe = jnp.where((x_rot != 0) | (y_rot != 0), r, 1e-6)

		v_circ = jnp.where(
			cosi != 0,
			self.rot_model.rotation_curve(r_safe, all_params) * sini,
			self.rot_model.rotation_curve(jnp.abs(y_rot), all_params),
		)
		return jnp.where(r_safe != 0, v_circ * (y_rot / r_safe), 0.0)

	def compute_posterior_means_parametric(self, inference_data):
		"""
			Retreive the best sample from the MCMC chains for the main disk variables
		"""

		self.PA_mean = jnp.array(inference_data.posterior['PA'].median(dim=["chain", "draw"]))
		self.y0_vel_mean = jnp.array(inference_data.posterior['y0_vel'].median(dim=["chain", "draw"]))
		self.x0_vel_mean = jnp.array(inference_data.posterior['x0_vel'].median(dim=["chain", "draw"]))
		self.sigma0_mean_model = jnp.array(inference_data.posterior['sigma0'].median(dim=["chain", "draw"]))

		self.PA_16 = jnp.array(inference_data.posterior['PA'].quantile(0.16, dim=["chain", "draw"]))
		self.PA_84 = jnp.array(inference_data.posterior['PA'].quantile(0.84, dim=["chain", "draw"]))
		self.sigma0_16 = jnp.array(inference_data.posterior['sigma0'].quantile(0.16, dim=["chain", "draw"]))
		self.sigma0_84 = jnp.array(inference_data.posterior['sigma0'].quantile(0.84, dim=["chain", "draw"]))
		self.x0_vel_16 = jnp.array(inference_data.posterior['x0_vel'].quantile(0.16, dim=["chain", "draw"]))
		self.x0_vel_84 = jnp.array(inference_data.posterior['x0_vel'].quantile(0.84, dim=["chain", "draw"]))
		self.y0_vel_16 = jnp.array(inference_data.posterior['y0_vel'].quantile(0.16, dim=["chain", "draw"]))
		self.y0_vel_84 = jnp.array(inference_data.posterior['y0_vel'].quantile(0.84, dim=["chain", "draw"]))

		# v0 may be absent in multi-obs fits (per-obs v0 instead)
		if 'v0' in inference_data.posterior:
			self.v0_mean = jnp.array(inference_data.posterior['v0'].median(dim=["chain", "draw"]))
			self.v0_16 = jnp.array(inference_data.posterior['v0'].quantile(0.16, dim=["chain", "draw"]))
			self.v0_84 = jnp.array(inference_data.posterior['v0'].quantile(0.84, dim=["chain", "draw"]))
		else:
			self.v0_mean = None
			self.v0_16 = None
			self.v0_84 = None

		# Dynamic rotation parameter extraction — works for any CompositeRotationCurve
		self.rot_means = {}
		self.rot_quantiles = {}
		for spec in self.rot_model.parameters:
			if spec.fixed:
				continue
			name = spec.name
			if name in inference_data.posterior:
				med = jnp.array(inference_data.posterior[name].median(dim=["chain", "draw"]))
				q16 = jnp.array(inference_data.posterior[name].quantile(0.16, dim=["chain", "draw"]))
				q84 = jnp.array(inference_data.posterior[name].quantile(0.84, dim=["chain", "draw"]))
				self.rot_means[name] = med
				self.rot_quantiles[name] = {'16': q16, '84': q84}
				setattr(self, f'{name}_mean', med)
				setattr(self, f'{name}_16', q16)
				setattr(self, f'{name}_84', q84)

	def compute_parametrix_flux_posterior(self, inference_data):
		# Dynamic morphology posterior — works for any MorphologyModel.
		# amplitude absent in multi-obs fits (per-obs amplitude sampled instead).
		self.morph_means = {}
		self.morph_quantiles = {}
		for spec in self.morph_model.parameters:
			if spec.fixed:
				continue
			name = spec.name
			if name not in inference_data.posterior:
				self.morph_means[name] = None
				self.morph_quantiles[name] = {'16': None, '84': None}
				setattr(self, f'{name}_mean', None)
				setattr(self, f'{name}_16', None)
				setattr(self, f'{name}_84', None)
				continue
			med = jnp.array(inference_data.posterior[name].median(dim=["chain", "draw"]))
			q16 = jnp.array(inference_data.posterior[name].quantile(0.16, dim=["chain", "draw"]))
			q84 = jnp.array(inference_data.posterior[name].quantile(0.84, dim=["chain", "draw"]))
			self.morph_means[name] = med
			self.morph_quantiles[name] = {'16': q16, '84': q84}
			setattr(self, f'{name}_mean', med)
			setattr(self, f'{name}_16', q16)
			setattr(self, f'{name}_84', q84)

		# i and ellip are special: ellip is derived from i and added to the trace here.
		num_samples = inference_data.posterior['i'].shape[1]
		num_chains = inference_data.posterior['i'].shape[0]
		num_samples_prior = inference_data.prior['i'].shape[1]

		inference_data.posterior['ellip'] = xr.DataArray(np.zeros((num_chains, num_samples)), dims = ('chain', 'draw'))
		inference_data.prior['ellip'] = xr.DataArray(np.zeros((1, num_samples_prior)), dims = ('chain', 'draw'))
		for i in range(num_chains):
			for sample in range(num_samples-1):
				inference_data.posterior['ellip'][i,int(sample)] = 1 - utils.compute_axis_ratio(inc = float(inference_data.posterior['i'][i,int(sample)].values), q0 = 0.2)

		for sample in range(num_samples_prior-1):
			inference_data.prior['ellip'][0,int(sample)] = 1 - utils.compute_axis_ratio(inc = float(inference_data.prior['i'][0,int(sample)].values), q0 = 0.2)

		self.i_mean = jnp.array(inference_data.posterior['i'].median(dim=["chain", "draw"]))
		self.i_16 = jnp.array(inference_data.posterior['i'].quantile(0.16, dim=["chain", "draw"]))
		self.i_84 = jnp.array(inference_data.posterior['i'].quantile(0.84, dim=["chain", "draw"]))
		self.ellip_mean = jnp.array(inference_data.posterior['ellip'].median(dim=["chain", "draw"]))
		self.ellip_16 = jnp.array(inference_data.posterior['ellip'].quantile(0.16, dim=["chain", "draw"]))
		self.ellip_84 = jnp.array(inference_data.posterior['ellip'].quantile(0.84, dim=["chain", "draw"]))

		# Flux rendering — Sersic-specific; skipped when amplitude is None (multi-obs fit)
		factor = self.factor
		image_shape = self.direct_shape[0]

		if self.morph_means.get('amplitude') is not None:
			amplitude_re_mean = utils.flux_to_Ie(self.amplitude_mean, self.n_mean, self.r_eff_mean, self.ellip_mean)
			x = jnp.linspace(0 - self.xc_morph_mean, image_shape - self.xc_morph_mean - 1, image_shape*factor)
			y = jnp.linspace(0 - self.yc_morph_mean, image_shape - self.yc_morph_mean - 1, image_shape*factor)
			x_grid, y_grid = jnp.meshgrid(x, y)
			self.fluxes_mean_high = utils.sersic_profile(x_grid, y_grid, amplitude_re_mean/factor**2, self.r_eff_mean, self.n_mean, 0.0, 0.0, self.ellip_mean, (90 - self.PA_morph_mean)*jnp.pi/180)
			self.fluxes_mean = utils.resample(self.fluxes_mean_high, factor, factor)
		else:
			self.fluxes_mean_high = None
			self.fluxes_mean = None
		self.fluxes_mean_masked = self.fluxes_mean
		return inference_data
	



class GrismFitter(KinModels):
	"""
		Class for the one component exponential disk model
	"""

	def __init__(self):
		print('GrismFitter created')
		self.var_names = []
		self.labels = []

	def set_bounds(self, im_shape, factor, wave_factor, x0, x0_vel, y0, y0_vel):
		"""
		Set grism-specific configuration for the disk model.

		Parameters
		----------
		im_shape : tuple
			Shape of the image
		factor : int
			Spatial oversampling factor
		wave_factor : int
			Wavelength oversampling factor
		x0, y0 : float
			Morphological centroid positions (pixels)
		x0_vel, y0_vel : float
			Velocity centroid positions (pixels)

		Notes
		-----
		All priors (morphological and kinematic) should be set separately
		using set_priors_from_config() or set_parametric_priors() after this method.
		This method only handles grism configuration and centroid positions.
		"""
		# Set basic configuration
		self.set_main_bounds(factor, wave_factor, x0, x0_vel, y0, y0_vel)

		self.im_shape = im_shape
		self.galaxy_model = GalaxyModel(self.im_shape, self.factor)

		from .param_spec import all_param_specs
		_specs = [s for s in all_param_specs(
		              self.galaxy_model.morph_model,
		              self.galaxy_model.shared_kin_specs,
		              self.galaxy_model.rot_model)
		          if not s.fixed]
		self.var_names = [s.name for s in _specs]
		self.labels    = [s.label for s in _specs]


	def inference_model_parametric(self, grism_object, obs_map, obs_error, mask = None):
		"""
		Single-observation inference (backward compatible).

		This method wraps the multi-observation framework for single observations.
		It creates a GrismObservation with theta_rot=0 (assumes priors are already
		in the grism observation frame) and calls inference_model_parametric_multi().

		Parameters
		----------
		grism_object : Grism
			Grism object for this observation
		obs_map : jax.numpy.ndarray
			Observed 2D grism spectrum
		obs_error : jax.numpy.ndarray
			Error map for observation
		mask : jax.numpy.ndarray, optional
			Source mask (default: None)
		"""
		from .grism import GrismObservation

		# Create a GrismObservation from the inputs
		# theta_rot=0 assumes priors were already set in grism observation frame
		obs = GrismObservation(
			grism=grism_object,
			obs_map=obs_map,
			obs_error=obs_error,
			theta_rot=0.0,  # Priors already in observation frame
			dispersion=grism_object.pupil,
			name='single'
		)

		# Call multi-observation method with single observation
		masks = [mask] if mask is not None else None
		return self.inference_model_parametric_multi([obs], masks=masks)

	def inference_model_parametric_multi(self, observations, masks=None):
		"""
		Joint inference model for multiple grism observations.

		Parameters
		----------
		observations : list of GrismObservation
			List of grism observations to fit jointly
		masks : list of jax.numpy.ndarray, optional
			Source masks for each observation (default: None)
			If provided, must be same length as observations

		Notes
		-----
		The galaxy's physical properties (kinematics, morphology) are sampled
		once and shared across all observations. The morphology and kinematics
		are defined in the reference frame used when setting priors
		(typically theta_rot=0 for sky frame).

		For each observation, the PA and centroids are rotated by theta_rot
		to match that observation's orientation.
		"""
		from .grism import GrismObservation

		if not isinstance(observations, list):
			observations = [observations]

		# Handle masks
		if masks is None:
			masks = [None] * len(observations)
		elif len(masks) != len(observations):
			raise ValueError(
				f"Number of masks ({len(masks)}) must match "
				f"number of observations ({len(observations)})"
			)

		print(f"\nFitting {len(observations)} observations jointly:")
		for obs in observations:
			print(f"  - {obs}")

		if len(observations) > 1:
			print("Multi-obs run: v0 and amplitude are sampled independently per observation "
			      "and are not part of the shared parameter spec.")

		# Sample shared galaxy parameters ONCE (in prior reference frame)
		# amplitude and v0 are NOT shared — each observation gets its own
		morph_params = self.galaxy_model.sample_morphology_params(include_amplitude=False)
		shared_params = self.galaxy_model._sample_shared_kinematics(morph_params, include_v0=False)
		rot_params = self.galaxy_model.sample_rot_params(morph_params)

		r_eff = morph_params['r_eff']
		n = morph_params['n']
		i = shared_params['i']
		PA_morph_ref = morph_params['PA_morph']
		xc_morph_ref = morph_params['xc_morph']
		yc_morph_ref = morph_params['yc_morph']
		Pa_ref = shared_params['PA']
		x0_vel_ref = shared_params['x0_vel']
		y0_vel_ref = shared_params['y0_vel']
		sigma0 = shared_params['sigma0']
		ellip = 1.0 - utils.compute_axis_ratio(inc=i, q0=0.2)

		image_shape = self.im_shape[0]
		center = (image_shape - 1) / 2
		n_obs = len(observations)

		# Loop over observations and compute likelihood for each
		for idx, (obs, mask) in enumerate(zip(observations, masks)):

			# Use shared parameter names for single-obs (backward compat with compute_model_parametric)
			# and per-obs names for multi-obs so each observation has independent amplitude/v0
			if n_obs == 1:
				amp_name = 'amplitude'
				v0_name = 'v0'
			else:
				amp_name = f'amplitude_{obs.name}'
				v0_name = f'v0_{obs.name}'

			# Per-observation amplitude (different sensitivity curves and flux calibration)
			amp_mu = self.galaxy_model.amplitude_mu
			amp_std = self.galaxy_model.amplitude_std
			unscaled_amplitude_obs = numpyro.sample(
				f'unscaled_{amp_name}',
				dist.TruncatedNormal(low=(0.0 - amp_mu) / amp_std)
			)
			amplitude_obs = numpyro.deterministic(amp_name, unscaled_amplitude_obs * amp_std + amp_mu)

			# Per-observation v0 (different wavelength calibration between R and C)
			unscaled_v0_obs = numpyro.sample(f'unscaled_{v0_name}', dist.Normal())
			v0_obs = numpyro.deterministic(v0_name, unscaled_v0_obs * 200)

			# Apply rotation for this observation
			theta_rot_rad = jnp.radians(obs.theta_rot)

			# Adjust PA for this observation (rotate by -theta_rot)
			PA_morph_obs = PA_morph_ref - obs.theta_rot
			Pa_obs = Pa_ref - obs.theta_rot

			# Rotate morphological centroids for this observation
			xc_morph_obs, yc_morph_obs = utils.rotate_coords(
				xc_morph_ref, yc_morph_ref,
				center, center,
				theta_rot_rad
			)

			# Rotate velocity centroids for this observation
			x0_vel_obs, y0_vel_obs = utils.rotate_coords(
				x0_vel_ref, y0_vel_ref,
				center, center,
				theta_rot_rad
			)

			# Generate flux map for this observation with adjusted PA and centroids
			morph_params_obs = dict(morph_params)
			morph_params_obs['amplitude'] = amplitude_obs
			morph_params_obs['PA_morph'] = PA_morph_obs
			morph_params_obs['xc_morph'] = xc_morph_obs
			morph_params_obs['yc_morph'] = yc_morph_obs
			fluxes_high = self.galaxy_model.generate_flux_map(morph_params_obs, shared_params)

			# Build velocity coordinate grids using direct high-res method (Gemini's suggestion)
			X_grid = jnp.linspace(0 - x0_vel_obs, image_shape - x0_vel_obs - 1, image_shape * obs.grism.factor)
			Y_grid = jnp.linspace(0 - y0_vel_obs, image_shape - y0_vel_obs - 1, image_shape * obs.grism.factor)
			X_grid, Y_grid = jnp.meshgrid(X_grid, Y_grid)

			# Compute velocity field with adjusted PA using the composable rotation model
			all_params = {**morph_params_obs, **rot_params}
			velocities = jnp.asarray(self.galaxy_model.velocity_field(X_grid, Y_grid, Pa_obs, i, all_params))
			velocities_scaled = velocities + v0_obs

			# Compute dispersion field
			dispersions = sigma0 * jnp.ones_like(velocities_scaled)

			# Disperse through grism
			model_map = obs.grism.disperse(fluxes_high, velocities_scaled, dispersions)
			model_map = utils.resample(model_map, obs.grism.factor, self.wave_factor)

			# Apply mask if provided
			obs_error_masked = obs.obs_error if mask is None else jnp.where(
				mask == 1, obs.obs_error, 1e6
			)

			# Add likelihood for this observation
			numpyro.sample(
				f'obs_{obs.name}',
				dist.Normal(model_map, obs_error_masked),
				obs=obs.obs_map
			)

	def compute_model_parametric(self, inference_data, grism_object):
		"""

		Function used to post-process the MCMC samples and plot results from the model

		"""

		self.galaxy_model.compute_posterior_means_parametric(inference_data)
		gm = self.galaxy_model
		self.PA_mean           = gm.PA_mean
		self.sigma0_mean_model = gm.sigma0_mean_model
		self.y0_vel_mean       = gm.y0_vel_mean
		self.x0_vel_mean       = gm.x0_vel_mean
		self.v0_mean           = gm.v0_mean
		self.PA_16             = gm.PA_16
		self.PA_84             = gm.PA_84
		self.sigma0_16         = gm.sigma0_16
		self.sigma0_84         = gm.sigma0_84
		self.x0_vel_16         = gm.x0_vel_16
		self.x0_vel_84         = gm.x0_vel_84
		self.y0_vel_16         = gm.y0_vel_16
		self.y0_vel_84         = gm.y0_vel_84
		self.v0_16             = gm.v0_16
		self.v0_84             = gm.v0_84
		self.rot_means         = gm.rot_means
		self.rot_quantiles     = gm.rot_quantiles
		for _name in gm.rot_means:
			setattr(self, f'{_name}_mean', gm.rot_means[_name])
			setattr(self, f'{_name}_16',   gm.rot_quantiles[_name]['16'])
			setattr(self, f'{_name}_84',   gm.rot_quantiles[_name]['84'])

		inference_data = self.galaxy_model.compute_parametrix_flux_posterior(inference_data)
		gm = self.galaxy_model
		self.fluxes_mean      = gm.fluxes_mean
		self.fluxes_mean_high = gm.fluxes_mean_high
		self.morph_means      = gm.morph_means
		self.morph_quantiles  = gm.morph_quantiles
		for _name, _val in gm.morph_means.items():
			setattr(self, f'{_name}_mean', _val)
			setattr(self, f'{_name}_16', gm.morph_quantiles[_name]['16'])
			setattr(self, f'{_name}_84', gm.morph_quantiles[_name]['84'])
		self.i_mean    = gm.i_mean;    self.i_16    = gm.i_16;    self.i_84    = gm.i_84
		self.ellip_mean = gm.ellip_mean; self.ellip_16 = gm.ellip_16; self.ellip_84 = gm.ellip_84
		self.model_flux = self.fluxes_mean_high

		image_shape =  self.im_shape[0]
		# Create velocity coordinate grids using direct high-res method (Gemini's suggestion)
		X_grid = jnp.linspace(0 - self.x0_vel_mean, image_shape - self.x0_vel_mean - 1, image_shape * grism_object.factor)
		Y_grid = jnp.linspace(0 - self.y0_vel_mean, image_shape - self.y0_vel_mean - 1, image_shape * grism_object.factor)
		X_grid, Y_grid = jnp.meshgrid(X_grid, Y_grid)

		_all_params_mean = {**{k: v for k, v in self.morph_means.items() if v is not None}, **self.rot_means}
		self.model_velocities = jnp.asarray(self.galaxy_model.velocity_field(X_grid, Y_grid, self.PA_mean, self.i_mean, _all_params_mean))
		# self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

		self.model_velocities = self.model_velocities  + self.v0_mean

		self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities) #self.sigma0_mean_model *jnp.ones_like(self.model_velocities)

		self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
		# self.model_map_high = grism_object.disperse(self.convolved_fluxes, self.convolved_velocities, self.convolved_dispersions)

		self.model_map = utils.resample(self.model_map_high, grism_object.factor, self.wave_factor)
		# print('Model vels:', self.model_velocities)
		#compute velocity grid in flux image resolution for plotting velocity maps
		self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='linear')
		# print(self.fluxes_mean)
		# Fix: Create a dedicated mask for the kinematics and apply it using np.nan
		vel_mask = np.where(self.fluxes_mean > 0.01 * self.fluxes_mean.max(), 1.0, np.nan)
		# self.model_velocities_low = np.where(self.fluxes_mean == 0, np.nan, self.model_velocities_low)
		# self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='linear')
		# self.model_dispersions_low = jnp.where(self.fluxes_mean == 0, np.nan, self.model_dispersions_low)
		self.model_velocities_low = np.where(np.isnan(vel_mask), np.nan, self.model_velocities_low)
		
		self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='linear')
		self.model_dispersions_low = jnp.where(np.isnan(vel_mask), np.nan, self.model_dispersions_low)
		return inference_data, self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions

	def compute_model_parametric_multi(self, inference_data, observations):
		"""
		Post-process MCMC samples for multiple observations.

		Computes posterior statistics (shared across observations) and generates
		model predictions for each observation with appropriate rotations.

		Parameters
		----------
		inference_data : arviz.InferenceData
			MCMC results from multi-observation fit
		observations : list of GrismObservation
			List of observations that were fit

		Returns
		-------
		inference_data : arviz.InferenceData
			Updated inference data
		results : dict
			Dictionary with keys being observation names, values being dicts with:
			- 'model_map': 2D grism model prediction
			- 'model_flux': High-res flux map
			- 'fluxes_mean': Low-res flux map
			- 'model_velocities': High-res velocity field
			- 'model_dispersions': High-res dispersion field
			- 'model_velocities_low': Low-res velocity field
			- 'model_dispersions_low': Low-res dispersion field
		"""
		from .grism import GrismObservation

		if not isinstance(observations, list):
			observations = [observations]

		# Compute posterior statistics (shared across all observations)
		self.galaxy_model.compute_posterior_means_parametric(inference_data)
		gm = self.galaxy_model
		self.PA_mean           = gm.PA_mean
		self.sigma0_mean_model = gm.sigma0_mean_model
		self.y0_vel_mean       = gm.y0_vel_mean
		self.x0_vel_mean       = gm.x0_vel_mean
		self.v0_mean           = gm.v0_mean
		self.PA_16             = gm.PA_16
		self.PA_84             = gm.PA_84
		self.sigma0_16         = gm.sigma0_16
		self.sigma0_84         = gm.sigma0_84
		self.x0_vel_16         = gm.x0_vel_16
		self.x0_vel_84         = gm.x0_vel_84
		self.y0_vel_16         = gm.y0_vel_16
		self.y0_vel_84         = gm.y0_vel_84
		self.v0_16             = gm.v0_16
		self.v0_84             = gm.v0_84
		self.rot_means         = gm.rot_means
		self.rot_quantiles     = gm.rot_quantiles
		for _name in gm.rot_means:
			setattr(self, f'{_name}_mean', gm.rot_means[_name])
			setattr(self, f'{_name}_16',   gm.rot_quantiles[_name]['16'])
			setattr(self, f'{_name}_84',   gm.rot_quantiles[_name]['84'])

		# Extract per-observation v0 and amplitude posteriors
		self.v0_per_obs = {}
		self.amplitude_per_obs = {}
		for obs in observations:
			obs_name = obs.name
			v0_key = f'v0_{obs_name}'
			amp_key = f'amplitude_{obs_name}'
			if v0_key in inference_data.posterior:
				self.v0_per_obs[obs_name] = {
					'mean': jnp.array(inference_data.posterior[v0_key].median(dim=["chain", "draw"])),
					'16': jnp.array(inference_data.posterior[v0_key].quantile(0.16, dim=["chain", "draw"])),
					'84': jnp.array(inference_data.posterior[v0_key].quantile(0.84, dim=["chain", "draw"])),
				}
			if amp_key in inference_data.posterior:
				self.amplitude_per_obs[obs_name] = {
					'mean': jnp.array(inference_data.posterior[amp_key].median(dim=["chain", "draw"])),
					'16': jnp.array(inference_data.posterior[amp_key].quantile(0.16, dim=["chain", "draw"])),
					'84': jnp.array(inference_data.posterior[amp_key].quantile(0.84, dim=["chain", "draw"])),
				}

		# Compute morphology posterior
		inference_data = self.galaxy_model.compute_parametrix_flux_posterior(inference_data)
		gm = self.galaxy_model
		self.fluxes_mean      = gm.fluxes_mean
		self.fluxes_mean_high = gm.fluxes_mean_high
		self.morph_means      = gm.morph_means
		self.morph_quantiles  = gm.morph_quantiles
		for _name, _val in gm.morph_means.items():
			setattr(self, f'{_name}_mean', _val)
			setattr(self, f'{_name}_16', gm.morph_quantiles[_name]['16'])
			setattr(self, f'{_name}_84', gm.morph_quantiles[_name]['84'])
		self.i_mean    = gm.i_mean;    self.i_16    = gm.i_16;    self.i_84    = gm.i_84
		self.ellip_mean = gm.ellip_mean; self.ellip_16 = gm.ellip_16; self.ellip_84 = gm.ellip_84

		image_shape = self.im_shape[0]
		center = (image_shape - 1) / 2

		# Generate model predictions for each observation
		results = {}

		for obs in observations:
			print(f"\nGenerating model for observation: {obs.name}")

			# Get per-obs amplitude and v0 means
			obs_amplitude_mean = self.amplitude_per_obs[obs.name]['mean'] if obs.name in self.amplitude_per_obs else self.amplitude_mean
			obs_v0_mean = self.v0_per_obs[obs.name]['mean'] if obs.name in self.v0_per_obs else self.v0_mean

			# Apply rotation for this observation
			theta_rot_rad = jnp.radians(obs.theta_rot)

			# Adjust PA and centroids for this observation
			PA_morph_obs = self.PA_morph_mean - obs.theta_rot
			Pa_obs = self.PA_mean - obs.theta_rot

			# Rotate morphological centroids
			xc_morph_obs, yc_morph_obs = utils.rotate_coords(
				self.xc_morph_mean, self.yc_morph_mean,
				center, center,
				theta_rot_rad
			)

			# Rotate velocity centroids
			x0_vel_obs, y0_vel_obs = utils.rotate_coords(
				self.x0_vel_mean, self.y0_vel_mean,
				center, center,
				theta_rot_rad
			)

			# Generate flux map for this observation using per-obs amplitude
			morph_params_obs = {
				'amplitude': obs_amplitude_mean,
				'r_eff': self.r_eff_mean,
				'n': self.n_mean,
				'PA_morph': PA_morph_obs,
				'xc_morph': xc_morph_obs,
				'yc_morph': yc_morph_obs,
			}
			shared_params_mean = {'i': self.i_mean}
			model_flux = self.galaxy_model.generate_flux_map(morph_params_obs, shared_params_mean)

			# Build velocity coordinate grids using direct high-res method (Gemini's suggestion)
			X_grid = jnp.linspace(0 - x0_vel_obs, image_shape - x0_vel_obs - 1, image_shape * obs.grism.factor)
			Y_grid = jnp.linspace(0 - y0_vel_obs, image_shape - y0_vel_obs - 1, image_shape * obs.grism.factor)
			X_grid, Y_grid = jnp.meshgrid(X_grid, Y_grid)

			# Compute velocity and dispersion fields using per-obs v0
			_all_params_mean = {**{k: v for k, v in self.morph_means.items() if v is not None}, **self.rot_means}
			model_velocities = jnp.asarray(self.galaxy_model.velocity_field(X_grid, Y_grid, Pa_obs, self.i_mean, _all_params_mean))
			model_velocities = model_velocities + obs_v0_mean
			model_dispersions = self.sigma0_mean_model * jnp.ones_like(model_velocities)

			# Generate grism model
			model_map_high = obs.grism.disperse(model_flux, model_velocities, model_dispersions)
			model_map = utils.resample(model_map_high, obs.grism.factor, self.wave_factor)

			# Downsample for plotting
			model_velocities_low = image.resize(model_velocities, (int(model_velocities.shape[0]/obs.grism.factor), int(model_velocities.shape[1]/obs.grism.factor)), method='linear')
			model_dispersions_low = image.resize(model_dispersions, (int(model_dispersions.shape[0]/obs.grism.factor), int(model_dispersions.shape[1]/obs.grism.factor)), method='linear')

			# Downsample flux map for this observation
			fluxes_mean = utils.resample(model_flux, self.galaxy_model.factor, self.galaxy_model.factor)
			# # Apply masking to flux map (mask out low-flux regions, similar to compute_parametrix_flux_posterior)
			# fluxes_mean_masked = jnp.where(fluxes_mean > 0.01 * fluxes_mean.max(), fluxes_mean, 0.0)
			# # Mask velocity and dispersion maps where flux is zero
			# model_velocities_low = np.where(fluxes_mean_masked == 0, np.nan, model_velocities_low)
			# model_dispersions_low = jnp.where(fluxes_mean_masked == 0, np.nan, model_dispersions_low)

			# Fix: Create a mask ONLY for the kinematics
			vel_mask = np.where(fluxes_mean > 0.01 * fluxes_mean.max(), 1.0, np.nan)
			
			# Mask velocity and dispersion maps
			model_velocities_low = np.where(np.isnan(vel_mask), np.nan, model_velocities_low)
			model_dispersions_low = jnp.where(np.isnan(vel_mask), np.nan, model_dispersions_low)

			# Store results for this observation
			results[obs.name] = {
				'model_map': model_map,
				'model_flux': model_flux,
				'fluxes_mean': fluxes_mean, #fluxes_mean_masked,  # Store masked flux for plotting
				'model_velocities': model_velocities,
				'model_dispersions': model_dispersions,
				'model_velocities_low': model_velocities_low,
				'model_dispersions_low': model_dispersions_low,
				'observation': obs,  # Keep reference to observation
				'v0_mean': obs_v0_mean,
				'amplitude_mean': obs_amplitude_mean,
			}

		# Store first observation's results as default (for backward compatibility)
		first_obs_name = observations[0].name
		self.model_map = results[first_obs_name]['model_map']
		self.model_flux = results[first_obs_name]['model_flux']
		self.fluxes_mean = results[first_obs_name]['fluxes_mean']  # Already masked
		self.model_velocities = results[first_obs_name]['model_velocities']
		self.model_dispersions = results[first_obs_name]['model_dispersions']
		self.model_velocities_low = results[first_obs_name]['model_velocities_low']  # Already masked
		self.model_dispersions_low = results[first_obs_name]['model_dispersions_low']  # Already masked

		return inference_data, results

	def compute_model(self,inference_data, grism_object, parametric = False):
		"""

		Function used to post-process the MCMC samples and plot results from the model

		"""

		if parametric:
			return self.compute_model_parametric(inference_data, grism_object)
		else:
			raise NotImplementedError('Non-parametric flux model not implemented yet for GrismFitter')

	def log_likelihood(self, grism_object, obs_map, obs_error, values = {}):
		Pa = values['PA']
		i = values['i']
		Va = values['Va']
		r_t = values['r_t']
		sigma0 = values['sigma0']

		fluxes = jnp.where(self.mask ==1, self.flux_prior, 0.0)

		fluxes_high = utils.oversample(fluxes, grism_object.factor, grism_object.factor)

		image_shape = fluxes.shape[0]
		# print(image_shape//2)
		x_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*grism_object.factor)
		y_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*grism_object.factor)
		X_10, Y_10 = jnp.meshgrid(x_10,y_10)

		
		velocities = jnp.array(self.v(X_10, Y_10, Pa, i, Va, r_t))


		velocities_scaled = velocities

		dispersions = sigma0*jnp.ones_like(velocities_scaled)

		self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)


		self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, self.wave_factor)


		mask_obs = jnp.where(obs_map/obs_error > 5, 1, 0)
		model_mask = jnp.where(mask_obs == 1, self.model_map, 0.0)
		obs_mask = jnp.where(mask_obs == 1, obs_map, 0.0)
		obs_error_mask = jnp.where(mask_obs == 1, obs_error, 1e6)

		#compute the gaussian likelihood for the model
		log_likelihood = dist.Normal(model_mask, obs_error_mask).log_prob(obs_mask)

		log_likelihood_sum = jnp.sum(log_likelihood)

		print('Log likelihood: ', log_likelihood_sum)

		return log_likelihood_sum
	
	def log_prior(self, values = {}):
		Pa = values['PA']
		i = values['i']
		Va = values['Va']
		r_t = values['r_t']
		sigma0 = values['sigma0']
		fluxes = values['fluxes']
		fluxes_errors = values['fluxes_error']

		log_prior_PA = dist.TruncatedNormal(Pa, 5,low = -10,high = 100).log_prob(self.mu_PA)
		log_prior_i = dist.TruncatedNormal(i, 5,low = 0,high = 90).log_prob(self.mu_i)
		log_prior_Va = dist.Uniform(self.Va_min, self.Va_max).log_prob(Va)
		log_prior_r_t = dist.Normal(0,4).log_prob(r_t)
		log_prior_sigma0 = dist.Uniform(0, 400).log_prob(sigma0)
		log_prior_fluxes = dist.Normal(fluxes,fluxes_errors).log_prob(self.flux_prior)
		log_prior_fluxes_tot = jnp.sum(log_prior_fluxes)
		log_prior = log_prior_PA + log_prior_i + log_prior_Va + log_prior_r_t + log_prior_sigma0 + log_prior_fluxes_tot

		print('Log prior: ', log_prior)

		return log_prior
	
	def log_posterior(self, grism_object, obs_map, obs_error,values = {}):
		return -(self.log_likelihood(grism_object, obs_map, obs_error,values) + self.log_prior(values))
	def plot_summary(self, obs_map, obs_error, inf_data, wave_space, save_to_folder = None, name = None, v_re = None, PA = None, i = None, Va = None, r_t = None, sigma0 = None, obs_radius = None, ellip = None, theta_obs = None, theta_Ha =None, n = None, save_runs_path = None, ID = None):
		obs_radius = self.r_eff_mean
		ellip = self.ellip_mean
		theta_Ha = self.PA_morph_mean/(180/jnp.pi) + jnp.pi/2 #need to convert to radians and match plotting ref frame
		n = self.n_mean

		ymin,ymax = plotting.plot_disk_summary(obs_map, self.model_map, obs_error, self.model_velocities_low, self.model_dispersions_low, v_re, self.fluxes_mean, inf_data, wave_space, x0 = self.x0_vel_mean, y0 = self.y0_vel_mean, factor = 1, direct_image_size = self.im_shape[0], save_to_folder = save_to_folder, name = name, PA = PA, i = i, Va = Va, r_t = r_t, sigma0 = sigma0, obs_radius = obs_radius, ellip = ellip, theta_obs = theta_obs, theta_Ha =theta_Ha, n = n, save_runs_path  = save_runs_path, ID = ID, galaxy_model = self.galaxy_model)
		return ymin, ymax



