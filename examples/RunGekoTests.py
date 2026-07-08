"""
RunGekoTests.py — run mock recovery tests using the geko testing framework.

Tests are defined in a config table (default: testing/mock_results) and run
via the geko.testing.run_mock pipeline. Results are written to testing/<folder>/.

Run from your project directory:
    python examples/RunGekoTests.py --test kin_test2 --parametric 1 --PA 75
"""
import argparse
import os
import sys
import numpy as np
from astropy.table import Table

from geko.testing.run_mock import run_test, read_config_table

from geko import utils



#main part

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='testing/mock_results',
					help='config table path') #config table name
parser.add_argument('--test', type=str, default='',
					help='name of the test running') #test name
parser.add_argument('--parametric', type=int, default=0,
					help='parametric or non-parametric flux model') #parametric or non-parametric model
parser.add_argument('--ideal', type=int, default=0,
					help='run in ideal mode: no PSF/LSF convolution and no noise (1=ideal, 0=realistic)')
parser.add_argument('--PA', type=float, default=None,
					help='override PA_image and PA_grism for all rows (degrees); if not set, uses config table values')
parser.add_argument('--psf_mode', type=str, default='2d', choices=['1d', '2d'],
					help='PSF convolution mode for mock: 2d=standard (both axes), 1d=only spatial y-axis')
parser.add_argument('--num_chains',  type=int, default=2,
					help='number of MCMC chains')
parser.add_argument('--num_warmup',  type=int, default=1000,
					help='number of MCMC warmup steps')
parser.add_argument('--num_samples', type=int, default=1000,
					help='number of MCMC samples per chain')
parser.add_argument('--inference_psf', type=str, default='2d', choices=['2d', 'rank1'],
					help='PSF used in inference forward model: 2d=full PSF (default), rank1=rank-1 SVD approximation')
parser.add_argument('--inference_lsf', action='store_true', default=False,
					help='Include LSF broadening in inference forward model (mock is always LSF-free)')
# parser.add_argument('--psf_path', type=str, default='gdn_mpsf_F356W_small.fits',
#                     help='psf file path') #psf file path --> add the default here

if __name__ == "__main__":
	
	args = parser.parse_args()
	config_path = args.config
	test = args.test
	parametric = bool(args.parametric)
	ideal = bool(args.ideal)
	PA_override = args.PA
	psf_mode    = args.psf_mode
	num_chains    = args.num_chains
	num_warmup    = args.num_warmup
	num_samples   = args.num_samples
	inference_psf = args.inference_psf
	inference_lsf = args.inference_lsf
	# psf_path = args.psf_path

	if parametric:
		print('Running parametric model')
	else:
		print('Running non-parametric model')
	if ideal:
		print('Running in ideal mode: no PSF/LSF convolution, no noise')
	if psf_mode == '1d':
		print('Running with 1D PSF + LSF (spatial y-axis only) for both mock and inference — self-consistent test')

	PA_image, PA_grism, i, sigma0, SN_image, SN_grism, n, params_dict = read_config_table(config_path, test)

	# Load PSF - for ideal mode, create idealized Gaussian PSF
	if ideal:
		# Create idealized Gaussian PSF (better than real instrument, but not too sharp)
		# Real NIRCam F444W: FWHM ~ 0.15" ~ 2.4 pixels at 0.063"/pix (sigma ~ 1.0 pixel)
		# Idealized: sigma = 0.5 pixels (FWHM ~ 1.2 pixels, 2x better than real)
		ideal_sigma = 0.25  # pixels
		psf_size = 11  # 11x11 grid to capture ~10 sigma (>99.9999% of flux)
		psf_center = psf_size // 2
		y_psf, x_psf = np.mgrid[0:psf_size, 0:psf_size]
		psf = np.exp(-((x_psf - psf_center)**2 + (y_psf - psf_center)**2) / (2 * ideal_sigma**2))
		psf = psf / np.sum(psf)  # Normalize
		print(f'Ideal mode: Using idealized Gaussian PSF with sigma={ideal_sigma} pixels (FWHM ~ 1.2 pix), {psf_size}x{psf_size} grid')
	else:
		psf = utils.load_psf(filter = 'F444W', y_factor = 1, size = 9)

	if PA_override is not None:
		print(f'Overriding PA_image and PA_grism to {PA_override} degrees for all rows')
		PA_image = np.full_like(PA_image, PA_override, dtype=float)
		PA_grism = np.full_like(PA_grism, PA_override, dtype=float)
		params_dict['PA_image'] = PA_image
		params_dict['PA_grism'] = PA_grism

	ideal_suffix    = '_ideal' if ideal else ''
	pa_suffix       = f'_pa{int(PA_override)}' if PA_override is not None else ''
	psf_suffix      = '_psf1d' if psf_mode == '1d' else ''
	inf_psf_suffix  = '_infrank1' if inference_psf == 'rank1' else ''
	inf_lsf_suffix  = '_inflsf' if inference_lsf else ''
	save_folder = test + ideal_suffix + pa_suffix + psf_suffix + inf_psf_suffix + inf_lsf_suffix
	os.makedirs('testing/' + save_folder, exist_ok=True)

	# Determine result param names from model — change these to match the model under test
	# Supported rotation components: 'Arctan' (more to come)
	# Supported morphology models:   'Sersic' (more to come)
	from geko.models import GalaxyModel
	from geko.param_spec import all_param_specs
	from geko.postprocess import DERIVED_QUANTITIES
	from geko.config import FitConfiguration
	import copy
	_default_cfg = FitConfiguration(rotation_components=['Arctan'], morphology_model='Sersic')

	# Prior tests: the config table encodes the PRIOR CENTRE for each row, not the truth.
	# Mock data is always generated from a fixed standard galaxy; only the prior moves.
	#   prior_test1/3 → vary r_eff and r_t prior centre  (truth: r_eff=4.19, r_t=1.0, n=1.0)
	#   prior_test2/4 → vary n prior centre               (truth: r_eff=4.19, r_t=1.0, n=1.0)
	_PRIOR_TEST_TRUTH = {'r_eff': 4.19, 'r_t': 1.0, 'n': 1.0}
	is_prior_test = test.startswith('prior_test')
	if is_prior_test:
		print(f'Prior test detected: mock truth fixed at {_PRIOR_TEST_TRUTH}; '
		      f'config table values are used as prior centres.')
		# Keep originals as prior-centre arrays, then fix truth in params_dict and n
		_prior_r_eff = params_dict['r_eff'].copy()
		_prior_r_t   = params_dict['r_t'].copy()
		_prior_n     = n.copy()
		params_dict  = copy.copy(params_dict)
		params_dict['r_eff'] = np.full_like(params_dict['r_eff'], _PRIOR_TEST_TRUTH['r_eff'])
		params_dict['r_t']   = np.full_like(params_dict['r_t'],   _PRIOR_TEST_TRUTH['r_t'])
		n = np.full_like(n, _PRIOR_TEST_TRUTH['n'])
	_gm = GalaxyModel((31, 31), 5, rot_model=_default_cfg.build_rot_model())
	_specs = all_param_specs(_gm.morph_model, _gm.shared_kin_specs, _gm.rot_model)
	params_single = [s.name for s in _specs if not s.fixed] + [dq.name for dq in DERIVED_QUANTITIES]

	# Config columns: all input columns (no _q16/_q50/_q84 suffixes, not 'test', not 'v_re'
	# which is added separately as the truth column)
	config_params = [c for c in params_dict.keys()
	                 if not any(c.endswith(sfx) for sfx in ('_16', '_50', '_84'))
	                 and c not in ('test', 'v_re')]
	all_params_single = [[p + '_q16', p + '_q50', p + '_q84'] for p in params_single]
	cat_col = config_params + ['v_re'] + [x for triplet in all_params_single for x in triplet]
	n_rows = len(PA_image)
	res = Table({col: np.zeros(n_rows) for col in cat_col})
	for col in config_params:
		try:
			res[col] = params_dict[col].astype(float)
		except (ValueError, TypeError):
			pass


	for j in range(len(PA_image) - 1, -1, -1):
		print('Running test ' + str(test) + ' iteration ' + str(j))
		rot_info = {k: params_dict[k][j] for k in params_dict if k not in
		            ('test', 'PA_image', 'PA_grism', 'i', 'sigma0', 'SN_image', 'SN_grism', 'n')
		            and not any(k.endswith(s) for s in ('_16', '_50', '_84'))}
		print('Parameters: PA_image = ' + str(PA_image[j]) + ', PA_grism = ' + str(PA_grism[j]) +
		      ', i = ' + str(i[j]) + ', sigma0 = ' + str(sigma0[j]) +
		      ', SN_image = ' + str(SN_image[j]) + ', SN_grism = ' + str(SN_grism[j]) +
		      ', rot_params = ' + str(rot_info))

		if n[j] is None or not np.isfinite(float(n[j])) or n[j] <= 0:
			n[j] = 1

		if is_prior_test:
			# Build per-iteration config with prior centres from the original table values
			iter_cfg = FitConfiguration(
			    rotation_components=_default_cfg.rotation_components,
			    morph_prior_overrides={
			        'r_eff_mu': float(_prior_r_eff[j]),
			        'r_eff_std': float(max(3.0, _prior_r_eff[j])),
			        'n_mu': float(_prior_n[j]),
			    },
			)
		else:
			iter_cfg = _default_cfg

		run_test(test, j, config_path, parametric, PA_image, PA_grism, i, sigma0,
		         SN_image, SN_grism, n, psf, params_dict, params_single, res,
		         save_folder=save_folder, psf_mode=psf_mode,
		         num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples,
		         fit_config=iter_cfg, inference_psf=inference_psf,
		         inference_lsf=inference_lsf)

		
	res.write('testing/' + save_folder + '/' + 'results', format='ascii', overwrite=True)