"""
RunGeko.py — example script for running single-observation geko fits.

TEMPLATE: adapt the following sections for your survey before running:
  - catalog path and column names (lines marked SURVEY-SPECIFIC below)
  - field names in the field→catalog and field→theta_rot mapping
  - MORPH/GEOM/ROT prior overrides and FIXED_PARAMS in the configuration section

Run from your project directory:
    python examples/RunGeko.py --out_folder MyRuns --start 0 --end 10
"""
from geko.fitting import run_geko_fit
import geko.config as config
import numpyro


import jax
from jax import random
jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(2)

try:
    if jax.devices('gpu'):
        numpyro.set_platform('gpu')
        print("GPU detected - using GPU platform for Numpyro")
except (RuntimeError, Exception):
    print("No GPU available - using CPU platform")

XLA_FLAGS = "--xla_gpu_force_compilation_parallelism=1"

import argparse
from astropy.table import Table
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# ============================================================================
# USER CONFIGURATION SECTION - CUSTOMIZE YOUR PRIORS HERE
# ============================================================================
# Keys follow the pattern {param_name}_{min|max|mu|std}
# Set a dict to {} to use all defaults for that group.

# Morphology priors: amplitude, r_eff, n, PA_morph, xc_morph, yc_morph
MORPH_PRIOR_OVERRIDES = {
    # 'r_eff_mu': 3.0,   # Effective radius prior centre (pixels)
    # 'r_eff_std': 1.0,  # Effective radius prior width
    # 'n_mu': 1.0,       # Sersic index prior centre (1=exponential, 4=de Vaucouleurs)
    # 'n_std': 0.5,
}

# Geometry / shared kinematics: PA, i, sigma0, x0_vel, y0_vel, v0
GEOM_PRIOR_OVERRIDES = {
    'PA_min': 0,         # Kinematic position angle minimum (degrees)
    'PA_max': 180,       # Kinematic position angle maximum (degrees)
    'i_min': 30,         # Inclination minimum (degrees) - avoid very face-on
    'i_max': 80,         # Inclination maximum (degrees) - avoid edge-on
    'sigma0_min': 20,    # Velocity dispersion minimum (km/s)
    'sigma0_max': 200,   # Velocity dispersion maximum (km/s)
}

# Rotation curve priors: Va, r_t (Arctan); log_M_star (Sersic); log_M_halo, c_halo (NFW)
ROT_PRIOR_OVERRIDES = {
    'Va_min': 50,        # Asymptotic velocity minimum (km/s)
    'Va_max': 500,       # Asymptotic velocity maximum (km/s)
    # 'r_t_mu': 2.0,     # Turnover radius prior centre (pixels)
    # 'r_t_std': 1.0,
}

# EXAMPLE CONFIGURATIONS:
#
# High-redshift galaxies (z>3):
#   GEOM_PRIOR_OVERRIDES = {'i_min': 40, 'i_max': 70, 'sigma0_min': 30, 'sigma0_max': 150}
#   ROT_PRIOR_OVERRIDES  = {'Va_min': 100, 'Va_max': 600}
#
# Edge-on systems:
#   GEOM_PRIOR_OVERRIDES = {'i_min': 60, 'i_max': 85}
#
# Fix inclination to a known value:
#   FIXED_PARAMS = {'i': 60.0}

FIXED_PARAMS = {}
# Example: tie velocity centroids to morphological centroids (removes 2 free parameters)
# FIXED_PARAMS = {
#     'x0_vel': 'xc_morph',
#     'y0_vel': 'yc_morph',
# }

# Model selection — change these lines to switch rotation/morphology model
# Supported rotation components: 'Arctan' (more to come)
# Supported morphology models:   'Sersic' (more to come)
ROTATION_COMPONENTS = ['Arctan']
MORPHOLOGY_MODEL    = 'Sersic'

print("GEKO Configuration:")
print(f"  morph_prior_overrides: {MORPH_PRIOR_OVERRIDES}")
print(f"  geom_prior_overrides:  {GEOM_PRIOR_OVERRIDES}")
print(f"  rot_prior_overrides:   {ROT_PRIOR_OVERRIDES}")
print("=" * 60)

# ============================================================================

def create_geko_config(num_chains, num_warmup, num_samples,
                       morph_overrides=None, geom_overrides=None, rot_overrides=None,
                       fixed_params=None, rotation_components=None, morphology_model=None):
    """Create a FitConfiguration from override dicts."""
    return config.FitConfiguration(
        mcmc=config.MCMCSettings(
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
        ),
        rotation_components=rotation_components or ['Arctan'],
        morphology_model=morphology_model or 'Sersic',
        morph_prior_overrides=morph_overrides or {},
        geom_prior_overrides=geom_overrides or {},
        rot_prior_overrides=rot_overrides or {},
        fixed_params=fixed_params or {},
    )

#command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--line', type=str, default='H_alpha',
		    		help='line to fit')
parser.add_argument('--parametric', action='store_true', default=False,
		    		help='parametric flux model or not')
parser.add_argument('--start', type=int, default=0,
						help = 'start index of the list of IDs to run geko on')
parser.add_argument('--end', type=int, default=100,
						help = 'end index of the list of IDs to run geko on')
parser.add_argument('--out_folder', type=str, default='GekoRuns_v1',
						help = 'folder to save the geko runs in')
parser.add_argument('--num_chains', type=int, default=2,
						help = 'number of chains to run in parallel')
parser.add_argument('--num_warmup', type=int, default=500,
						help = 'number of warmup steps for each chain')
parser.add_argument('--num_samples', type=int, default=500,
						help = 'number of samples to draw from each chain after warmup')

if __name__ == "__main__":
	args = parser.parse_args()
	# # output = args.output + '/'
	# master_cat = args.master_cat
	line = args.line
	parametric = args.parametric
	start = args.start
	end = args.end
	out_folder = args.out_folder + '/'
	num_chains = args.num_chains
	num_warmup = args.num_warmup
	num_samples = args.num_samples

	#load the catalog  # SURVEY-SPECIFIC: replace with your catalog path and columns
	catalog = Table.read('catalogs/Gold_Silver_Unres_FRESCO_CONGRESS.txt', format='ascii')
	if end > catalog['ID'].shape[0]-1:
		print(f"End index {end} exceeds the number of IDs in the catalog {catalog['ID'].shape[0]}. Adjusting to the maximum available index.")
		end = catalog['ID'].shape[0] -1

	list_IDs = catalog['ID'].data[start:end]
	sample = catalog['sample'].data[start:end]

	for output in list_IDs:
		try:
			#choose the mastercat based on the field that the ID belongs to
			output_id = int(output)

			#check if that galaxy has already been run
			out_file = out_folder + str(output_id) + '_results'
			#if the outfile already exists, skip this ID
			if os.path.exists(out_file):
				print(f"Skipping ID {output_id} as results file already exists: {out_file}")
				continue

			# Get the row corresponding to this ID
			match = catalog[catalog['ID'] == output_id]

			if len(match) == 0:
				print(f"ID {output_id} not found in the catalog.")
				continue  # skip to next
			# if match['sample'][0] != 'gold':
			# 	print(f"Skipping ID {output_id} as it is not in the 'gold' sample.")
			# 	continue

			field_value = match['field'][0]
			print('Field value for ID {}: {}'.format(output_id, field_value))

			# Choose catalog  # SURVEY-SPECIFIC: map your field names to master catalog paths
			if field_value in ['GOODS-S-FRESCO', 'GDN-FRESCO']:
				master_cat = 'catalogs/fresco_Ha_cat.txt'
			else:
				master_cat = 'catalogs/congress_Ha_cat'

			# Get redshift from catalog (try different possible column names)
			redshift = None
			possible_z_columns = ['z_spec', 'redshift', 'z', 'zspec', 'z_phot']
			for col in possible_z_columns:
				if col in match.colnames and not match[col].mask[0]:  # Check if column exists and is not masked
					redshift = float(match[col][0])
					break
			
			if redshift is None:
				print(f"Warning: No redshift found for ID {output_id}, using default z=3.0")
				redshift = 3.0  # Default redshift
			
			# Create geko configuration with custom priors
			fit_config = create_geko_config(
				num_chains=num_chains,
				num_warmup=num_warmup,
				num_samples=num_samples,
				morph_overrides=MORPH_PRIOR_OVERRIDES,
				geom_overrides=GEOM_PRIOR_OVERRIDES,
				rot_overrides=ROT_PRIOR_OVERRIDES,
				fixed_params=FIXED_PARAMS,
				rotation_components=ROTATION_COMPONENTS,
				morphology_model=MORPHOLOGY_MODEL,
			)

			print('Running geko for galaxy ID: ', output_id, ' with line: ', line, ' redshift: ', redshift)
			print('Master catalog: ', master_cat, ' parametric: ', parametric)
			fit_config.print_summary()

			# Save configuration file in the output directory for this galaxy
			config_filename = f"geko_config_{output_id}.yaml"
			fit_config.save(config_filename, output_dir=out_folder)

			run_geko_fit(str(output) + '/', master_cat, line, parametric=parametric,
						 save_runs_path=out_folder, num_chains=num_chains, num_warmup=num_warmup,
						 num_samples=num_samples, source_id=output_id, field=field_value,
						 config=fit_config)
			plt.close('all')
		except Exception as e:
			print(f"Error processing ID {output_id}: {e}")
			continue
