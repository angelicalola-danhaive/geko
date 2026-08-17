"""
RunGekoMulti.py — example script for jointly fitting multiple grism observations.

TEMPLATE: adapt the following sections for your survey before running:
  - multi_obs_catalog format and column names (ID, grism_file, theta_rot, dispersion, field)
  - field names in the field→catalog mapping
  - MORPH/GEOM/ROT prior overrides and FIXED_PARAMS in the configuration section

The catalog must have one row per observation; galaxies with multiple rows are
fitted both individually (one fit per observation) and jointly.

Run from your project directory:
    python examples/RunGekoMulti.py --multi_obs_catalog my_catalog.txt --out_folder MyMultiRuns
"""
from geko.fitting import run_geko_fit_multi, run_geko_fit
from geko.config import FluxScalingConfig
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
# When PySersic fits are available, morph priors are loaded from those first;
# anything set here takes precedence.

# Morphology priors: amplitude, r_eff, n, PA_morph, xc_morph, yc_morph
MORPH_PRIOR_OVERRIDES = {
    'r_eff_mu': 3.0,
    'r_eff_std': 3.0,
    'r_eff_min': 0.0,
    'r_eff_max': 10.0,
    'n_mu': 1.0,
    'n_std': 0.5,
    'n_min': 0.5,
    'n_max': 4.0,
    'amplitude_mu': 0.005,
    'amplitude_std': 0.005,
    'amplitude_min': 0.0,
    'amplitude_max': 0.1,
    'xc_morph_mu': 15.0,
    'xc_morph_std': 1.0,
    'yc_morph_mu': 15.0,
    'yc_morph_std': 1.0,
    'PA_morph_mu': 10.0,
    'PA_morph_std': 10.0,
}

# Geometry / shared kinematics: PA, i, sigma0, x0_vel, y0_vel, v0
GEOM_PRIOR_OVERRIDES = {
    'PA_mu': 10.0,
    'PA_std': 10.0,
    'i_mu': 60.0,
    'i_std': 10.0,
    'sigma0_min': 0.0,
    'sigma0_max': 200.0,
}

# Rotation curve priors: Va, r_t (Arctan); log_M_star (Sersic); log_M_halo, c_halo (NFW)
ROT_PRIOR_OVERRIDES = {
    'Va_min': -1000.0,
    'Va_max': 1000.0,
}

FIXED_PARAMS = {}
# Example: tie velocity centroids to morphological centroids (removes 2 free parameters)
# FIXED_PARAMS = {
#     'x0_vel': 'xc_morph',
#     'y0_vel': 'yc_morph',
# }

# ============================================================================
# FLUX SCALING — set to None to disable (default Sérsic-only model)
#
# 'row_wise'  : analytical per-row S(y) rescaling in the grism plane.
#               No extra sampled parameters. Only pixels with S/N > 1 contribute.
#               Use this to reduce sensitivity to non-Sérsic emission structure.
#
# 'pixel_wise': sampled 2D log-scale map in the galaxy frame (image_shape × image_shape
#               free parameters) with smoothness regularisation. Scale is shared across
#               all observations and rotated into each observation's frame.
#               sigma_reg   — prior width per pixel (scale freedom, ~30% per pixel)
#               sigma_smooth — smoothness penalty between adjacent pixels (smaller = smoother)
#
# FLUX_SCALING = FluxScalingConfig(mode='row_wise')
# FLUX_SCALING = FluxScalingConfig(mode='pixel_wise', sigma_reg=0.3, sigma_smooth=0.1)
FLUX_SCALING = None
# ============================================================================

# Model selection — change these lines to switch rotation/morphology model
# Supported rotation components: 'Arctan' (more to come)
# Supported morphology models:   'Sersic' (more to come)
ROTATION_COMPONENTS = ['Arctan']
MORPHOLOGY_MODEL    = 'Sersic'

print("GEKO Multi-Observation Configuration:")
print(f"  morph_prior_overrides: {MORPH_PRIOR_OVERRIDES}")
print(f"  geom_prior_overrides:  {GEOM_PRIOR_OVERRIDES}")
print(f"  rot_prior_overrides:   {ROT_PRIOR_OVERRIDES}")
print("=" * 60)

# ============================================================================

def create_geko_config(num_chains, num_warmup, num_samples,
                       morph_overrides=None, geom_overrides=None, rot_overrides=None,
                       fixed_params=None, rotation_components=None, morphology_model=None,
                       flux_scaling=None):
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
        flux_scaling=flux_scaling,
    )


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--line', type=str, default='H_alpha',
                    help='line to fit')
parser.add_argument('--parametric', action='store_true', default=False,
                    help='parametric flux model or not')
parser.add_argument('--start', type=int, default=0,
                    help='start index of the list of IDs to run geko on')
parser.add_argument('--end', type=int, default=100,
                    help='end index of the list of IDs to run geko on')
parser.add_argument('--out_folder', type=str, default='GekoRunsMulti_v1',
                    help='folder to save the geko runs in')
parser.add_argument('--num_chains', type=int, default=2,
                    help='number of chains to run in parallel')
parser.add_argument('--num_warmup', type=int, default=500,
                    help='number of warmup steps for each chain')
parser.add_argument('--num_samples', type=int, default=500,
                    help='number of samples to draw from each chain after warmup')
parser.add_argument('--multi_obs_catalog', type=str, default='catalogs/multi_obs_catalog.txt',
                    help='catalog file with multiple observations per galaxy')
parser.add_argument('--manual_master_cat', type=str, default=None,
                    help='manual master catalog path (for field=manual)')
parser.add_argument('--manual_psf', type=str, default=None,
                    help='manual PSF filename in psfs/ directory (for field=manual)')

if __name__ == "__main__":
    args = parser.parse_args()
    line = args.line
    parametric = args.parametric
    start = args.start
    end = args.end
    out_folder = args.out_folder + '/'
    num_chains = args.num_chains
    num_warmup = args.num_warmup
    num_samples = args.num_samples
    multi_obs_catalog_file = args.multi_obs_catalog
    manual_master_cat = args.manual_master_cat
    manual_psf = args.manual_psf

    # Load the multi-observation catalog
    # Expected columns: ID, grism_file, theta_rot, dispersion, field, z_spec (optional)
    catalog = Table.read(multi_obs_catalog_file, format='ascii')

    # Get unique galaxy IDs
    unique_IDs = set(catalog['ID'].data)
    list_IDs = sorted(list(unique_IDs))[start:end]

    if end > len(list_IDs):
        print(f"End index {end} exceeds the number of unique IDs {len(list_IDs)}. Adjusting.")
        end = len(list_IDs)

    print(f"\nProcessing {len(list_IDs)} galaxies with multi-observation data")
    print(f"Total observations in catalog: {len(catalog)}")
    print("=" * 60)

    for galaxy_id in list_IDs:
        try:
            output_id = int(galaxy_id)

            # Check if that galaxy has already been run
            out_file = out_folder + str(output_id) + '_results'
            # if os.path.exists(out_file):
            #     print(f"Skipping ID {output_id} as results file already exists: {out_file}")
            #     continue

            # Get all observations for this galaxy ID
            galaxy_obs = catalog[catalog['ID'] == output_id]

            if len(galaxy_obs) == 0:
                print(f"ID {output_id} not found in catalog.")
                continue

            # Get field and redshift from first observation (should be same for all)
            field_value = galaxy_obs['field'][0]
            print(f"\nProcessing Galaxy ID {output_id} from field {field_value}")
            print(f"  Number of observations: {len(galaxy_obs)}")

            # Choose master catalog based on field  # SURVEY-SPECIFIC: map your field names to master catalog paths
            if field_value == 'manual':
                if manual_master_cat is None:
                    raise ValueError("--manual_master_cat must be provided when field='manual'")
                master_cat = manual_master_cat
            elif field_value in ['GOODS-S-FRESCO', 'GDN-FRESCO']:
                master_cat = 'catalogs/fresco_Ha_cat.txt'
            else:
                master_cat = 'catalogs/congress_Ha_cat'

            # Get redshift
            redshift = None
            possible_z_columns = ['z_spec', 'redshift', 'z', 'zspec', 'z_phot']
            for col in possible_z_columns:
                if col in galaxy_obs.colnames:
                    try:
                        # Check if column has mask attribute and if value is masked
                        if hasattr(galaxy_obs[col], 'mask') and galaxy_obs[col].mask[0]:
                            continue
                        redshift = float(galaxy_obs[col][0])
                        break
                    except (ValueError, IndexError, TypeError):
                        continue

            if redshift is None:
                print(f"Warning: No redshift found for ID {output_id}, skipping this galaxy.")
                #break  # or continue, depending on whether you want to skip this galaxy
                continue  # Skip this galaxy if no redshift is found

            # Build observations_config list from catalog rows
            observations_config = []
            for i, obs_row in enumerate(galaxy_obs):
                obs_dict = {
                    'grism_file': obs_row['grism_file'],
                    'theta_rot': float(obs_row['theta_rot']),
                    'dispersion': obs_row['dispersion'].strip().upper(),
                    'name': obs_row.get('obs_name', f'obs{i+1}')  # Use obs_name column if exists, else obs1, obs2, etc.
                }
                observations_config.append(obs_dict)
                print(f"    Obs {i+1}: {obs_dict['grism_file']} "
                      f"(θ={obs_dict['theta_rot']:.1f}°, {obs_dict['dispersion']} dispersion)")

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
                flux_scaling=FLUX_SCALING,
            )

            print(f'Running multi-observation geko for galaxy ID: {output_id}')
            print(f'  Line: {line}, Redshift: {redshift:.3f}, Parametric: {parametric}')
            print(f'  Master catalog: {master_cat}')
            fit_config.print_summary()

            # Save configuration file
            config_filename = f"geko_config_{output_id}_multi.yaml"
            fit_config.save(config_filename, output_dir=out_folder)

            # Run individual fits for each observation
            print("\n" + "=" * 60)
            print("Running individual observation fits...")
            print("=" * 60)

            for i, obs_dict in enumerate(observations_config):
                obs_name = obs_dict['name']
                obs_dispersion = obs_dict['dispersion']
                obs_theta_rot = obs_dict['theta_rot']
                obs_grism_file = obs_dict['grism_file']

                # Individual fit output subfolder
                individual_output = str(output_id) + f'/{obs_name}/'

                # Check if individual fit already exists
                individual_out_file = out_folder + individual_output + str(output_id) + '_results'
                # if os.path.exists(individual_out_file):
                #     print(f"  Skipping {obs_name} as results already exist: {individual_out_file}")
                #     continue

                print(f"\n  Fitting {obs_name} individually ({obs_dispersion} @ {obs_theta_rot}°)...")

                try:
                    # Create subdirectory and copy grism file there
                    individual_dir = out_folder + individual_output
                    os.makedirs(individual_dir, exist_ok=True)

                    # Copy grism file to individual subdirectory
                    import shutil
                    source_grism = out_folder + str(output_id) + '/' + obs_grism_file
                    dest_grism = individual_dir + obs_grism_file
                    if not os.path.exists(dest_grism):
                        shutil.copy2(source_grism, dest_grism)

                    run_geko_fit(
                        output=individual_output,
                        master_cat=master_cat,
                        line=line,
                        parametric=parametric,
                        save_runs_path=out_folder,
                        num_chains=num_chains,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        source_id=output_id,  # Keep original ID for catalog lookup
                        field=field_value,
                        config=fit_config,
                        manual_psf_name=manual_psf if field_value == 'manual' else None,
                        manual_theta_rot=obs_theta_rot,
                        manual_grism_file=obs_grism_file
                    )
                    print(f"  ✓ Completed {obs_name} individual fit")
                except Exception as e:
                    print(f"  ✗ Error fitting {obs_name} individually: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            plt.close('all')

            # Run multi-observation fitting
            print("\n" + "=" * 60)
            print("Running multi-observation fit (all observations jointly)...")
            print("=" * 60)

            run_geko_fit_multi(
                observations_config=observations_config,
                output=str(output_id) + '/',
                master_cat=master_cat,
                line=line,
                parametric=parametric,
                save_runs_path=out_folder,
                num_chains=num_chains,
                num_warmup=num_warmup,
                num_samples=num_samples,
                source_id=output_id,
                field=field_value,
                config=fit_config,
                manual_psf_name=manual_psf if field_value == 'manual' else None
            )

            plt.close('all')
            print(f"\n✓ Completed galaxy ID {output_id}")
            print(f"  - Individual fits: {len(observations_config)} observations")
            print(f"  - Multi-observation joint fit: {len(observations_config)} observations")
            print("=" * 60)

        except Exception as e:
            print(f"✗ Error processing ID {output_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("All fitting complete!")
    print("  - Individual observation fits")
    print("  - Multi-observation joint fits")
    print("=" * 60)
