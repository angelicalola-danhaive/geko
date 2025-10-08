Usage Guide
===========

Quick Start
-----------

The typical Geko workflow involves preparing your data files, running the MCMC fitting, and analyzing the results.

Minimal Example
^^^^^^^^^^^^^^^

.. code-block:: python

   from geko.fitting import run_geko_fit
   import jax

   # Enable 64-bit precision for JAX
   jax.config.update('jax_enable_x64', True)

   # Run the fit
   inference_data = run_geko_fit(
       output='my_galaxy',
       master_cat='path/to/catalog.cat',
       line='H_alpha',
       parametric=True,
       save_runs_path='./data/',
       num_chains=2,
       num_warmup=500,
       num_samples=1000,
       source_id=191250,
       field='manual',
       manual_psf_name='mpsf_jw018950.gs.f444w.fits',
       manual_theta_rot=0.0,
       manual_pysersic_file='summary_191250_image_F150W_svi.cat',
       manual_grism_file='spec_2d_FRESCO_F444W_ID191250_comb.fits'
   )

   # Print summary
   import arviz as az
   print(az.summary(inference_data))

Required Data Structure
-----------------------

Geko expects files to be organized in a specific directory structure:

.. code-block:: text

   <save_runs_path>/                     # Base directory
   ├── <output_name>/                    # Subfolder for your galaxy/run
   │   └── spec_2d_*_ID<ID>_comb.fits    # 2D grism spectrum (required)
   ├── morph_fits/                       # Morphology directory
   │   └── summary_<ID>_image_F150W_svi.cat  # PySersic fits
   ├── psfs/                             # PSF files directory
   │   ├── mpsf_jw018950.gn.f444w.fits   # GOODS-N PSF
   │   ├── mpsf_jw035770.f356w.fits      # GOODS-N-CONGRESS PSF
   │   └── mpsf_jw018950.gs.f444w.fits   # GOODS-S-FRESCO PSF
   └── catalogs/                         # Optional catalog directory

Required Files
^^^^^^^^^^^^^^

1. **Master Catalog** - ASCII table with source properties (can be anywhere)

   Required columns: ``ID``, ``zspec``, ``<line>_lambda``, ``fit_flux_cgs``, ``fit_flux_cgs_e``

2. **Grism Spectrum** - FITS file with 2D spectrum data

   Location: ``<save_runs_path>/<output_name>/spec_2d_*_ID<source_id>_comb.fits``

3. **PSF File** - Point spread function FITS file

   Location: ``<save_runs_path>/psfs/mpsf_*.fits``

4. **PySersic Morphology** (optional but recommended) - ASCII catalog from PySersic fits

   Location: ``<save_runs_path>/morph_fits/summary_<source_id>_image_F150W_svi.cat``

   Contains: Sersic parameters (n, r_eff, PA, q, x0, y0) used to set morphological priors

Field Options
-------------

Predefined Fields
^^^^^^^^^^^^^^^^^

For standard JWST fields, use predefined field names that automatically set PSF, rotation angles, and file naming:

.. code-block:: python

   inference_data = run_geko_fit(
       field='GOODS-S-FRESCO',  # or 'GOODS-N', 'GOODS-N-CONGRESS'
       source_id=12345,
       # ... other parameters
   )

Predefined fields automatically select:

- **GOODS-N**: PSF ``mpsf_jw018950.gn.f444w.fits``, rotation 230.5°, F444W filter
- **GOODS-N-CONGRESS**: PSF ``mpsf_jw035770.f356w.fits``, rotation 228.2°, F356W filter
- **GOODS-S-FRESCO**: PSF ``mpsf_jw018950.gs.f444w.fits``, rotation 0.0°, F444W filter

Manual Field Mode
^^^^^^^^^^^^^^^^^

For custom data or other fields, use ``field='manual'`` and specify all parameters:

.. code-block:: python

   inference_data = run_geko_fit(
       field='manual',
       manual_psf_name='my_psf.fits',           # PSF filename in psfs/
       manual_theta_rot=45.0,                    # Rotation angle in degrees
       manual_pysersic_file='my_morphology.cat', # PySersic file in morph_fits/
       manual_grism_file='my_spectrum.fits',     # Grism file in output_name/
       # ... other parameters
   )

Configuration System
--------------------

The configuration system allows you to customize priors for your fit.

Scenario 1: With PySersic Fits (Typical)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have PySersic morphology fits, morphological priors are loaded automatically. You can optionally override kinematic priors:

.. code-block:: python

   from geko.config import FitConfiguration, KinematicPriors

   # Override kinematic priors only
   config = FitConfiguration(
       kinematics=KinematicPriors(
           Va_min=50.0,        # Minimum asymptotic velocity (km/s)
           Va_max=300.0,       # Maximum asymptotic velocity (km/s)
           sigma0_min=10.0,    # Minimum velocity dispersion (km/s)
           sigma0_max=150.0    # Maximum velocity dispersion (km/s)
       )
   )

   # Run with custom kinematic priors (morphology from PySersic)
   inference_data = run_geko_fit(
       config=config,
       manual_pysersic_file='summary_12345_image_F150W_svi.cat',
       # ... other parameters
   )

Scenario 2: Without PySersic Fits (Manual)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have PySersic fits, you must provide complete morphological priors:

.. code-block:: python

   from geko.config import FitConfiguration, MorphologyPriors

   # Set all morphology priors manually
   config = FitConfiguration(
       morphology=MorphologyPriors(
           PA_mean=90.0, PA_std=30.0,           # Position angle
           inc_mean=55.0, inc_std=15.0,         # Inclination
           r_eff_mean=3.0, r_eff_std=1.0,       # Effective radius
           r_eff_min=0.5, r_eff_max=10.0,
           n_mean=1.0, n_std=0.5,               # Sersic index
           n_min=0.5, n_max=4.0,
           xc_mean=0.0, xc_std=2.0,             # Centroid x
           yc_mean=0.0, yc_std=2.0,             # Centroid y
           amplitude_mean=100.0, amplitude_std=50.0,
           amplitude_min=1.0, amplitude_max=1000.0
       )
   )

   # Run without PySersic file
   inference_data = run_geko_fit(
       config=config,
       manual_pysersic_file=None,  # Not needed with complete config
       # ... other parameters
   )

Understanding Results
---------------------

The ``inference_data`` object is an ArviZ InferenceData object containing MCMC posterior samples.

Key Parameters
^^^^^^^^^^^^^^

**Kinematic Parameters:**

- ``Va``: Asymptotic rotation velocity (km/s)
- ``sigma0``: Central velocity dispersion (km/s)
- ``v_re``: Rotation velocity at effective radius (derived, km/s)
- ``r_t``: Turnover radius (pixels)

**Morphological Parameters:**

- ``PA``: Position angle (degrees)
- ``inc``: Inclination angle (degrees)
- ``r_eff``: Effective radius (pixels)
- ``n``: Sersic index
- ``xc``, ``yc``: Centroid coordinates (pixels)

Accessing Results
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import arviz as az

   # Get summary statistics (median, std, HDI)
   summary = az.summary(inference_data, hdi_prob=0.68)
   print(summary)

   # Access specific parameter samples
   va_samples = inference_data.posterior['Va'].values
   pa_samples = inference_data.posterior['PA'].values

   # Check convergence (r_hat should be < 1.01)
   print(f"Va r_hat: {summary.loc['Va', 'r_hat']}")

Output Files
------------

Geko saves several files in ``<save_runs_path>/<output_name>/``. **All output files are named using the source ID**, not the folder name:

1. **<source_id>_output** - NetCDF file with full MCMC posterior and prior samples

   Load with: ``az.InferenceData.from_netcdf()``

2. **<source_id>_results** - ASCII table with summary statistics

   Contains median values and 16th/84th percentiles for all parameters

3. **<source_id>_summary.png** - Diagnostic plot showing:

   - Observed 2D spectrum
   - Best-fit model spectrum
   - Residuals
   - 1D velocity and dispersion profiles
   - Rotation curve

4. **<source_id>_v_sigma_corner.png** - Corner plot for v/σ ratio posteriors

5. **<source_id>_summary_corner.png** - Full corner plot for all parameters

Example: Loading Saved Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import arviz as az
   from astropy.table import Table
   import os

   # Define paths
   output_dir = os.path.join(save_runs_path, output_name)

   # Load inference data
   inference_data = az.InferenceData.from_netcdf(
       os.path.join(output_dir, f'{source_id}_output')
   )

   # Load results table
   results = Table.read(
       os.path.join(output_dir, f'{source_id}_results'),
       format='ascii'
   )
   print(results)

MCMC Parameters
---------------

Key MCMC parameters to tune for your fit:

.. code-block:: python

   inference_data = run_geko_fit(
       num_chains=4,        # Number of parallel chains (typically 2-4)
       num_warmup=500,      # Warmup iterations (typically 500-1000)
       num_samples=1000,    # Sampling iterations (typically 1000-2000)
       # ... other parameters
   )

For quick tests, use lower values:

.. code-block:: python

   # Quick test run
   inference_data = run_geko_fit(
       num_chains=1,
       num_warmup=50,
       num_samples=50,
       factor=1,           # Spatial oversampling (default: 5)
       wave_factor=1,      # Wavelength oversampling (default: 10)
       # ... other parameters
   )

GPU Acceleration
----------------

Geko automatically uses GPU if JAX detects CUDA:

.. code-block:: python

   import jax
   print(f"Available devices: {jax.devices()}")
   # If GPU available: [cuda(id=0)]
   # If CPU only: [cpu(id=0)]

For more detailed examples, see the demo notebook at ``doc/simple_fit_demo.ipynb``.
