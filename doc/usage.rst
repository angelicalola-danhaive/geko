Usage Guide
===========

Quick Start
-----------

Basic Workflow
^^^^^^^^^^^^^^

The typical Geko workflow involves three main steps:

1. **Data Preprocessing**: Load and prepare grism data
2. **MCMC Fitting**: Run Bayesian inference
3. **Visualization**: Analyze and plot results

Example: Fitting a Single Galaxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a minimal example of fitting a galaxy with Geko:

.. code-block:: python

   from geko.fitting import run_geko_fit
   import arviz as az

   # Run the full fitting pipeline
   inference_data = run_geko_fit(
       output='test_output',
       master_cat='path/to/master_catalog.cat',
       line='Ha',
       parametric=True,
       save_runs_path='./saves/',
       num_chains=2,
       num_warmup=500,
       num_samples=1000,
       source_id=191250,
       field='manual',
       grism_filter='F444W',
       manual_psf_name='mpsf_jw018950.gs.f444w.fits',
       manual_theta_rot=0.0,
       manual_pysersic_file='summary_191250_image_F150W_svi.cat',
       manual_grism_file='spec_2d_FRESCO_F444W_ID191250_comb.fits'
   )

   # Print summary statistics
   print(az.summary(inference_data))

Configuration-Based Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control, you can use custom configuration files:

.. code-block:: python

   from geko.config import FitConfiguration, MorphologyPriors, KinematicPriors
   from geko.fitting import run_geko_fit

   # Create custom configuration
   config = FitConfiguration(
       morphology=MorphologyPriors(
           PA=0.0,
           PA_sigma=10.0,
           inc=45.0,
           inc_sigma=5.0
       ),
       kinematics=KinematicPriors(
           Va_min=10.0,
           Va_max=500.0,
           sigma0_min=10.0,
           sigma0_max=300.0
       )
   )

   # Run with configuration
   inference_data = run_geko_fit(
       output='configured_fit',
       source_id=12345,
       config=config,
       # ... other parameters ...
   )

Field Options
^^^^^^^^^^^^^

Geko supports multiple field configurations:

**Predefined Fields** (e.g., GOODS-S-FRESCO):

.. code-block:: python

   inference_data = run_geko_fit(
       field='GOODS-S-FRESCO',
       # ... other parameters ...
   )

**Manual Field** (custom PSF and files):

.. code-block:: python

   inference_data = run_geko_fit(
       field='manual',
       manual_psf_name='custom_psf.fits',
       manual_theta_rot=0.0,
       manual_pysersic_file='morphology.cat',
       manual_grism_file='grism_spectrum.fits',
       # ... other parameters ...
   )

Output Structure
----------------

Geko creates the following output structure:

.. code-block:: text

   saves/
   └── output_name_ID12345/
       ├── output/
       │   ├── corner_plot.png
       │   ├── disk_summary.png
       │   └── output_name_ID12345_summary.png
       ├── trace.nc  # MCMC chains
       └── all_params.pkl  # Fit parameters

Understanding Results
---------------------

The inference_data object is an Arviz InferenceData object containing:

* **Posterior samples**: MCMC chains for all parameters
* **Log probability**: Model likelihood values
* **Divergences**: MCMC diagnostic information

Access posterior samples:

.. code-block:: python

   import arviz as az

   # Get parameter means
   summary = az.summary(inference_data)
   print(summary)

   # Access specific parameters
   pa_samples = inference_data.posterior['PA'].values
   va_samples = inference_data.posterior['Va'].values

Advanced Topics
---------------

Custom Priors
^^^^^^^^^^^^^

Override specific priors using the configuration system:

.. code-block:: python

   from geko.config import FitConfiguration, KinematicPriors
   from numpyro import distributions as dist

   config = FitConfiguration(
       kinematics=KinematicPriors(
           Va_min=50.0,
           Va_max=300.0,
           # Other parameters use defaults
       )
   )

GPU Acceleration
^^^^^^^^^^^^^^^^

Geko automatically uses GPU if JAX detects CUDA:

.. code-block:: python

   import jax
   print(f"Using device: {jax.devices()}")

For more examples, see the ``demo/`` directory in the repository.
