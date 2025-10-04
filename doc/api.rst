API Reference
=============

This page contains the API reference for all public modules and functions in Geko.

Main Fitting Interface
----------------------

.. autofunction:: geko.fitting.run_geko_fit

Grism Modeling
--------------

.. autoclass:: geko.grism.Grism
   :members:
   :special-members: __init__

Preprocessing
-------------

.. autofunction:: geko.preprocess.run_full_preprocessing

Kinematic Models
----------------

.. autoclass:: geko.models.KinModel
   :members:
   :special-members: __init__

.. autoclass:: geko.models.Disk
   :members:
   :special-members: __init__

.. autofunction:: geko.models.set_parametric_priors

Configuration
-------------

.. autoclass:: geko.config.FitConfiguration
   :members:
   :special-members: __init__

.. autoclass:: geko.config.MorphologyPriors
   :members:

.. autoclass:: geko.config.KinematicPriors
   :members:

.. autoclass:: geko.config.MCMCSettings
   :members:

Plotting
--------

.. autofunction:: geko.plotting.plot_disk_summary

Utility Functions
-----------------

.. autofunction:: geko.utils.downsample_error

.. autofunction:: geko.utils.downsample_psf_centered

.. autofunction:: geko.utils.load_psf
