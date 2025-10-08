Geko Documentation
==================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/yourusername/geko/blob/main/LICENSE
   :alt: License

Geko is a Python package for analyzing grism spectroscopy and morphology data from JWST observations.
The package uses JAX for accelerated computation and Numpyro for Bayesian inference to model galaxy
kinematics and morphology.

Key Features
------------

* **JAX-accelerated grism modeling**: Fast forward modeling of 2D dispersed spectra
* **Bayesian inference**: MCMC fitting using Numpyro's NUTS sampler
* **Morphology integration**: Incorporates Sérsic profile fitting from PySersic
* **Flexible configuration**: YAML-based parameter and prior specification
* **Comprehensive visualization**: Diagnostic plots and corner plots for fit results

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Demo & Tutorials

   simple_fit_demo

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_fitting
   api_grism
   api_preprocess
   api_models
   api_config
   api_plotting
   api_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
