Installation
============

Requirements
------------

Geko requires Python >= 3.8 and has the following core dependencies:

* **JAX/JAXlib**: Accelerated numerical computing
* **Numpyro**: Probabilistic programming and MCMC sampling
* **Astropy**: Astronomical data handling
* **Photutils**: Photometry and source detection
* **PySersic**: Sérsic profile fitting

Installation Methods
--------------------

Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install Geko is using the provided conda environment file:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/geko.git
   cd geko

   # Create conda environment
   conda env create -f environment.yml
   conda activate geko_env

   # Install geko in development mode
   pip install -e .

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For development work, install in editable mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/geko.git
   cd geko

   # Install in editable mode with dev dependencies
   pip install -e .
   pip install pytest  # For running tests

Verifying Installation
----------------------

To verify that Geko is correctly installed:

.. code-block:: python

   import geko
   from geko.grism import Grism
   from geko.fitting import run_geko_fit

   print("Geko successfully imported!")

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   pytest tests/

Optional Dependencies
---------------------

* **GPU Support**: For GPU acceleration, install JAX with CUDA support following the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
* **Visualization**: Additional plotting capabilities may require matplotlib and arviz
