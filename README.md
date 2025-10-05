<img src="doc/_static/geko_logo.png" alt="Geko Logo" width="200" align="left"/>

<h1 style="display: inline-block;">the <strong>G</strong>rism <strong>E</strong>mission-line <strong>K</strong>inematics t<strong>O</strong>ol</h1>
<br clear="left"/>

<p>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/></a>
</p>

<br clear="left"/>

## Description

Geko is a Python package for analyzing grism spectroscopy and morphology data from JWST observations. The package uses JAX for accelerated computation and Numpyro for Bayesian inference to model galaxy kinematics and morphology from 2D dispersed spectra.

**Key Features:**
- **JAX-accelerated grism modeling**: Fast forward modeling of 2D dispersed spectra with GPU support
- **Bayesian inference**: MCMC fitting using Numpyro's No-U-Turn Sampler (NUTS)
- **Morphology integration**: Incorporates Sérsic profile fitting from PySersic
- **Flexible configuration**: Easy-to-use configuration system for priors and MCMC parameters
- **Comprehensive visualization**: Diagnostic plots and corner plots for fit results

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/geko.git
cd geko

# Create conda environment
conda env create -f environment.yml
conda activate geko_env

# Install geko in development mode
pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/geko.git
cd geko

# Install in editable mode
pip install -e .
```

### Requirements

- Python >= 3.8
- JAX/JAXlib (with optional GPU support)
- Numpyro
- Astropy
- Photutils
- PySersic

For GPU acceleration, follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for CUDA support.

## Quick Start

```python
from geko.fitting import run_geko_fit

# Run a fit with minimal configuration
inference_data = run_geko_fit(
    output='my_fit',
    master_cat='path/to/catalog.cat',
    line='Ha',
    parametric=True,
    save_runs_path='./saves/',
    num_chains=2,
    num_warmup=500,
    num_samples=1000,
    source_id=12345,
    field='GOODS-S-FRESCO',
    grism_filter='F444W'
)
```

See the [documentation](https://geko.readthedocs.io) for detailed usage examples and tutorials.

## Citation

If you use Geko in your research, please cite the following paper:

```bibtex
@article{geko2024,
  author  = {Author, Name and Coauthor, Name},
  title   = {Geko: A Tool for JWST Grism Emission-line Kinematics},
  journal = {Journal Name},
  year    = {2024},
  volume  = {XXX},
  pages   = {XXX},
  doi     = {XX.XXXX/XXXXX}
}
```

## Acknowledgements

We thank Amanda Stoffers for creating our beautiful logo. 
We acknowledge support from the Royal Society Research Grants.

This package makes use of:
- [JAX](https://github.com/google/jax) for accelerated numerical computing
- [Numpyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [Astropy](https://www.astropy.org/) for astronomical data handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
