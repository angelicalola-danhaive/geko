<div align="center">
  <img src="https://raw.githubusercontent.com/angelicalola-danhaive/geko/main/doc/_static/geko_logo.png" alt="Geko Logo" width="300"/>

  # the **G**rism **E**mission-line **K**inematics t**O**ol

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![CI with Conda + pip](https://github.com/angelicalola-danhaive/geko/actions/workflows/tests.yml/badge.svg)](https://github.com/angelicalola-danhaive/geko/actions/workflows/tests.yml)

</div>

## Description

Geko is a Python package for analyzing grism spectroscopy from JWST NIRCam observations. The package uses JAX for accelerated computation and Numpyro for Bayesian inference to recover emission-line kinematics and morphology from 2D slitless spectroscopy spectra.

**Key Features:**
- **JAX-accelerated**: GPU support for fast inference
- **Bayesian inference**: MCMC fitting using Numpyro's No-U-Turn Sampler (NUTS)
- **Flexible configuration**: Easy-to-use configuration system for priors and MCMC parameters
- **Comprehensive visualization**: Diagnostic plots and corner plots for fit results

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/angelicalola-danhaive/geko.git
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

See the [documentation](https://astro-geko.readthedocs.io) for detailed usage examples and tutorials.

## Citation

If you use Geko in your research, please cite the following paper:

```bibtex

@article{Danhaive:2025ac,
	author = {{Danhaive}, A. Lola and {Tacchella}, Sandro},
	journal = {arXiv e-prints},
	month = oct,
	pages = {arXiv:xxxx},
	title = {{Modelling the kinematics and morphology of galaxies in slitless spectroscopy  with _geko_}},
	year = 2025}
```

## Acknowledgements

We acknowledge support from the Royal Society Research Grants (G125142). We thank Amanda Stoffers for creating our beautiful logo. 

This package makes use of:
- [JAX](https://github.com/google/jax) for accelerated numerical computing
- [Numpyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [Astropy](https://www.astropy.org/) for astronomical data handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
