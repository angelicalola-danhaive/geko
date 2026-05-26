<div align="center">
  <img src="https://raw.githubusercontent.com/angelicalola-danhaive/geko/main/doc/_static/geko_logo.png" alt="Geko Logo" width="300"/>

  # the **G**rism **E**mission-line **K**inematics t**O**ol

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Tests](https://github.com/angelicalola-danhaive/geko/actions/workflows/tests.yml/badge.svg)](https://github.com/angelicalola-danhaive/geko/actions/workflows/tests.yml)

</div>

## Description

Geko is a Python package for analyzing grism spectroscopy from JWST NIRCam observations. It forward-models the 2D slitless spectrum of an emission-line galaxy — jointly fitting morphology, kinematics, and geometry — using JAX for accelerated computation and Numpyro for Bayesian inference via the No-U-Turn Sampler (NUTS).

**Key features:**
- **JAX-accelerated forward model**: GPU-ready 3D cube → PSF convolution → grism projection pipeline
- **Bayesian inference**: full posterior sampling with NUTS via Numpyro
- **Composable rotation curves**: plug in any combination of rotation curve components (Arctan, or custom) via `FitConfiguration`
- **Swappable morphology models**: Sérsic profile by default; extensible via a registry
- **Model-agnostic parameter handling**: `ParameterSpec` infrastructure drives sampling, priors, and postprocessing without hardcoding parameter names
- **Comprehensive diagnostics**: corner plots, summary images, chain convergence plots, and derived-quantity posteriors

---

## Code structure

```
geko/
├── config.py          # FitConfiguration and MCMCSettings dataclasses
├── fitting.py         # Fit_Numpyro: MCMC entry point (run_geko_fit)
├── models.py          # GalaxyModel, GrismFitter: forward model and inference model
├── grism.py           # Grism: 3D cube → 2D grism projection (disperse)
├── preprocess.py      # run_full_preprocessing: loads data, initialises objects
├── postprocess.py     # compute_derived_posterior, DERIVED_QUANTITIES, result tables
├── param_spec.py      # ParameterSpec, BaseModel: parameter infrastructure
├── rotation_models.py # RotationCurveComponent, CompositeRotationCurve, ArctanComponent
├── morph_models.py    # MorphologyModel, SersicMorphology
├── plotting.py        # Diagnostic and summary plots
└── utils.py           # PSF loading, resampling, coordinate utilities
```

### Key classes and their roles

| Class | Module | Role |
|---|---|---|
| `FitConfiguration` | `config.py` | Single object controlling rotation components, morphology model, prior overrides, and MCMC settings. Save/load to JSON. |
| `MCMCSettings` | `config.py` | MCMC hyperparameters (chains, warmup, samples, target acceptance). |
| `ParameterSpec` | `param_spec.py` | Describes one free or fixed parameter: name, label, prior type/bounds. Drives sampling and postprocessing uniformly. |
| `BaseModel` | `param_spec.py` | Abstract base for any model component; holds a list of `ParameterSpec` and exposes `apply_prior_overrides`. |
| `ArctanComponent` | `rotation_models.py` | Arctangent rotation curve: `v(r) = Va·(2/π)·arctan(r/r_t)`. |
| `CompositeRotationCurve` | `rotation_models.py` | Sums any number of `RotationCurveComponent` instances. Built by `FitConfiguration.build_rot_model()`. |
| `SersicMorphology` | `morph_models.py` | Sérsic flux map with free PA, `r_eff`, `n`, amplitude, and centroid. |
| `GalaxyModel` | `models.py` | Combines morphology + kinematics + rotation curve. Computes velocity fields and drives the forward model. |
| `GrismFitter` | `models.py` | Wraps `GalaxyModel` for inference: builds Numpyro model, sets bounds, applies priors. |
| `Grism` | `grism.py` | Forward-models the grism: builds the 3D emission cube, applies PSF convolution, and projects to 2D. |
| `Fit_Numpyro` | `fitting.py` | Runs NUTS inference, manages chains, saves `InferenceData`. |
| `DerivedQuantity` | `postprocess.py` | Registry entry for a derived posterior quantity (v_re, v_sigma, v_circ, M_dyn). |

### Forward model overview

```
Input: flux map F(y,x), velocity field V(y,x), dispersion D(y,x)
  │
  ├─ GalaxyModel.velocity_field()     — arctan (or composite) rotation curve + geometry
  │
  └─ Grism.disperse()
       ├─ Build 3D cube: Gaussian emission at Doppler-shifted wavelengths
       ├─ Convolve with 2D PSF (spatial)
       └─ Project to 2D grism spectrum
```

The PSF convolution is the dominant computational cost. The PSF is applied as a 2D spatial convolution in the oversampled cube (no separate LSF term — the 2D PSF already captures the spectral broadening in the dispersion direction).

### Adding a new rotation curve component

1. Subclass `RotationCurveComponent` in `rotation_models.py`, define `_DEFAULT_PARAMETERS` and `rotation_curve(r, all_params)`.
2. Register it in `COMPONENT_REGISTRY`.
3. Use it by name in `FitConfiguration(rotation_components=['Arctan', 'MyComponent'])`.

No changes to `GalaxyModel`, `GrismFitter`, the prior system, or postprocessing are required.

---

## Installation

### Using pip (recommended)

```bash
pip install astro-geko
```

### Development installation

```bash
git clone https://github.com/angelicalola-danhaive/geko.git
cd geko
conda env create -f environment.yml
conda activate geko_env
pip install -e .
```

### Requirements

- Python >= 3.8
- JAX/JAXlib (with optional GPU support — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html))
- Numpyro
- Astropy
- Photutils
- PySersic

---

## Quick start

```python
from geko.fitting import run_geko_fit

inference_data = run_geko_fit(
    output='my_fit',
    master_cat='path/to/catalog.cat',
    line='H_alpha',
    parametric=True,
    save_runs_path='./saves/',
    num_chains=2,
    num_warmup=500,
    num_samples=1000,
    source_id=12345,
    field='GOODS-S-FRESCO',
    grism_filter='F444W',
)
```

### Custom configuration

```python
from geko.config import FitConfiguration, MCMCSettings

config = FitConfiguration(
    rotation_components=['Arctan'],       # swap or extend here
    mcmc=MCMCSettings(num_chains=4, num_warmup=1000, num_samples=1000),
    morph_prior_overrides={'r_eff_mu': 5.0, 'r_eff_std': 2.0},
    geom_prior_overrides={'i_mu': 60.0, 'i_std': 5.0},
)
config.save('my_config.json')

inference_data = run_geko_fit(..., fit_config=config)
```

See the [documentation](https://astro-geko.readthedocs.io) and the `demo/` notebooks for detailed usage examples.

---

## Citation

If you use Geko in your research, please cite:

```bibtex
@article{Danhaive:2025ac,
    author  = {{Danhaive}, A. Lola and {Tacchella}, Sandro},
    journal = {arXiv e-prints},
    month   = oct,
    pages   = {arXiv:2510.07369},
    title   = {{Modelling the kinematics and morphology of galaxies in slitless spectroscopy with \textit{geko}}},
    year    = 2025
}
```

## Acknowledgements

We acknowledge support from the Royal Society Research Grants (G125142). We thank Amanda Stoffers for creating the logo.

This package makes use of [JAX](https://github.com/google/jax), [Numpyro](https://github.com/pyro-ppl/numpyro), and [Astropy](https://www.astropy.org/).

## License

MIT License — see [LICENSE](LICENSE) for details.
