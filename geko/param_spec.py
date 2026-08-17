"""
Parameter specification and sampling infrastructure for geko model-agnostic fitting.
"""

__all__ = [
    'ParameterSpec', 'BaseModel',
    'sample_specs', '_apply_fixed_to_specs', '_apply_overrides_to_specs',
    'SHARED_KINEMATIC_SPEC', 'all_param_specs',
]

from dataclasses import dataclass, field
from copy import deepcopy
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


@dataclass
class ParameterSpec:
    name:        str
    label:       str          # full axis label (LaTeX) for corner plots
    title:       str          # short title (LaTeX) for corner plot titles
    prior_type:  str          # 'Uniform' | 'Normal' | 'TruncatedNormal'
    prior_min:   float = None
    prior_max:   float = None  # None → resolved from context['r_eff'] at sample time
    prior_mu:    float = None
    prior_std:   float = None
    fixed:       bool  = False
    fixed_value:        None = None  # float → pin to number; str → link to named param


# ---------------------------------------------------------------------------
# Standalone helpers (work on plain lists, not tied to BaseModel)
# ---------------------------------------------------------------------------

def _apply_fixed_to_specs(specs: list, fixed_params: dict):
    """Set fixed=True / fixed_value on matching specs."""
    for name, value in fixed_params.items():
        for spec in specs:
            if spec.name == name:
                spec.fixed = True
                spec.fixed_value = value


def _apply_overrides_to_specs(specs: list, overrides: dict):
    """Override prior bounds/mu/std from a flat dict of {param_name_suffix: value}.
    E.g. {'r_eff_max': 5.0, 'PA_mu': 45.0}. Unknown keys are silently ignored."""
    for spec in specs:
        for suffix in ('min', 'max', 'mu', 'std'):
            key = f'{spec.name}_{suffix}'
            if key in overrides:
                setattr(spec, f'prior_{suffix}', overrides[key])


# ---------------------------------------------------------------------------
# Core sampling function
# ---------------------------------------------------------------------------

def sample_specs(specs: list, context: dict = None) -> dict:
    """Sample a list of ParameterSpec via numpyro.

    Parameters
    ----------
    specs : list of ParameterSpec
    context : dict, optional
        Already-sampled values used to:
        (a) resolve dynamic upper bounds (prior_max=None → context['r_eff'])
        (b) look up linked parameters when fixed_value is a string
    """
    context = context or {}
    params = {}

    for spec in specs:
        if spec.fixed:
            if isinstance(spec.fixed_value, str):
                source = context.get(spec.fixed_value)
                if source is None:
                    raise ValueError(
                        f"'{spec.name}' links to '{spec.fixed_value}' "
                        f"but it is not in context yet"
                    )
                params[spec.name] = numpyro.deterministic(spec.name, source)
            else:
                params[spec.name] = numpyro.deterministic(
                    spec.name, jnp.array(float(spec.fixed_value))
                )
            continue

        lo = spec.prior_min
        hi = spec.prior_max if spec.prior_max is not None else context.get('r_eff')

        if spec.prior_type in ('Normal', 'TruncatedNormal'):
            if spec.prior_mu is None:
                raise ValueError(
                    f"Parameter '{spec.name}' has prior_type='{spec.prior_type}' but prior_mu is None. "
                    f"Set it via morph_prior_overrides={{'{spec.name}_mu': <value>}} in FitConfiguration, "
                    f"or provide a PySersic catalog file."
                )
            if spec.prior_std is None:
                raise ValueError(
                    f"Parameter '{spec.name}' has prior_type='{spec.prior_type}' but prior_std is None. "
                    f"Set it via morph_prior_overrides={{'{spec.name}_std': <value>}} in FitConfiguration, "
                    f"or provide a PySersic catalog file."
                )

        if spec.prior_type == 'Uniform':
            u = numpyro.sample(f'unscaled_{spec.name}', dist.Uniform())
            params[spec.name] = numpyro.deterministic(
                spec.name, u * (hi - lo) + lo
            )
        elif spec.prior_type == 'Normal':
            u = numpyro.sample(f'unscaled_{spec.name}', dist.Normal())
            params[spec.name] = numpyro.deterministic(
                spec.name, u * spec.prior_std + spec.prior_mu
            )
        elif spec.prior_type == 'TruncatedNormal':
            low_s  = (lo - spec.prior_mu) / spec.prior_std if lo is not None else float('-inf')
            high_s = (hi - spec.prior_mu) / spec.prior_std if hi is not None else float('inf')
            u = numpyro.sample(
                f'unscaled_{spec.name}',
                dist.TruncatedNormal(low=low_s, high=high_s)
            )
            params[spec.name] = numpyro.deterministic(
                spec.name, u * spec.prior_std + spec.prior_mu
            )
        else:
            raise ValueError(f"Unknown prior_type '{spec.prior_type}' for '{spec.name}'")

    return params


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Base class for morphology models and rotation curve components.

    Subclasses declare _DEFAULT_PARAMETERS at class level; __init__ deep-copies
    them so each instance has an independent mutable list.
    """
    _DEFAULT_PARAMETERS: list = []

    def __init__(self):
        self.parameters = deepcopy(self._DEFAULT_PARAMETERS)

    def apply_prior_overrides(self, overrides: dict):
        _apply_overrides_to_specs(self.parameters, overrides)

    def _sample(self, context: dict = None) -> dict:
        return sample_specs(self.parameters, context)

    def apply_fixed_params(self, fixed_params: dict):
        _apply_fixed_to_specs(self.parameters, fixed_params)


# ---------------------------------------------------------------------------
# Shared kinematic parameters (always present regardless of rotation model)
# ---------------------------------------------------------------------------
# sigma0 is here — CompositeRotationCurve.rotation_curve() reads it from
# all_params for the asymmetric drift correction (TODO: not yet implemented).

SHARED_KINEMATIC_SPEC = [
    ParameterSpec('PA',     r'PA$_{\rm kin}$ [deg]', r'PA$_{\rm kin}$', 'Normal'),
    ParameterSpec('i',      r'$i$ [deg]',         r'$i$',         'TruncatedNormal',
                  prior_min=0.0, prior_max=90.0),
    ParameterSpec('sigma0', r'$\sigma_0$ [km/s]', r'$\sigma_0$',  'Uniform'),
    ParameterSpec('x0_vel', r'$x_v$ [px]',        r'$x_v$',        'Normal',
                  prior_mu=15.0, prior_std=1.0),
    ParameterSpec('y0_vel', r'$y_v$ [px]',        r'$y_v$',        'Normal',
                  prior_mu=15.0, prior_std=1.0),
    ParameterSpec('v0',     r'$v_{\rm sys}$ [km/s]', r'$v_{\rm sys}$', 'Normal',
                  prior_mu=0.0, prior_std=200.0),
]


def all_param_specs(morph_model, shared_kin_specs: list, composite_rot) -> list:
    """Concatenate all ParameterSpec objects across the three parameter groups."""
    return morph_model.parameters + list(shared_kin_specs) + composite_rot.parameters
