"""Prior overrides: {param}_type changes the prior shape; ignored keys are reported."""
import numpy as np
import pytest
from numpyro import handlers
from jax import random

from geko.param_spec import ParameterSpec, _apply_overrides_to_specs, sample_specs
from geko.rotation_models import ArctanComponent


def _va_spec():
	return [ParameterSpec('Va', 'Va', 'Va', 'Uniform', prior_min=-1000.0, prior_max=1000.0)]


def test_type_override_makes_va_gaussian():
	specs = _va_spec()
	_apply_overrides_to_specs(specs, {'Va_type': 'TruncatedNormal', 'Va_mu': 100.0, 'Va_std': 50.0})
	assert specs[0].prior_type == 'TruncatedNormal'
	draws = []
	for k in range(400):
		with handlers.seed(rng_seed=k):
			draws.append(float(sample_specs(specs)['Va']))
	assert abs(np.mean(draws) - 100.0) < 10.0
	assert abs(np.std(draws) - 50.0) < 10.0
	assert min(draws) >= -1000.0 and max(draws) <= 1000.0


def test_invalid_type_raises():
	with pytest.raises(ValueError):
		_apply_overrides_to_specs(_va_spec(), {'Va_type': 'Gaussian'})


def test_unused_suffix_is_reported(capsys):
	_apply_overrides_to_specs(_va_spec(), {'Va_mu': 0.0, 'Va_max': 400.0})
	out = capsys.readouterr().out
	assert 'Va_mu is ignored' in out
	assert 'Va_max is ignored' not in out


def test_unknown_key_is_reported(capsys):
	_apply_overrides_to_specs(_va_spec(), {'Va_max': 400.0, 'Vaa_min': 0.0})
	out = capsys.readouterr().out
	assert "no such parameter here: ['Vaa_min']" in out


def test_used_keys_print_nothing(capsys):
	matched = _apply_overrides_to_specs(_va_spec(), {'Va_min': -400.0, 'Va_max': 400.0})
	assert matched == {'Va_min', 'Va_max'}
	assert capsys.readouterr().out == ''


def test_component_returns_matched_without_reporting(capsys):
	comp = ArctanComponent()
	matched = comp.apply_prior_overrides({'Va_max': 400.0, 'other_mu': 1.0}, report_unknown=False)
	assert matched == {'Va_max'}
	assert capsys.readouterr().out == ''
