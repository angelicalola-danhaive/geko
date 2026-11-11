"""MCMCSettings.chain_method / init_strategy: config round-trip, mapping to
numpyro init functions, and that vectorized chains with dispersed inits really
start from different points."""
import numpy as np
import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

from geko.config import FitConfiguration, MCMCSettings
from geko.fitting import make_init_strategy


def test_defaults_keep_previous_behaviour():
    m = MCMCSettings()
    assert m.chain_method == 'parallel'
    assert m.init_strategy == 'median'


def test_config_yaml_roundtrip(tmp_path):
    cfg = FitConfiguration(mcmc=MCMCSettings(num_chains=4, chain_method='vectorized',
                                             init_strategy='uniform'))
    cfg.save('cfg.yaml', output_dir=str(tmp_path))
    loaded = FitConfiguration.load(str(tmp_path / 'cfg.yaml'))
    assert loaded.mcmc.chain_method == 'vectorized'
    assert loaded.mcmc.init_strategy == 'uniform'
    assert loaded.mcmc.num_chains == 4


def test_unknown_init_strategy_raises():
    with pytest.raises(ValueError, match='init_strategy'):
        make_init_strategy('bogus')


def _toy_model():
    # same structure as geko: a standardized Uniform(0,1) site mapped to a bounded
    # parameter, plus a Normal site
    u = numpyro.sample('unscaled_Va', dist.Uniform())
    numpyro.deterministic('Va', -1000 + 2000 * u)
    numpyro.sample('unscaled_x', dist.Normal())


@pytest.mark.parametrize('name,dispersed', [('median', False), ('uniform', True), ('sample', True)])
def test_vectorized_chain_inits(name, dispersed):
    kernel = NUTS(_toy_model, init_strategy=make_init_strategy(name))
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1, num_chains=4,
                chain_method='vectorized', progress_bar=False)
    mcmc.warmup(random.PRNGKey(0), collect_warmup=False)
    # last_state.z holds each chain's position after (zero) warmup = its init
    z = np.asarray(mcmc.last_state.z['unscaled_Va'])
    assert z.shape == (4,)
    spread = np.ptp(z)
    if dispersed:
        assert spread > 0.1
    else:
        assert spread < 0.2
