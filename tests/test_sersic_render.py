"""Tests for the subpixel model grid and the Gaussian-mixture Sersic renderer."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf, gammainc

jax.config.update('jax_enable_x64', True)

from geko import utils
from geko.morph_models import SersicMorphology

N, FAC = 31, 5


def test_subpixel_grid_centres():
    g = np.asarray(utils.subpixel_grid(N, FAC, 15.3))
    # each block of FAC subpixels is centred on its native pixel, spacing exactly 1/FAC
    assert np.allclose(g.reshape(N, FAC).mean(axis=1), np.arange(N) - 15.3)
    assert np.allclose(np.diff(g), 1 / FAC)
    # factor 1 is the native pixel grid
    assert np.allclose(utils.subpixel_grid(N, 1, 4.0), np.arange(N) - 4.0)


@pytest.mark.parametrize("n", [0.65, 0.8, 1.0, 2.0, 4.0, 8.0])
def test_mog_decomposition_matches_analytic_sersic(n):
    """Enclosed flux of the Gaussian decomposition vs the analytic Sersic (r_eff = 1)."""
    fl, s = map(np.asarray, utils.sersic_mog_components(n))
    assert np.all(fl >= 0)
    r = np.logspace(-2, np.log10(5), 200)
    enc = (fl[None, :] * (1 - np.exp(-r[:, None]**2 / (2 * s[None, :]**2)))).sum(1)
    bn = float(utils.bn_approx(n))
    assert np.abs(enc - gammainc(2 * n, bn * r**(1 / n))).max() < 3e-3


def _exact_round(flux, r_eff, n, xc, yc):
    """Round galaxy: every Gaussian of the decomposition integrated exactly over each native pixel."""
    fl, s = map(np.asarray, utils.sersic_mog_components(n))
    s = s * r_eff
    e = np.arange(N + 1) - 0.5
    px = np.diff(0.5 * erf((e[None, :] - xc) / (np.sqrt(2) * s[:, None])), axis=1)
    py = np.diff(0.5 * erf((e[None, :] - yc) / (np.sqrt(2) * s[:, None])), axis=1)
    return flux * np.einsum('k,ki,kj->ij', fl, py, px)


@pytest.mark.parametrize("n,r_eff", [(1.0, 1.0), (1.0, 3.0), (4.0, 1.0), (4.0, 3.0), (8.0, 1.0)])
def test_sersic_image_pixel_integration(n, r_eff):
    """Native-pixel image (after a ~2.2 px FWHM PSF) and total flux vs the exact pixel integral."""
    from jax.scipy.signal import fftconvolve
    yy, xx = np.mgrid[-7:8, -7:8]
    psf = np.exp(-(xx**2 + yy**2) / (2 * 0.95**2)); psf /= psf.sum()
    for (xc, yc) in [(15.0, 15.0), (15.3, 14.8), (15.5, 15.5)]:
        m = np.asarray(utils.sersic_image(N, FAC, 2.0, r_eff, n, xc, yc, 0.0, 0.4)).reshape(N, FAC, N, FAC).sum((1, 3))
        t = _exact_round(2.0, r_eff, n, xc, yc)
        assert np.isclose(m.sum(), t.sum(), rtol=1e-4)
        cm, ct = np.asarray(fftconvolve(m, psf, 'same')), np.asarray(fftconvolve(t, psf, 'same'))
        assert np.abs(cm - ct).max() < 0.008 * ct.max()


def test_sersic_image_flux_and_gradient_smooth_in_centre():
    """Compact high-n, flattened profile aligned with the grid: the total flux must not
    depend on the sub-pixel centre (the old point-sampled renderer varied by up to 300x)."""
    xs = jnp.linspace(14.5, 15.5, 101)
    tot = jax.vmap(lambda x: utils.sersic_image(N, FAC, 1.0, 0.5, 8.0, x, 15.0, 0.8, 0.0).sum())(xs)
    assert float(tot.max() - tot.min()) < 1e-4
    cen = lambda x: utils.sersic_image(N, FAC, 1.0, 0.5, 8.0, x, 15.0, 0.5, 0.5).reshape(N, FAC, N, FAC).sum((1, 3))[15, 15]
    g = jax.vmap(jax.grad(cen))(xs)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_sersic_image_gradients_finite_over_prior_range():
    L = lambda r, n, x: (utils.sersic_image(N, FAC, 1.0, r, n, x, 15.0, 0.5, 0.5)**2).sum()
    g = jax.vmap(jax.grad(L, argnums=(0, 1, 2)), in_axes=(None, 0, None))(0.7, jnp.linspace(0.65, 8.0, 30), 15.13)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)


def test_generate_flux_map_uses_renderer():
    mm = SersicMorphology()
    mp = {'amplitude': 3.0, 'r_eff': 2.0, 'n': 1.5, 'PA_morph': 30.0, 'xc_morph': 15.2, 'yc_morph': 14.9}
    img = mm.generate_flux_map(mp, {'i': 60.0}, N, FAC)
    ell = 1 - utils.compute_axis_ratio(inc=60.0, q0=0.2)
    ref = utils.sersic_image(N, FAC, 3.0, 2.0, 1.5, 15.2, 14.9, ell, (90 - 30.0) * np.pi / 180)
    assert img.shape == (N * FAC, N * FAC)
    assert np.allclose(img, ref)
    assert np.isclose(float(img.sum()), 3.0, rtol=2e-3)


def test_sersic_n_prior_floor():
    mm = SersicMorphology()
    n_spec = next(s for s in mm.parameters if s.name == 'n')
    assert n_spec.prior_min == utils.SERSIC_N_MIN == 0.65
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mm.apply_prior_overrides({'n_min': 0.36})
    assert n_spec.prior_min == utils.SERSIC_N_MIN
    assert any('below' in str(x.message) for x in w)
    mm.apply_prior_overrides({'n_min': 1.0})          # higher minima are kept
    assert n_spec.prior_min == 1.0
