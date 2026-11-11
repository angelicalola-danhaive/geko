"""Instrumental line width, sigma0 floor and sigma0 upper-limit reporting."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import arviz as az

sys.path.insert(0, str(Path(__file__).parent))
from test_grism import _gaussian_psf_grism  # noqa: E402

from geko.postprocess import instrumental_sigma0_floor, sigma0_limits, build_results_table

C_KMS = 299792.458


def _kms_per_bin(g):
    return float(g.wave_scale) / g.wavelength * C_KMS


@pytest.mark.parametrize("pupil", ["R", "C"])
def test_sigma_inst_wide_gaussian_psf(pupil):
    """Gaussian PSF (sigma 2 detector px): line width = PSF along dispersion (+) one pixel."""
    g = _gaussian_psf_grism(pupil, sigma_px=2.0)
    pix = abs(g.dispersion_slope) * g.factor                  # detector pixel in wavelength bins
    expected = np.sqrt((2.0 * pix)**2 + pix**2 / 12) * _kms_per_bin(g)
    assert g.sigma_inst_kms == pytest.approx(expected, rel=0.03)


def test_sigma_inst_grows_with_psf_width():
    s = [_gaussian_psf_grism('R', sigma_px=w).sigma_inst_kms for w in (0.6, 1.0, 1.5)]
    assert s[0] < s[1] < s[2]


def test_instrumental_floor_is_half_mean_width():
    class G:
        def __init__(self, s): self.sigma_inst_kms = s
    assert instrumental_sigma0_floor(G(72.0)) == pytest.approx(36.0)
    assert instrumental_sigma0_floor([G(70.0), G(80.0)]) == pytest.approx(37.5)
    assert instrumental_sigma0_floor([G(70.0), G(80.0)], fraction=1.0) == pytest.approx(75.0)


def _inf(sigma0, v_re):
    post = xr.Dataset({'sigma0': (('chain', 'draw'), sigma0[None, :]),
                       'v_re': (('chain', 'draw'), v_re[None, :])})
    return az.InferenceData(posterior=post)


def test_sigma0_limits_unresolved():
    rng = np.random.default_rng(1)
    s0 = np.abs(rng.normal(0, 15, 20000))                      # piles up at zero
    vre = -rng.normal(60, 10, 20000)                           # signed (negative) rotation
    lim = sigma0_limits(_inf(s0, vre), sigma0_floor=36.0)
    assert lim['sigma0_is_upper_limit']
    assert lim['sigma0_limit'] == pytest.approx(np.percentile(s0, 84))
    expected_vs = np.percentile(np.abs(vre) / s0, 16)          # |v|, all samples
    assert lim['v_sigma_limit'] == pytest.approx(expected_vs)
    assert lim['v_sigma_limit'] > 0


def test_sigma0_limits_resolved():
    rng = np.random.default_rng(2)
    s0 = rng.normal(90, 10, 20000)
    lim = sigma0_limits(_inf(s0, rng.normal(100, 10, 20000)), sigma0_floor=36.0)
    assert not lim['sigma0_is_upper_limit']
    assert np.isnan(lim['sigma0_limit']) and np.isnan(lim['v_sigma_limit'])
