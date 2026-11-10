"""
Tests for geko.utils.resample_psf_centered / read_psf_pixel_scale: any odd-
sized PSF at any pixel scale must land on the grism detector grid at the
correct scale, with its centre fixed on the central pixel.
"""
import numpy as np
import pytest
from astropy.io import fits

from geko import utils
from geko.wcs_utils import GRISM_PIXEL_SCALE_ARCSEC

OUT_SIZE = 15

# (input size [px, odd], input pixel scale [arcsec/px]): the real mpsf case,
# other oversamplings (integer and non-integer ratios), the native grism scale
# and a coarser-than-grism PSF.
PSF_CONFIGS = [
	(669, 0.031),
	(101, 0.031),
	(121, 0.0315),
	(181, 0.02),
	(301, GRISM_PIXEL_SCALE_ARCSEC / 5),
	(41, GRISM_PIXEL_SCALE_ARCSEC),
	(31, 0.08),
]


def _pixel_coords_arcsec(n, scale):
	c = (n - 1) / 2
	x = (np.arange(n) - c) * scale
	return np.meshgrid(x, x, indexing='ij')


def _second_moment_sigma_arcsec(psf, scale):
	yy, xx = _pixel_coords_arcsec(psf.shape[0], scale)
	p = psf / psf.sum()
	return np.sqrt(0.5 * ((p * xx**2).sum() + (p * yy**2).sum()))


@pytest.mark.parametrize('n_in, scale_in', PSF_CONFIGS)
def test_linear_ramp_maps_exactly(n_in, scale_in):
	"""Bilinear interpolation is exact for a linear function, so a ramp 1 + a*x + b*y (x, y in arcsec from the array centre)
	must come out as exactly 1 + a*x_out + b*y_out on the output grid. This
	pins the pixel-scale mapping and the centring together, for any input
	size and scale."""
	a, b = 0.7, -0.4
	yy, xx = _pixel_coords_arcsec(n_in, scale_in)
	ramp = 1 + a * xx + b * yy
	out = utils.resample_psf_centered(ramp, scale_in, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)
	yy_out, xx_out = _pixel_coords_arcsec(OUT_SIZE, GRISM_PIXEL_SCALE_ARCSEC)
	expected = 1 + a * xx_out + b * yy_out
	np.testing.assert_allclose(out / out[OUT_SIZE // 2, OUT_SIZE // 2], expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('n_in, scale_in', PSF_CONFIGS)
def test_gaussian_psf_centred_normalized_and_correct_width(n_in, scale_in):
	"""A Gaussian PSF sampled at any scale/size must come out normalized,
	exactly centred, and keep its physical width (in arcsec)."""
	sigma = 0.06  # arcsec (FWHM ~0.14", like F444W)
	yy, xx = _pixel_coords_arcsec(n_in, scale_in)
	psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
	out = utils.resample_psf_centered(psf, scale_in, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)

	assert out.shape == (OUT_SIZE, OUT_SIZE)
	assert out.sum() == pytest.approx(1.0, abs=1e-12)
	assert np.unravel_index(np.argmax(out), out.shape) == (OUT_SIZE // 2, OUT_SIZE // 2)
	oy, ox = _pixel_coords_arcsec(OUT_SIZE, GRISM_PIXEL_SCALE_ARCSEC)
	assert (out * ox).sum() == pytest.approx(0.0, abs=1e-12)
	assert (out * oy).sum() == pytest.approx(0.0, abs=1e-12)

	# Point-sampled at the output pixel centres, so the width is the intrinsic
	# one; bilinear interpolation of the input adds up to ~s_in^2/6.
	measured = _second_moment_sigma_arcsec(out, GRISM_PIXEL_SCALE_ARCSEC)
	expected = sigma
	tol = np.sqrt(sigma**2 + scale_in**2 / 6) - expected + 1e-3
	assert measured == pytest.approx(expected, abs=tol)


def test_same_scale_is_identity():
	"""Input already at the output scale and size -> unchanged (up to normalization)."""
	rng = np.random.default_rng(0)
	psf = rng.random((OUT_SIZE, OUT_SIZE))
	out = utils.resample_psf_centered(psf, GRISM_PIXEL_SCALE_ARCSEC, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)
	np.testing.assert_allclose(out, psf / psf.sum(), rtol=1e-12)


def test_even_input_raises():
	with pytest.raises(ValueError, match='odd'):
		utils.resample_psf_centered(np.ones((40, 40)), 0.031, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)


def test_non_square_input_raises():
	with pytest.raises(ValueError, match='square'):
		utils.resample_psf_centered(np.ones((41, 43)), 0.031, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)


def test_even_output_raises():
	with pytest.raises(ValueError, match='odd'):
		utils.resample_psf_centered(np.ones((41, 41)), 0.031, GRISM_PIXEL_SCALE_ARCSEC, 14)


def test_too_small_input_warns():
	with pytest.warns(UserWarning, match='zero-padding'):
		utils.resample_psf_centered(np.ones((11, 11)), 0.031, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)


def test_off_centre_peak_warns():
	psf = np.zeros((41, 41))
	psf[5, 5] = 1.0
	with pytest.warns(UserWarning, match='peak'):
		utils.resample_psf_centered(psf, 0.031, GRISM_PIXEL_SCALE_ARCSEC, OUT_SIZE)


def test_read_psf_pixel_scale():
	h = fits.Header()
	assert utils.read_psf_pixel_scale(h) is None
	h['PIXSCALE'] = 0.02
	assert utils.read_psf_pixel_scale(h) == pytest.approx(0.02)
	h['PIXELSCL'] = 0.0315
	assert utils.read_psf_pixel_scale(h) == pytest.approx(0.0315)  # PIXELSCL preferred
	# Falls through to later headers (e.g. data-extension header, then primary).
	assert utils.read_psf_pixel_scale(fits.Header(), None, h) == pytest.approx(0.0315)
