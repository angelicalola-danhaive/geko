"""
Tests for geko.wcs_utils and its composition with the existing
theta_rot/adjust_for_observation rotation machinery.

Every real SAPPHIRES PySersic cutout WCS happens to be exactly north-up/
east-left with zero rotation, so real data can validate scale and centering
but structurally cannot catch a sign/parity error in the orientation-
reconciliation logic. These tests use synthetic, deliberately-rotated WCS
objects specifically to close that gap.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.table import Table
from astropy.wcs import WCS

from geko import wcs_utils
from geko.models import GalaxyModel
from geko.morph_models import SersicMorphology

RA0, DEC0 = 150.0, 2.0
SCALE_DEG = 0.03 / 3600.0  # matches real SAPPHIRES cutouts (0.03"/px)
SHAPE = 31
CENTER = (SHAPE - 1) / 2


def make_wcs(crota2_deg, crpix=None, scale_deg=SCALE_DEG):
	"""Synthetic cutout WCS: north-up/east-left (matching every real SAPPHIRES
	cutout) rotated by crota2_deg (standard FITS CROTA2 convention)."""
	w = WCS(naxis=2)
	w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
	w.wcs.crval = [RA0, DEC0]
	w.wcs.crpix = [(crpix if crpix is not None else CENTER) + 1, (crpix if crpix is not None else CENTER) + 1]
	crota = np.radians(crota2_deg)
	cdelt1, cdelt2 = -scale_deg, scale_deg
	w.wcs.cd = [
		[cdelt1 * np.cos(crota), -cdelt2 * np.sin(crota)],
		[cdelt1 * np.sin(crota),  cdelt2 * np.cos(crota)],
	]
	return w


# ============================================================================
# cutout_pixel_scale_arcsec
# ============================================================================

def test_cutout_pixel_scale_arcsec():
	w = make_wcs(0.0)
	assert wcs_utils.cutout_pixel_scale_arcsec(w) == pytest.approx(0.03, abs=1e-9)


def test_cutout_pixel_scale_arcsec_invariant_to_rotation():
	for phi in [0.0, 30.0, 137.0]:
		w = make_wcs(phi)
		assert wcs_utils.cutout_pixel_scale_arcsec(w) == pytest.approx(0.03, abs=1e-9)


# ============================================================================
# sky_to_reference_pixel_offset
# ============================================================================

def test_centroid_offset_absolute_anchor():
	"""Target placed exactly 1" due East of the reference position -> pins
	the absolute sign (dx negative, since +x_ref=West means East is -x)."""
	w = make_wcs(0.0)
	target = w.pixel_to_world(CENTER, CENTER).spherical_offsets_by(1 * u.arcsec, 0 * u.arcsec)
	xc_pix, yc_pix = w.world_to_pixel(target)
	dx, dy = wcs_utils.sky_to_reference_pixel_offset(
		w, xc_pix, yc_pix, RA0, DEC0, wcs_utils.GRISM_PIXEL_SCALE_ARCSEC)
	assert dx == pytest.approx(-1 / wcs_utils.GRISM_PIXEL_SCALE_ARCSEC, abs=1e-6)
	assert dy == pytest.approx(0.0, abs=1e-6)


def test_centroid_offset_invariant_to_cutout_rotation():
	"""The offset must depend only on true sky separation, never on how the
	input cutout happens to be oriented -- compute pixel coords of a fixed
	target sky position on an unrotated and a rotated WCS, and confirm both
	give identical reference-frame offsets."""
	target = make_wcs(0.0).pixel_to_world(CENTER + 3.0, CENTER - 2.0)
	results = []
	for phi in [0.0, 30.0, 137.0, -60.0]:
		w = make_wcs(phi)
		xc_pix, yc_pix = w.world_to_pixel(target)
		dx, dy = wcs_utils.sky_to_reference_pixel_offset(
			w, xc_pix, yc_pix, RA0, DEC0, wcs_utils.GRISM_PIXEL_SCALE_ARCSEC)
		results.append((dx, dy))
	for dx, dy in results[1:]:
		assert dx == pytest.approx(results[0][0], abs=1e-6)
		assert dy == pytest.approx(results[0][1], abs=1e-6)


# ============================================================================
# cutout_direction_to_reference_frame_angle_deg
# ============================================================================

def test_reference_frame_angle_degenerate_case():
	"""Zero-rotation WCS (matches every real SAPPHIRES cutout exactly):
	theta=0 (pointing along the cutout's own +x = West) -> 0deg;
	theta=pi/2 (pointing along +y = North) -> 270deg. Hand-derivable directly
	from the function's definition (-atan2(dy_ref, dx_ref) mod 360) without
	going through any rotated-WCS reasoning."""
	w = make_wcs(0.0)
	angle0 = wcs_utils.cutout_direction_to_reference_frame_angle_deg(w, CENTER, CENTER, 0.0)
	angle90 = wcs_utils.cutout_direction_to_reference_frame_angle_deg(w, CENTER, CENTER, np.pi / 2)
	assert min(angle0 % 360, 360 - angle0 % 360) < 1e-4
	assert abs((angle90 % 360) - 270.0) < 1e-4


def test_reference_frame_angle_rotation_sign():
	"""This is the test that actually catches a sign/parity bug -- real
	SAPPHIRES data structurally cannot, since every real cutout has zero
	rotation. A cutout WCS physically rotated by +phi must shift the derived
	angle by exactly +phi (mod 360), not -phi."""
	theta_rad = 0.6
	phi = 30.0
	angle0 = wcs_utils.cutout_direction_to_reference_frame_angle_deg(make_wcs(0.0), CENTER, CENTER, theta_rad)
	angle_phi = wcs_utils.cutout_direction_to_reference_frame_angle_deg(make_wcs(phi), CENTER, CENTER, theta_rad)
	assert (angle_phi - angle0) % 360 == pytest.approx(phi, abs=1e-6)
	assert (angle_phi - angle0) % 360 != pytest.approx(-phi % 360, abs=1e-3)


# ============================================================================
# End-to-end composition with the existing, already-validated
# adjust_for_observation/rotate_coords rotation machinery
# ============================================================================

def _make_py_table():
	return Table({
		'ellip_q50': [0.4], 'ellip_q84': [0.45], 'ellip_q16': [0.35],
		'r_eff_q50': [3.0],
		'n_q50': [1.5],
		'xc_q50': [16.3], 'yc_q50': [14.7],
		'theta_q50': [0.6], 'theta_q84': [0.65], 'theta_q16': [0.55],
	})


def _get_mu(morph_model, name):
	for spec in morph_model.parameters:
		if spec.name == name:
			return float(spec.prior_mu)
	raise KeyError(name)


def _run_set_parametric_priors(cutout_wcs):
	gm = GalaxyModel(SHAPE, 5)
	gm.set_parametric_priors(
		_make_py_table(), [1e-17, 1e-18], redshift=6.0, wavelength=4.0, delta_wave=0.01,
		theta_rot=0.0, shape=SHAPE, cutout_wcs=cutout_wcs, ref_ra=RA0, ref_dec=DEC0)
	return {
		'xc_morph': _get_mu(gm.morph_model, 'xc_morph'),
		'yc_morph': _get_mu(gm.morph_model, 'yc_morph'),
		'PA_morph': _get_mu(gm.morph_model, 'PA_morph'),
	}


def test_wcs_rotation_composes_correctly_with_adjust_for_observation():
	"""The key end-to-end check: set_parametric_priors applied to a cutout
	WCS physically rotated by phi, at theta_rot=0, must equal
	set_parametric_priors applied to an *unrotated* cutout WCS followed by
	the existing, already-validated adjust_for_observation(theta_rot_deg=+phi)
	-- confirming the new WCS code composes correctly with the pre-existing
	rotation machinery, for both the centroid and PA_morph simultaneously.
	"""
	phi = 30.0
	unrotated = _run_set_parametric_priors(make_wcs(0.0))
	rotated_direct = _run_set_parametric_priors(make_wcs(phi))

	morph = SersicMorphology()
	obs_params = morph.adjust_for_observation(
		{'PA_morph': unrotated['PA_morph'], 'xc_morph': unrotated['xc_morph'], 'yc_morph': unrotated['yc_morph']},
		phi, CENTER)

	assert float(obs_params['xc_morph']) == pytest.approx(rotated_direct['xc_morph'], abs=1e-6)
	assert float(obs_params['yc_morph']) == pytest.approx(rotated_direct['yc_morph'], abs=1e-6)
	assert float(obs_params['PA_morph']) == pytest.approx(rotated_direct['PA_morph'], abs=1e-6)

	# Guard against a future accidental sign flip: the opposite sign must NOT match.
	obs_params_wrong_sign = morph.adjust_for_observation(
		{'PA_morph': unrotated['PA_morph'], 'xc_morph': unrotated['xc_morph'], 'yc_morph': unrotated['yc_morph']},
		-phi, CENTER)
	assert not (
		np.isclose(float(obs_params_wrong_sign['xc_morph']), rotated_direct['xc_morph'], atol=1e-3)
		and np.isclose(float(obs_params_wrong_sign['yc_morph']), rotated_direct['yc_morph'], atol=1e-3)
		and np.isclose(float(obs_params_wrong_sign['PA_morph']), rotated_direct['PA_morph'], atol=1e-3)
	)
