"""
WCS-based conversions between PySersic morphology-fit cutouts and geko's
grism model coordinate frame.

	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = [
	'GRISM_PIXEL_SCALE_ARCSEC',
	'cutout_pixel_scale_arcsec',
	'sky_to_reference_pixel_offset',
	'cutout_direction_to_reference_frame_angle_deg',
]

import warnings

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

# NIRCam LW grism detector plate scale -- documented instrumental constant,
# not derivable from a WCS (the grism FITS files carry no WCS information at
# all). Matches grism.py's set_detector_scale(0.0629) and preprocess.py's
# im_scale = 0.0629/factor.
GRISM_PIXEL_SCALE_ARCSEC = 0.0629


def cutout_pixel_scale_arcsec(cutout_wcs):
	"""Pixel scale of a PySersic input cutout, in arcsec/pixel.

	Averages the two axis scales from the cutout's own WCS; warns if they
	differ by more than 1% (geko's model only has one scalar pixel scale
	downstream, so non-square pixels can't be represented anyway).
	"""
	scales_deg = proj_plane_pixel_scales(cutout_wcs)
	scales_arcsec = np.asarray(scales_deg) * 3600.0
	if len(scales_arcsec) >= 2:
		rel_diff = abs(scales_arcsec[0] - scales_arcsec[1]) / np.mean(scales_arcsec[:2])
		if rel_diff > 0.01:
			warnings.warn(
				f"cutout WCS has non-square pixels (scales {scales_arcsec} arcsec/px, "
				f"{rel_diff:.1%} different) -- using their mean.")
	return float(np.mean(scales_arcsec))


def sky_to_reference_pixel_offset(cutout_wcs, xc_pix, yc_pix, ref_ra_deg, ref_dec_deg,
                                   grism_pixel_scale_arcsec):
	"""Offset (in grism reference-frame pixels) from (ref_ra, ref_dec) to the
	sky position of a PySersic cutout pixel coordinate (xc_pix, yc_pix).

	Reference-frame convention: +x = West, +y = North (matches the confirmed
	orientation of every SAPPHIRES cutout WCS: PC1_1=-1, PC2_2=+1).

	Returns (dx_ref_px, dy_ref_px).
	"""
	sky_centroid = cutout_wcs.pixel_to_world(xc_pix, yc_pix)
	ref_coord = SkyCoord(ra=ref_ra_deg * u.deg, dec=ref_dec_deg * u.deg)
	d_lon, d_lat = ref_coord.spherical_offsets_to(sky_centroid)
	d_east_arcsec = d_lon.to(u.arcsec).value
	d_north_arcsec = d_lat.to(u.arcsec).value

	dx_ref_px = -d_east_arcsec / grism_pixel_scale_arcsec
	dy_ref_px = d_north_arcsec / grism_pixel_scale_arcsec
	return dx_ref_px, dy_ref_px


def cutout_direction_to_reference_frame_angle_deg(cutout_wcs, x_pix, y_pix, theta_rad, step_px=1.0):
	"""Angle (degrees) of a cutout-frame direction `theta_rad` (radians, CCW
	from the cutout's own +x pixel axis, PySersic's convention), expressed as
	a standard CCW-from-+x math angle in the grism reference frame's own
	(x=West, y=North) axes.

	Deliberately NOT the astronomical "East of North" position angle: the
	legacy theta_rot_adj/PA wrap-around logic downstream (set_parametric_priors)
	expects a standard math angle in the reference frame's own axes (the same
	convention as PySersic's raw theta and as rotate_coords), which is the
	*opposite* rotational sense from an East-of-North PA when +x=West -- using
	position_angle() directly here was verified (via a synthetic end-to-end
	test) to compose with the wrong sign relative to
	sky_to_reference_pixel_offset's centroid convention. This function reuses
	that same West/North projection instead, so the two stay consistent.

	Goes through pixel_to_world (not raw CD/PC algebra) so it's robust to
	whatever WCS representation the cutout uses.

	Note the sign: this is the *negative* of the (x=West,y=North) reference-
	frame math angle you'd naively expect. The pre-existing, unchanged
	theta_rot_adj/PA wrap-around formula downstream in set_parametric_priors
	has an effective -1 slope relative to its `theta` input (calibrated for
	photutils' own `cat.orientation` sign convention) -- verified via a
	synthetic end-to-end composition test (set_parametric_priors with a
	rotated cutout WCS at theta_rot=0 vs. an unrotated cutout WCS +
	adjust_for_observation) that the *centroid* conversion
	(sky_to_reference_pixel_offset, which has no such negation) composes
	correctly with adjust_for_observation(theta_rot_deg=+phi) for a cutout
	physically rotated by +phi; matching that for PA_morph requires this
	function's raw West/North angle to be negated first, to counteract that
	downstream -1 slope.
	"""
	sky0 = cutout_wcs.pixel_to_world(x_pix, y_pix)
	sky1 = cutout_wcs.pixel_to_world(
		x_pix + step_px * np.cos(theta_rad),
		y_pix + step_px * np.sin(theta_rad),
	)
	d_lon, d_lat = sky0.spherical_offsets_to(sky1)
	dx_ref = -d_lon.to(u.arcsec).value  # West component
	dy_ref = d_lat.to(u.arcsec).value   # North component
	return -np.degrees(np.arctan2(dy_ref, dx_ref)) % 360.0
