"""
The per-observation transform (adjust_for_observation for the morphology,
Pa_obs = PA + theta_rot for the kinematics) must be a rigid rotation: the
same clockwise rotate_coords(theta_rot) that moves the centroids must also
turn the light profile and velocity field. Ground truth is the sky-frame
model evaluated analytically at the inversely rotated pixel coordinates.
"""
import numpy as np
import pytest

from geko import utils
from geko.models import _v_core
from geko.morph_models import SersicMorphology

N = 61
CENTER = (N - 1) / 2
SKY = dict(amplitude=1.0, r_eff=6.0, n=1.0, PA_morph=30.0, xc_morph=34.0, yc_morph=27.0)
INC = 65.0
VA, R_T = 200.0, 3.0


def _sky_models_at(x, y):
	ellip = 1.0 - utils.compute_axis_ratio(inc=INC, q0=0.2)
	Ie = utils.flux_to_Ie(SKY['amplitude'], SKY['n'], SKY['r_eff'], ellip)
	dx, dy = x - SKY['xc_morph'], y - SKY['yc_morph']
	flux = utils.sersic_profile(dx, dy, Ie, SKY['r_eff'], SKY['n'], 0.0, 0.0, ellip,
	                            (90.0 - SKY['PA_morph']) * np.pi / 180.0)
	vel = _v_core(dx, dy, SKY['PA_morph'], INC, VA, R_T)
	return np.asarray(flux), np.asarray(vel)


@pytest.mark.parametrize('theta_rot', [20.0, 90.0, 324.42])
def test_observation_transform_is_rigid_rotation(theta_rot):
	yy, xx = np.mgrid[0:N, 0:N].astype(float)
	xs, ys = utils.rotate_coords(xx, yy, CENTER, CENTER, -np.radians(theta_rot))
	flux_true, vel_true = _sky_models_at(xs, ys)

	obs = SersicMorphology().adjust_for_observation(SKY, theta_rot, CENTER)
	flux = np.asarray(SersicMorphology().generate_flux_map(obs, {'i': INC}, N, 1))
	assert np.abs(flux - flux_true).max() < 1e-10 * flux_true.max()

	# Kinematics as in inference_model_parametric_multi: velocity centroid
	# rotated with rotate_coords, PA turned by +theta_rot
	x0, y0 = utils.rotate_coords(SKY['xc_morph'], SKY['yc_morph'], CENTER, CENTER, np.radians(theta_rot))
	vel = np.asarray(_v_core(xx - x0, yy - y0, SKY['PA_morph'] + theta_rot, INC, VA, R_T))
	assert np.abs(vel - vel_true).max() < 1e-8


def test_adjust_for_observation_turns_pa_by_plus_theta():
	obs = SersicMorphology().adjust_for_observation(SKY, 40.0, CENTER)
	assert float(obs['PA_morph']) == pytest.approx(SKY['PA_morph'] + 40.0)
