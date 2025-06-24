from geko.utils import *
import pytest
import numpy as np
from astropy.modeling.models import Sersic2D
import matplotlib.pyplot as plt

def test_oversample():
    #generate a sample 30x30 2D array
    arr = np.random.rand(30, 30)
    #oversample the array by a factor of 2
    oversampled_arr = oversample(arr, 2,2)
    #check if the shape of the oversampled array is correct
    assert oversampled_arr.shape == (60, 60)
    #check that the total flux is preserved
    assert np.isclose(np.sum(arr), np.sum(oversampled_arr))

def test_resample():
    #test that if you oversample an array and then resample it back to the original size, you get the same array
    arr = np.random.rand(30, 30)
    oversampled_arr = oversample(arr, 2, 2)
    resampled_arr = resample(oversampled_arr, 2, 2)
    assert np.allclose(arr, resampled_arr, atol=1e-6), "Resampling should return the original array within a small tolerance"

def test_compute_inclination_axis_ratio():
    axis_ratio = 0.5
    inclination = compute_inclination(axis_ratio, q0 = 0)
    assert np.isclose(float(inclination), 60.0)  # cos^-1(0.5) = 60 degrees
    new_axis_ratio = compute_axis_ratio(inclination, q0 = 0)
    assert np.isclose(float(new_axis_ratio), axis_ratio)  # cos(60 degrees) = 0.5

def test_rotating_prior():
    #create a mock image using astropy sersic 2d
    sersic = Sersic2D(amplitude=1, r_eff=1, n=1, x_0=1, y_0=1, ellip=0.5, theta=np.pi/2)
    #create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    #evaluate the sersic model on the grid
    image = sersic(X, Y)
    plt.imshow(image, origin='lower', extent=(-5, 5, -5, 5), cmap='viridis')
    plt.show()
    #rotate the image by 45 degrees
    x0_new, y0_new = rotate_coords(1,1,0,0,np.radians(45))
    print(f"New coordinates after rotation: x0={x0_new}, y0={y0_new}")
    theta_new = np.radians(45) + np.pi/2  # add pi/2 to the original theta
    print(f"New theta after rotation: {theta_new} radians")
    #create a new sersic model with the rotated parameters
    sersic_rotated = Sersic2D(amplitude=1, r_eff=1, n=1, x_0=x0_new, y_0=y0_new, ellip=0.5, theta=theta_new)
    #evaluate the rotated sersic model on the grid
    image_rotated = sersic_rotated(X, Y)
    plt.imshow(image_rotated, origin='lower', extent=(-5, 5, -5, 5), cmap='viridis')
    plt.show()
