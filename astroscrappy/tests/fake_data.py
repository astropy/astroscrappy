# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


# Make a simple Gaussian function for testing purposes
def gaussian(image_shape, x0, y0, brightness, fwhm):
    x = np.arange(image_shape[1])
    y = np.arange(image_shape[0])
    x2d, y2d = np.meshgrid(x, y)

    sig = fwhm  / 2.35482

    normfactor = brightness / 2.0 / np.pi * sig ** -2.0
    exponent = -0.5 * sig ** -2.0
    exponent *= (x2d - x0) ** 2.0 + (y2d - y0) ** 2.0

    return normfactor * np.exp(exponent)


def make_fake_data():
    """
    Generate fake data that can be used to test the detection and cleaning algorithms

    Returns
    -------
    imdata : numpy float array
        Fake Image data
    crmask : numpy boolean array
        Boolean mask of locations of injected cosmic rays
    """
    # Set a seed so that the tests are repeatable
    np.random.seed(200)

    # Create a simulated image to use in our tests
    imdata = np.zeros((1001, 1001), dtype=np.float32)

    # Add sky and sky noise
    imdata += 200

    # Add some fake sources
    for i in range(100):
        x = np.random.uniform(low=0.0, high=1001)
        y = np.random.uniform(low=0.0, high=1001)
        brightness = np.random.uniform(low=1000., high=30000.)
        imdata += gaussian(imdata.shape, x, y, brightness, 3.5)

    # Add the poisson noise
    imdata = np.float32(np.random.poisson(imdata))

    # Add readnoise
    imdata += np.random.normal(0.0, 10.0, size=(1001, 1001))

    # Add 100 fake cosmic rays
    cr_x = np.random.randint(low=5, high=995, size=100)
    cr_y = np.random.randint(low=5, high=995, size=100)

    cr_brightnesses = np.random.uniform(low=1000.0, high=30000.0, size=100)

    imdata[cr_y, cr_x] += cr_brightnesses
    imdata = imdata.astype('f4')

    # Make a mask where the detected cosmic rays should be
    crmask = np.zeros((1001, 1001), dtype=np.bool)
    crmask[cr_y, cr_x] = True
    return imdata, crmask
