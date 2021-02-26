# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import scipy.ndimage as ndi
from astropy.io import fits

from ..astroscrappy import detect_cosmics

TESTFILE = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', 'gmos.fits')


def test_gmos():
    """This test uses a small cutout from a standard observation with GMOS
    (S20190808S0048). The extracted region is [362:, 480:680], and the file has
    been reduced with DRAGONS.
    """
    with fits.open(TESTFILE) as hdul:
        data = hdul['SCI'].data
        var = hdul['VAR'].data
        sky = hdul['SKYFIT'].data

    m1, _ = detect_cosmics(data, readnoise=4.24, gain=1.933)
    m2, _ = detect_cosmics(data, inbkg=sky, readnoise=4.24, gain=1.933)
    m3, _ = detect_cosmics(data, inbkg=sky, invar=var, readnoise=4.24, gain=1.933)

    cosmic1 = (slice(41, 72), slice(142, 161))
    cosmic2 = (slice(117, 147), slice(35, 43))

    # We must find 2 cosmic rays, but m1 (without bkg and var) also flags
    # 2 additional pixels that are identified as independent regions
    label, nb = ndi.label(m1)
    assert nb == 4
    objects = ndi.find_objects(label)
    assert cosmic1 in objects
    assert cosmic2 in objects
    areas = sorted([np.sum(label == (i+1)) for i in range(nb)])
    assert areas == [1, 1, 74, 93]

    for mask in m2, m3:
        label, nb = ndi.label(mask)
        assert nb == 2
        objects = ndi.find_objects(label)
        assert objects[0] == cosmic1
        assert objects[1] == cosmic2
