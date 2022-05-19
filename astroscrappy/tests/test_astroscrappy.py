# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..astroscrappy import detect_cosmics
from . import fake_data
import numpy as np


def test_main():
    imdata, expected_crmask = fake_data.make_fake_data()
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    mask, _clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0)
    assert (mask == expected_crmask).sum() == (1001 * 1001)


def test_with_convolve_fine_structure():
    imdata, expected_crmask = fake_data.make_fake_data()
    # Convert from sigma to fwhm. Sigma is taken from the fake data utility.
    psf_fwhm = 3.5 * 2.0 * np.sqrt(2.0 * np.log(2.0))
    mask, _clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0, fsmode='convolve', psffwhm=psf_fwhm)
    assert (mask == expected_crmask).sum() == (1001 * 1001)
