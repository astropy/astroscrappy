# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..astroscrappy import detect_cosmics
from . import fake_data


def test_main():
    imdata, expected_crmask = fake_data.make_fake_data()
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    mask, _clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0)
    assert (mask == expected_crmask).sum() == (1001 * 1001)
