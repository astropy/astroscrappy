# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..astroscrappy import detect_cosmics

from . import fake_data

# Get fake data to work on
imdata, crmask = fake_data.make_fake_data()


def test_median_clean():
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    _mask, clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0, cleantype='median')

    assert (clean[crmask] != imdata[crmask]).sum() == crmask.sum()

    # Run it again on the clean data. We shouldn't find any new cosmic rays
    _mask2, _clean2 = detect_cosmics(clean, readnoise=10., gain=1.0,
                                     sigclip=6, sigfrac=1.0, cleantype='median')
    assert _mask2.sum() == 0


def test_medmask_clean():
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    _mask, clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0, cleantype='medmask')

    assert (clean[crmask] != imdata[crmask]).sum() == crmask.sum()

    # Run it again on the clean data. We shouldn't find any new cosmic rays
    _mask2, _clean2 = detect_cosmics(clean, readnoise=10., gain=1.0,
                                     sigclip=6, sigfrac=1.0, cleantype='medmask')
    assert _mask2.sum() == 0


def test_meanmask_clean():
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    _mask, clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0, cleantype='meanmask')

    assert (clean[crmask] != imdata[crmask]).sum() == crmask.sum()

    # Run it again on the clean data. We shouldn't find any new cosmic rays
    _mask2, _clean2 = detect_cosmics(clean, readnoise=10., gain=1.0,
                                     sigclip=6, sigfrac=1.0, cleantype='meanmask')
    assert _mask2.sum() == 0


def test_idw_clean():
    # Because our image only contains single cosmics, turn off
    # neighbor detection. Also, our cosmic rays are high enough
    # contrast that we can turn our detection threshold up.
    _mask, clean = detect_cosmics(imdata, readnoise=10., gain=1.0,
                                  sigclip=6, sigfrac=1.0, cleantype='idw')

    assert (clean[crmask] != imdata[crmask]).sum() == crmask.sum()

    # Run it again on the clean data. We shouldn't find any new cosmic rays
    _mask2, _clean2 = detect_cosmics(clean, readnoise=10., gain=1.0,
                                     sigclip=6, sigfrac=1.0, cleantype='idw')
    assert _mask2.sum() == 0
