# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from . import fake_data
from ..astroscrappy import detect_cosmics


@pytest.fixture
def testdata():
    # Get fake data to work on
    imdata, crmask = fake_data.make_fake_data()
    return imdata, crmask


@pytest.mark.xfail
def test_median_clean(testdata):
    imdata, crmask = testdata
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


def test_medmask_clean(testdata):
    imdata, crmask = testdata
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


def test_meanmask_clean(testdata):
    imdata, crmask = testdata
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


def test_idw_clean(testdata):
    imdata, crmask = testdata
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
