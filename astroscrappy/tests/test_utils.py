# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose

from ..utils import (median, optmed3, optmed5, optmed7, optmed9, optmed25,
                     medfilt3, medfilt5, medfilt7, sepmedfilt3, sepmedfilt5,
                     sepmedfilt7, sepmedfilt9, dilate3, dilate5, subsample,
                     rebin, laplaceconvolve, convolve)

from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage


def test_median():
    a = np.ascontiguousarray(np.random.random(1001)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(median(a, 1001))


def test_optmed3():
    a = np.ascontiguousarray(np.random.random(3)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(optmed3(a))


def test_optmed5():
    a = np.ascontiguousarray(np.random.random(5)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(optmed5(a))


def test_optmed7():
    a = np.ascontiguousarray(np.random.random(7)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(optmed7(a))


def test_optmed9():
    a = np.ascontiguousarray(np.random.random(9)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(optmed9(a))


def test_optmed25():
    a = np.ascontiguousarray(np.random.random(25)).astype('f4')
    assert np.float32(np.median(a)) == np.float32(optmed25(a))


def test_medfilt3():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('f4')
    npmed3 = ndimage.filters.median_filter(a, size=(3, 3), mode='nearest')
    npmed3[:1, :] = a[:1, :]
    npmed3[-1:, :] = a[-1:, :]
    npmed3[:, :1] = a[:, :1]
    npmed3[:, -1:] = a[:, -1:]

    med3 = medfilt3(a)
    assert np.all(med3 == npmed3)


def test_medfilt5():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed5 = ndimage.filters.median_filter(a, size=(5, 5), mode='nearest')
    npmed5[:2, :] = a[:2, :]
    npmed5[-2:, :] = a[-2:, :]
    npmed5[:, :2] = a[:, :2]
    npmed5[:, -2:] = a[:, -2:]

    med5 = medfilt5(a)
    assert np.all(med5 == npmed5)


def test_medfilt7():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed7 = ndimage.filters.median_filter(a, size=(7, 7), mode='nearest')
    npmed7[:3, :] = a[:3, :]
    npmed7[-3:, :] = a[-3:, :]
    npmed7[:, :3] = a[:, :3]
    npmed7[:, -3:] = a[:, -3:]

    med7 = medfilt7(a)
    assert np.all(med7 == npmed7)


def test_sepmedfilt3():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed3 = ndimage.filters.median_filter(a, size=(1, 3), mode='nearest')
    npmed3[:, :1] = a[:, :1]
    npmed3[:, -1:] = a[:, -1:]
    npmed3 = ndimage.filters.median_filter(npmed3, size=(3, 1), mode='nearest')
    npmed3[:1, :] = a[:1, :]
    npmed3[-1:, :] = a[-1:, :]
    npmed3[:, :1] = a[:, :1]
    npmed3[:, -1:] = a[:, -1:]

    med3 = sepmedfilt3(a)
    assert np.all(med3 == npmed3)


def test_sepmedfilt5():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed5 = ndimage.filters.median_filter(a, size=(1, 5), mode='nearest')
    npmed5[:, :2] = a[:, :2]
    npmed5[:, -2:] = a[:, -2:]
    npmed5 = ndimage.filters.median_filter(npmed5, size=(5, 1), mode='nearest')
    npmed5[:2, :] = a[:2, :]
    npmed5[-2:, :] = a[-2:, :]
    npmed5[:, :2] = a[:, :2]
    npmed5[:, -2:] = a[:, -2:]

    med5 = sepmedfilt5(a)
    assert np.all(med5 == npmed5)


def test_sepmedfilt7():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed7 = ndimage.filters.median_filter(a, size=(1, 7), mode='nearest')
    npmed7[:, :3] = a[:, :3]
    npmed7[:, -3:] = a[:, -3:]
    npmed7 = ndimage.filters.median_filter(npmed7, size=(7, 1), mode='nearest')
    npmed7[:3, :] = a[:3, :]
    npmed7[-3:, :] = a[-3:, :]
    npmed7[:, :3] = a[:, :3]
    npmed7[:, -3:] = a[:, -3:]

    med7 = sepmedfilt7(a)
    assert np.all(med7 == npmed7)


def test_sepmedfilt9():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npmed9 = ndimage.filters.median_filter(a, size=(1, 9), mode='nearest')
    npmed9[:, :4] = a[:, :4]
    npmed9[:, -4:] = a[:, -4:]
    npmed9 = ndimage.filters.median_filter(npmed9, size=(9, 1), mode='nearest')
    npmed9[:4, :] = a[:4, :]
    npmed9[-4:, :] = a[-4:, :]
    npmed9[:, :4] = a[:, :4]
    npmed9[:, -4:] = a[:, -4:]

    med9 = sepmedfilt9(a)
    assert np.all(med9 == npmed9)


def test_dilate5():
    # Put 5% of the pixels into a mask
    a = np.zeros((1001, 1001), dtype=np.bool)
    a[np.random.random((1001, 1001)) < 0.05] = True
    kernel = np.ones((5, 5))
    kernel[0, 0] = 0
    kernel[0, 4] = 0
    kernel[4, 0] = 0
    kernel[4, 4] = 0
    # Make a zero padded array for the numpy version to operate
    paddeda = np.zeros((1005, 1005), dtype=np.bool)
    paddeda[2:-2, 2:-2] = a[:, :]
    npdilate = binary_dilation(np.ascontiguousarray(paddeda),
                               structure=kernel, iterations=2)
    cdilate = dilate5(a, 2)

    assert np.all(npdilate[2:-2, 2:-2] == cdilate)


def test_dilate3():
    # Put 5% of the pixels into a mask
    a = np.zeros((1001, 1001), dtype=np.bool)
    a[np.random.random((1001, 1001)) < 0.05] = True
    kernel = np.ones((3, 3))
    npgrow = binary_dilation(np.ascontiguousarray(a),
                             structure=kernel, iterations=1)
    cgrow = dilate3(a)
    npgrow[:, 0] = a[:, 0]
    npgrow[:, -1] = a[:, -1]
    npgrow[0, :] = a[0, :]
    npgrow[-1, :] = a[-1, :]
    assert np.all(npgrow == cgrow)


def test_subsample():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    npsubsamp = np.zeros((a.shape[0] * 2, a.shape[1] * 2), dtype=np.float32)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            npsubsamp[2 * i, 2 * j] = a[i, j]
            npsubsamp[2 * i + 1, 2 * j] = a[i, j]
            npsubsamp[2 * i, 2 * j + 1] = a[i, j]
            npsubsamp[2 * i + 1, 2 * j + 1] = a[i, j]

    csubsamp = subsample(a)
    assert np.all(npsubsamp == csubsamp)


def test_rebin():
    a = np.ascontiguousarray(np.random.random((2002, 2002)), dtype=np.float32)
    a = a.astype('<f4')
    nprebin = np.zeros((1001, 1001), dtype=np.float32).astype('<f4')
    for i in range(1001):
        for j in range(1001):
            nprebin[i, j] = (a[2 * i, 2 * j] + a[2 * i + 1, 2 * j] +
                             a[2 * i, 2 * j + 1] + a[2 * i + 1, 2 * j + 1])
            nprebin[i, j] /= np.float32(4.0)
    crebin = rebin(a)

    assert_allclose(crebin, nprebin, rtol=0, atol=1.e-6)


def test_laplaceconvolve():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    k = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    k = k.astype('<f4')
    npconv = ndimage.filters.convolve(a, k, mode='constant', cval=0.0)
    cconv = laplaceconvolve(a)
    assert_allclose(npconv, cconv, rtol=0.0, atol=1e-6)


def test_convolve():
    a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
    k = np.ascontiguousarray(np.random.random((5, 5))).astype('<f4')
    npconv = ndimage.filters.convolve(a, k, mode='constant', cval=0.0)
    cconv = convolve(a, k)
    assert_allclose(cconv, npconv, rtol=0, atol=1e-5)
