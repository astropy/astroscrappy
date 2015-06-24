# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Astro-SCRAPPY: The Speedy Cosmic Ray Annihilation Package in Python
===================================================================

Name : Astro-SCRAPPY
Author : Curtis McCully
Date : October 2014

Optimized Cosmic Ray Detector:

Astro-SCRAPPY is designed to detect cosmic rays in images (numpy arrays),
originally based on Pieter van Dokkum's L.A.Cosmic algorithm.

Much of this was originally adapted from cosmics.py written by Malte Tewes.
I have ported all of the slow functions to Cython/C, and optimized
where I can. This is designed to be as fast as possible so some of the
readability has been sacrificed, specifically in the C code.

L.A.Cosmic = LAplacian Cosmic ray detection

If you use this code, please consider adding this repository address in a
footnote: https://github.com/astropy/astroscrappy.

Please cite the original paper which can be found at:
http://www.astro.yale.edu/dokkum/lacosmic/

van Dokkum 2001, PASP, 113, 789, 1420
(article : http://adsabs.harvard.edu/abs/2001PASP..113.1420V)

This code requires Cython, preferably version >= 0.21.

Parallelization is achieved using OpenMP. This code should compile (although
the Cython files may have issues) using a compiler that does not support OMP,
e.g. clang.

Notes
-----
There are some differences from original LACosmic:

    - Automatic recognition of saturated stars.
      This avoids treating such stars as large cosmic rays.

    - I have tried to optimize all of the code as much as possible while
      maintaining the integrity of the algorithm. One of the key speedups is to
      use a separable median filter instead of the true median filter. While these
      are not identical, they produce comparable results and the separable version
      is much faster.

    - This implementation is much faster than the Python by as much as a factor of
      28 depending on the given parameters.
      This implementation is much faster than the original IRAF version, by a factor
      of ~90.

Note that arrays always must be C-contiguous, thus all loops are y outer, x inner.
This follows the Pyfits convention.

scipy is required for certain tests to pass, but the code itself does not depend on
scipy.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .astroscrappy import *
    from .utils import *

__all__ = ['detect_cosmics']
