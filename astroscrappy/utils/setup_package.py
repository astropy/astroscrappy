import os
import numpy as np

from setuptools import Extension
from extension_helpers import add_openmp_flags_if_available

UTIL_DIR = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    med_sources = [os.path.join(UTIL_DIR, "median_utils.pyx"),
                   os.path.join(UTIL_DIR, "medutils.c")]

    im_sources = [os.path.join(UTIL_DIR, "image_utils.pyx"),
                  os.path.join(UTIL_DIR, "imutils.c")]

    include_dirs = [np.get_include(), UTIL_DIR]

    if 'CFLAGS' in os.environ:
        extra_compile_args = os.environ['CFLAGS'].split()
    else:
        extra_compile_args = ['-g', '-O3', '-funroll-loops', '-ffast-math']

    ext_med = Extension(name='astroscrappy.utils.median_utils',
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                        sources=med_sources,
                        include_dirs=include_dirs,
                        extra_compile_args=extra_compile_args)
    ext_im = Extension(name="astroscrappy.utils.image_utils",
                       define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                       sources=im_sources,
                       include_dirs=include_dirs,
                       extra_compile_args=extra_compile_args)

    add_openmp_flags_if_available(ext_med)
    add_openmp_flags_if_available(ext_im)

    return [ext_med, ext_im]
