from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import sys
import subprocess

from distutils.core import Extension
from distutils import log

UTIL_DIR = os.path.relpath(os.path.dirname(__file__))

CODELINES = """
import sys
from distutils.ccompiler import new_compiler
ccompiler = new_compiler()
ccompiler.add_library('gomp')
sys.exit(int(ccompiler.has_function('omp_get_num_threads')))
"""


def check_openmp():
    s = subprocess.Popen([sys.executable], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = s.communicate(CODELINES.encode('utf-8'))
    s.wait()
    return bool(s.returncode), (stdout, stderr)


def get_extensions():

    med_sources = [str(os.path.join(UTIL_DIR, "median_utils.pyx")),
                   str(os.path.join(UTIL_DIR, "medutils.c"))]

    im_sources = [str(os.path.join(UTIL_DIR, "image_utils.pyx")),
                  str(os.path.join(UTIL_DIR, "imutils.c"))]

    include_dirs = ['numpy', UTIL_DIR]

    libraries = []

    ext_med = Extension(name=str('astroscrappy.utils.median_utils'),
                    sources=med_sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=['-g', '-O3', '-funroll-loops',
                                        '-ffast-math'])
    ext_im = Extension(name=str("astroscrappy.utils.image_utils"),
                    sources=im_sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=['-g', '-O3', '-funroll-loops',
                                        '-ffast-math'])

    has_openmp, outputs = check_openmp()
    if has_openmp:
        ext_med.extra_compile_args.append('-fopenmp')
        ext_im.extra_compile_args.append('-fopenmp')
        ext_med.extra_link_args = ['-g', '-fopenmp']
        ext_im.extra_link_args = ['-g', '-fopenmp']
    else:
        log.warn('OpenMP was not found. '
                 'astroscrappy will be compiled without OpenMP. '
                 '(Use the "-v" option of setup.py for more details.)')
        log.debug(('(Start of OpenMP info)\n'
                   'compiler stdout:\n{0}\n'
                   'compiler stderr:\n{1}\n'
                   '(End of OpenMP info)').format(*outputs))

    return [ext_med, ext_im]
