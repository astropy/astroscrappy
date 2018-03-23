from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import sys
import subprocess

from distutils.core import Extension
from distutils import log

from astropy_helpers import setup_helpers

UTIL_DIR = os.path.relpath(os.path.dirname(__file__))

CODELINES = r"""
import sys
import os
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
ccompiler = new_compiler()
customize_compiler(ccompiler)
ccompiler.add_library('gomp')
has_omp_functions = ccompiler.has_function('omp_get_num_threads')
with open('openmp_check.c', 'w') as f:
    f.write('#include<stdio.h>\n')
    f.write('int main()\n')
    f.write('{\n')
    f.write('printf("Hello World");\n')
    f.write('}')
try:
    ccompiler.compile(['openmp_check.c'], extra_postargs=['-fopenmp'])
    fopenmp_flag_works = True
except:
    fopenmp_flag_works = False
os.remove('openmp_check.c')
if os.path.exists('openmp_check.o'):
    os.remove('openmp_check.o')
sys.exit(int(has_omp_functions & fopenmp_flag_works))
"""


def check_openmp():
    if setup_helpers.get_compiler_option() == 'msvc':
        # The free version of the Microsoft compilers supports
        # OpenMP in MSVC 2008 (python 2.7) and MSVC 2015 (python 3.5+),
        # but not MSVC 2010 (python 3.4 and lower).
        major, minor = sys.version_info[:2]
        has_openmp = not (major == 3 and minor < 5)
        # Empty return tuple is to match the alternative check, below.
        return has_openmp, ("", "")
    else:
        # Unix-y compiler, use this check.
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
    if 'CFLAGS' in os.environ:
        extra_compile_args = os.environ['CFLAGS'].split()
    else:
        extra_compile_args = ['-g', '-O3', '-funroll-loops','-ffast-math']
    ext_med = Extension(name=str('astroscrappy.utils.median_utils'),
                    sources=med_sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=extra_compile_args)
    ext_im = Extension(name=str("astroscrappy.utils.image_utils"),
                    sources=im_sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=extra_compile_args)

    has_openmp, outputs = check_openmp()
    if has_openmp:
        if setup_helpers.get_compiler_option() == 'msvc':
            ext_med.extra_compile_args.append('-openmp')
            ext_im.extra_compile_args.append('-openmp')
        else:
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
