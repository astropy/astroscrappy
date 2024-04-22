# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from Cython.Build import cythonize
from extension_helpers import get_extensions
from setuptools import setup

ext_modules = get_extensions()
compiler_directives = {}

if os.getenv('COVERAGE'):
    print('Adding linetrace directive')
    compiler_directives['profile'] = True
    compiler_directives['linetrace'] = True
    os.environ['CFLAGS'] = '-DCYTHON_TRACE_NOGIL=1 --coverage -fno-inline-functions -O0'

ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives)

setup(ext_modules=ext_modules)
