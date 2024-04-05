#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os

from setuptools import setup

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

# Import this later to allow checking deprecated options before
from extension_helpers import get_extensions  # noqa
from Cython.Build import cythonize  # noqa

ext_modules = get_extensions()
compiler_directives = {}

if os.getenv('COVERAGE'):
    print('Adding linetrace directive')
    compiler_directives['profile'] = True
    compiler_directives['linetrace'] = True
    os.environ['CFLAGS'] = '-DCYTHON_TRACE_NOGIL=1 --coverage -fno-inline-functions -O0'

ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives)

setup(use_scm_version={'write_to': os.path.join('astroscrappy', 'version.py'),
                       'write_to_template': VERSION_TEMPLATE},
      ext_modules=ext_modules)
