import os

from numpy import get_include
from setuptools import Extension

ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    sources = [os.path.join(ROOT, "astroscrappy.pyx")]
    ext = Extension(
        name="astroscrappy.astroscrappy",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[get_include()],
        sources=sources,
    )
    return [ext]
