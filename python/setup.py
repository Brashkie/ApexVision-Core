r"""
ApexVision-Core — Cython Build Script
Compila las extensiones Cython: pixel_ops y nms.

Uso:
    python python/setup.py build_ext --inplace
    o via script:
    .venv\Scripts\Activate.ps1
    powershell scripts\build_cython.ps1
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "python.cython_ext.pixel_ops",
        ["python/cython_ext/pixel_ops.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "python.cython_ext.nms",
        ["python/cython_ext/nms.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="apexvision-core",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
