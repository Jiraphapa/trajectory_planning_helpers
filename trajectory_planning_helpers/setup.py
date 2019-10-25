from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("calc_splines_cy", ["calc_splines_cy.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

setup(
    ext_modules = cythonize("calc_splines_cy.pyx"),
    include_dirs=[numpy.get_include()]
)