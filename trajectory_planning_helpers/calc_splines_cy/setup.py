from distutils.core import setup
from distutils.extension import Extension
#from Cython.Build import cythonize
from Cython.Distutils import build_ext as build_pyx
import numpy

# setup(
#     ext_modules=[
#         Extension("calc_splines_cy", ["calc_splines_cy.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# setup(
#     ext_modules = cythonize("calc_splines_cy.pyx"),
#     include_dirs=[numpy.get_include()]
# )

ext = [Extension('wrap_calc_splines_cy',
                 sources=['calc_splines_cy.pyx'],
                 include_dirs=[numpy.get_include()])]

setup(name = 'wrap_calc_splines_cy', ext_modules=ext, cmdclass = { 'build_ext': build_pyx })