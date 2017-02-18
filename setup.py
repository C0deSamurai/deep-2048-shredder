
from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(
    ext_modules=cythonize("make_move.pyx")
)
"""

# setup file for code 2
from distutils.core import setup
from distutils.extension import Extension
import distutils.debug
from Cython.Distutils import build_ext

import numpy

ext = Extension("make_move", ["make_move.pyx"],
    include_dirs = [numpy.get_include()])
    #extra_compile_args=['-ggdb'], 
    #extra_link_args=['-ggdb'],
    #define_macros=[('NDEBUG', '1')])

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})
"""