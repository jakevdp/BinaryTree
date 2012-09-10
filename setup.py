# run python setup.py build_ext
#  in order to build cpp files from the pyx files

import os

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ball_tree = Extension("ball_tree",
                      ["ball_tree.pyx"])

setup(cmdclass = {'build_ext': build_ext},
      name='ball_tree',
      version='1.0',
      ext_modules=[ball_tree],
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
      )
