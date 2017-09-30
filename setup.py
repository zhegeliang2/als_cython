from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name="raw_als", ext_modules=cythonize('raw_als.pyx'), 
        include_dirs=[numpy.get_include()]
        )
