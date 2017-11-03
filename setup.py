from __future__ import print_function

import logging
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import cython_gsl


extensions = [
    Extension("stellarWakes.*", ["stellarWakes/*.pyx"],
        include_dirs=[numpy.get_include(),cython_gsl.get_cython_include_dir()], extra_compile_args=["-ffast-math",'-O3',"-march=native"],libraries=cython_gsl.get_libraries())
]

setup_args = {'name':'stellarWakes',
    'version':'0.0',
    'description':'A Python package for DM subhalo searches using stellar wakes',
    'url':'https://github.com/bsafdi/stellarWakes',
    'author':'Benjamin R. Safdi',
    'author_email':'bsafdi@umich.edu',
    'license':'MIT',
    'install_requires':[
            'numpy',
            'matplotlib',
            'healpy',
            'Cython',
            'pymultinest',
            'jupyter',
            'scipy',
            'CythonGSL',
        ]}

setup(packages=['stellarWakes'],
    ext_modules = cythonize(extensions),
    **setup_args
)
print("Compilation successful!")
