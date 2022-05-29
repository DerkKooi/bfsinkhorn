# Copyright (c) 2022 Derk P. Kooi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys

import setuptools
from distutils.core import setup

long_description = '''A python package to calculate (non-interacting) orbital energies for 1-body Reduced Density Matrix Functional Theory in the canonical ensemble

See for details:
D.P. Kooi, 2022. Efficient Bosonic and Fermionic Sinkhorn Algorithms for Non-Interacting Ensembles in One-body Reduced Density Matrix Functional Theory in the Canonical Ensemble.
'''

setup(
    name='bfsinkhorn',
    version='0.1.0',
    author='Derk P. Kooi',
    author_email='derkkooi@gmail.com',
    description='A package to calculate (non-interacting) orbital energies for 1-body Reduced Density Matrix Functional Theory in the canonical ensemble',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/DerkKooi/bfsinkhorn",
    project_urls={
        "Bug Tracker": "https://github.com/DerkKooi/bfsinkhorn/issues",
    },
    packages=setuptools.find_packages(),
    license="MIT",
    license_files = ('LICENSE',),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
          'jax>=0.3.0', 
          'jaxlib>=0.3.0'
      ]
)
