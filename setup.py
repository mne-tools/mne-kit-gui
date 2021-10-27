#! /usr/bin/env python
"""A module for KIT MEG coregistration."""

import codecs
import os

from setuptools import find_packages, setup

# get the version from __init__.py
version = None
with open(os.path.join('mne_kit_gui', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'mne-kit-gui'
DESCRIPTION = 'A module for KIT MEG coregistration.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Christian Brodbeck'
MAINTAINER_EMAIL = 'christianbrodbeck@me.com'
URL = 'https://github.com/mne-tools/mne-kit-gui'
LICENSE = 'BSD-3'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-kit-gui'
VERSION = version
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'mayavi',
    'mne>=0.23',  # TODO: Should be 0.24
]
TEST_REQUIRES = [
    'flake8', 'pydocstyle', 'pytest', 'pytest-cov', 'check-manifest', 'twine',
    'wheel', 'pyvista',
]
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               ]

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      package_data={'mne_kit_gui': [os.path.join('help', '*.json')]},
      python_requires='>=3.7',
      install_requires=INSTALL_REQUIRES,
      extras_require={
          'test': TEST_REQUIRES,
      },
      )
