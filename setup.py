# This pip setup is made from instructions available at:
# https://towardsdatascience.com/how-to-build-your-first-python-package-6a00b02635c9

import pathlib
from setuptools import setup, find_packages
import rsr

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Package information
PACKAGE_NAME = 'rsr'
AUTHOR = 'Cyril Grima'
AUTHOR_EMAIL = 'cyril.grima@gmail.com'
URL = 'https://github.com/cgrima/rsr'

LICENSE = 'MIT'
DESCRIPTION = 'Python utilities for applying the Radar Statistical Reconnaissance technique'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

# Dependencies
INSTALL_REQUIRES = [
      'lmfit>=1.0.1',
      'matplotlib>=3.3.3',
      'numpy>=1.19.4',
      'pandas>=1.1.4',
      'subradar>=1.0.1',
      'scipy>=1.5.2',
      'scikit-learn>=0.0',
]

# Bundle everything above
setup(name=PACKAGE_NAME,
      version=rsr.__version__,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
