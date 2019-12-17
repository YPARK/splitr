import os
import re

from setuptools import setup, find_packages

_README           = os.path.join(os.path.dirname(__file__), 'README.md')
_LONG_DESCRIPTION = open(_README).read()

setup(
    name = 'splitr',
    version = '0.1.0',
    packages = find_packages(),
    description = 'single cell preprocessing and deconvolution',
    author = 'Yongjin Park',
    author_email = 'yongjin.peter.park@gmail.com',
    long_description = _LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    entry_points = {
        'console_scripts': [
            'splitr = splitr:__main__'
        ]
    },
    install_requires = [
    'numpy',
    'argparse',
    'keras',
    'mmutil',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research"
    ],
)
