"""exact-astro setup.py."""

import io
import pathlib
import re

from setuptools import setup

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('exact/__init__.py', encoding='utf_8_sig').read(),
).group(1)

install_requires = ['numba', 'numpy']
packages = ['exact']

description = 'Exact solutions to astrophysical problems.'
long_description = (pathlib.Path(__file__).parent / 'README.md').read_text()

setup(
    name='exact',
    version=__version__,
    author='Daniel Mentiplay',
    author_email='d.mentiplay@gmail.com',
    url='https://github.com/dmentipl/exact-astro',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=packages,
    license='MIT',
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
