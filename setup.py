#!/usr/bin/env python

from distutils.core import setup

execfile("packages/pygsti/_version.py")

setup(name='pyGSTi',
      version=__version__,
      description='A python implementation of Gate Set Tomography',
      long_description='TODO: long description',
      author='Erik Nielsen, Kenneth Rundinger, John Gamble, Robin Blume-Kohout',
      author_email='pygsti@sandia.gov',
      url='TODO: url',
      packages=['pygsti', 'pygsti.algorithms', 'pygsti.construction', 'pygsti.drivers', 'pygsti.io', 'pygsti.objects', 'pygsti.optimize', 'pygsti.report', 'pygsti.tools'],
      package_dir={'': 'packages'},
      package_data={'pygsti.report': ['templates/*.tex', 'templates/*.pptx']},
     )
