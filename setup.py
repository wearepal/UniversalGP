"""Setup config for installation"""
from setuptools import setup

setup(
    name='universalgp',
    version='0.1',
    description='A basic generic implementation of Gaussian Processes',
    url='https://github.com/predictive-analytics-lab/UniversalGP',
    author='PAL group',
    keywords="Gaussian process",
    author_email='pal@sussex.ac.uk',
    license='GPL-2.0',
    packages=['universalgp',
              'universalgp.cov',
              'universalgp.mean',
              'universalgp.datasets',
              'universalgp.inf',
              'universalgp.lik',
              'universalgp.util'],

)
