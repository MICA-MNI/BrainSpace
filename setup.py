"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path
from io import open as io_open


TEST_REQUIRES = [
        # testing and coverage
        'pytest', 'coverage', 'pytest-cov', 'coveralls',
    ]

INSTALL_REQUIRES = ['numpy>=1.20',
                    'scipy>=1.6',
                    'scikit-learn>=0.22',
                    'matplotlib>=3.3',
                    'vtk>=9.1.0,<9.7.0',
                    'nibabel>=3.2']

EXAMPLES_REQUIRES = ['nilearn>=0.9']


here = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = None
version_file = path.join(here, 'brainspace/_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())


setup(

    name='brainspace',
    version=__version__,
    description='Cortical gradients and beyond',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/MICA-MNI/BrainSpace',
    author='BrainSpace developers',
    author_email='enning.yang@mcgill.ca',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='brain cortex gradient manifold',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES,
        'examples': EXAMPLES_REQUIRES,
    },
    include_package_data=True,
    zip_safe=False,
    # package_data={  # Optional
    #     'mydata': ['brainspace_data/*'],
    # },
    project_urls={  # Optional
        'Documentation': 'https://brainspace.readthedocs.io',
        'Bug Reports': 'https://github.com/MICA-MNI/BrainSpace/issues',
        'Source': 'https://github.com/MICA-MNI/BrainSpace',
    },
    download_url='https://github.com/MICA-MNI/BrainSpace/archive/'
                 '{ver}.tar.gz'.format(ver=__version__),
)
