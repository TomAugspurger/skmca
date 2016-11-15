from setuptools import setup, find_packages
from os import path

import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='skmca',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    description='scikit-learn compatible Multiple Correspondence Analysis',
    long_description=long_description,

    url='https://github.com/tomaugspurger/skmca',

    author='Tom Augspurger',
    author_email='tom.augspurger88@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # keywords='sample setuptools development',
    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=[
        'scikit-learn>=0.18',
        'pandas>=0.19.0',
    ],

    extras_require={
        'dev': ['pytest', 'pytest-cov', 'sphinx', 'sphinx_rtd_theme',
                'numpydoc']
    },
    include_package_data=True,
)
