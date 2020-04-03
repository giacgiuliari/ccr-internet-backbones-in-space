from distutils.core import setup

setup(
    name='ccr-submission-code',
    version='1.0',
    packages=['analysis', 'lib', 'preproc'],
    license='ETH Zurich',
    long_description=open('README.md').read(),
)