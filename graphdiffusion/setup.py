from setuptools import setup, find_packages

# List of dependencies extracted from your environment.yml
# This list should be the list of pip dependencies. Conda dependencies need to be handled separately.
dependencies = [
    # Add other dependencies as needed
]

setup(
    name='graphdiffusion',
    version='0.0.0.1',
    packages=find_packages(),
    install_requires=dependencies
    # Add other parameters as needed
)
