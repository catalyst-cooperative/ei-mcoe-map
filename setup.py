"""Setup script to allow ei_mcoe to be installed with pip."""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text()

setup(
    name='ei_mcoe',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='This repository is a collaboration between Catalyst and Energy Innovations to output marginal cost of electricity for every coal plant across the US.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/catalyst-cooperative/ei-mcoe-map/',
    license="MIT",
    version='0.1.0',
    install_requires=[
        "catalystcoop.pudl @ git+https://github.com/catalyst-cooperative/pudl.git@dev",
        "fredapi",
    ],
    python_requires=">=3.8,<3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    author="Catalyst Cooperative",
    author_email="pudl@catalyst.coop",
    maintainer="Christina Gosnell",
    maintainer_email="cgosnell@catalyst.coop",
    keywords=['mcoe', 'ferc1', 'eia', 'coal', 'coal crossover']
)
