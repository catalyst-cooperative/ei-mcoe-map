"""Tools for generating MCOE for all US electric power plants."""

import pkg_resources
from pathlib import Path

import ei_mcoe.ei_mcoe

REPO_DIR = Path(__file__).resolve().parent.parent.parent

INPUTS_DIR = REPO_DIR / 'inputs'
"""
Directory of input files that are used in generating the MCOE outputs.
"""

OUTPUTS_DIR = REPO_DIR / 'outputs'
"""
Directory of output files that are generated from the MCOE processes.
"""

__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "cgosnell@catalyst.coop"
__version__ = pkg_resources.get_distribution('ei_mcoe').version
__docformat__ = "restructuredtext en"
__description__ = "This repository is a collaboration between Catalyst and Energy Innovations to output marginal cost of electricity for every coal plant across the US."
