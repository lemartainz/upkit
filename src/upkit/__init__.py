# upkit/__init__.py

"""
upkit
=====

A Python toolkit for nuclear and particle physics research. 

This package provides a collection of utilities for data analysis, including:
    - ROOT integration
    - Data visualization
    - Fitting and statistical analysis

Modules
=======
- upkit.root: Open ROOT files and access specific trees / branches through awkward arrays
- upkit.hists: Create and manipulate histograms for data visualization
- upkit.fit: Perform fitting and statistical analysis on data
- upkit.tools: Utility functions for data manipulation and analysis

Example
=======
    from upkit import root, hists, fit
"""

from .hipo_analysis import RootAnalysis
from .hists import Histo, Histo2D
from .fit import Fit
from . import tools 

# Define the public API
__all__ = ["RootAnalysis", "Histo", "Histo2D", "Fit", "tools"]

# Package version
__version__ = "0.1.0"