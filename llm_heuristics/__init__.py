"""LLM-based DICOM header analysis and heuristic file generation for heudiconv.

Privacy Note: All processing happens locally on your machine.
No DICOM data is sent to external services or shared with third parties.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0+unknown"

__author__ = "LLM Heuristics Developers"
__license__ = "BSD-3-Clause"
__maintainer__ = "LLM Heuristics Developers"

from llm_heuristics.core.heudiconv_extractor import HeuDiConvExtractor
from llm_heuristics.core.heuristic_generator import HeuristicGenerator

__all__ = [
    "__version__",
    "HeuDiConvExtractor",
    "HeuristicGenerator",
]
