"""Base interface for LLM models."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import pandas as pd


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces.

    Privacy Note: All LLM processing happens locally on your machine.
    No data is sent to external services or shared with third parties.
    """

    @abstractmethod
    def generate_heuristic(
        self,
        dicom_summary: pd.DataFrame,
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """
        Generate a heuristic file based on DICOM metadata.

        Parameters
        ----------
        dicom_summary : pd.DataFrame
            DataFrame containing series-level DICOM metadata
        study_info : dict[str, str]
            Additional study information (subject, session, etc.)
        additional_context : str | None
            Additional context or requirements for heuristic generation

        Returns
        -------
        str
            Generated heuristic file content
        """

    @abstractmethod
    def analyze_sequence_mapping(
        self,
        series_descriptions: list[str],
        protocol_names: list[str],
    ) -> dict[str, str]:
        """
        Analyze sequence descriptions and suggest BIDS mappings.

        Parameters
        ----------
        series_descriptions : list[str]
            List of series descriptions
        protocol_names : list[str]
            List of protocol names

        Returns
        -------
        dict[str, str]
            Mapping of series to suggested BIDS names
        """
