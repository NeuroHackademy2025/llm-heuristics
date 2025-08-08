"""Main heuristic generator that orchestrates DICOM analysis and LLM generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from llm_heuristics.core.heudiconv_extractor import HeuDiConvExtractor
from llm_heuristics.core.sequences_grouper import SequencesGrouper
from llm_heuristics.models.llama_model import LlamaModel
from llm_heuristics.utils.templates import HeuristicTemplate

logger = logging.getLogger(__name__)


class HeuristicGenerator:
    """Main class for generating heudiconv heuristic files using LLM analysis.

    Privacy Note: All processing happens locally on your machine.
    No DICOM data is sent to external services or shared with third parties.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        use_quantization: bool = True,
        n_cpus: int | None = None,
        slurm: bool = False,
        **model_kwargs,
    ) -> None:
        """
        Initialize the heuristic generator.

        Parameters
        ----------
        model_name : str
            LLM model name to use (used for map and generate commands)
        use_quantization : bool
            Whether to use quantization for the model (used for map and generate commands)
        n_cpus : int | None
            Number of CPU cores to use for parallel processing (only used for analyze command)
        slurm : bool
            Whether to generate SLURM job scripts instead of running directly
            (only used for analyze command)
        **model_kwargs
            Additional arguments for the model (used for map and generate commands)
        """
        # Only initialize components based on what's needed
        # For analyze command: only need dicom_extractor
        # For generate command: need llm_model and template
        # For map command: need llm_model
        self.dicom_extractor = HeuDiConvExtractor(n_cpus=n_cpus, slurm=slurm)
        self.sequences_grouper = SequencesGrouper()

        # LLM model is needed for both map and generate commands
        self.llm_model = LlamaModel(
            model_name=model_name,
            use_quantization=use_quantization,
            **model_kwargs,
        )

        self.template = HeuristicTemplate()

    def generate_from_mapped_tsv(
        self,
        mapped_dicominfo_path: Path,
        output_file: Path | None = None,
        custom_context: str | None = None,
    ) -> str:
        """
        Generate a heuristic file from pre-mapped dicominfo_mapped.tsv file.

        Parameters
        ----------
        mapped_dicominfo_path : Path
            Path to aggregated_dicominfo_mapped.tsv file created by map command
        output_file : Path | None
            Path to save the generated heuristic file
        custom_context : str | None
            Custom context for sequence selection rules, e.g.:
            "for func only use motion_corrected, for T1w only use Norm sequence"

        Returns
        -------
        str
            Generated heuristic file content
        """
        logger.info("Generating heuristic from mapped dicominfo: %s", mapped_dicominfo_path)

        # Read the pre-mapped DICOM data
        mapped_with_bids = pd.read_csv(mapped_dicominfo_path, sep="\t")

        if mapped_with_bids.empty:
            raise ValueError(f"No data found in {mapped_dicominfo_path}")

        # Prepare study information
        study_info = {
            "dicom_dir": str(mapped_dicominfo_path.parent),
            "num_series": mapped_with_bids["series_count"].sum(),
            "num_unique_groups": len(mapped_with_bids),
        }

        # Generate heuristic using LLM with mapped outputs
        logger.info("Generating heuristic with LLM using mapped outputs...")

        # Use the LLM to generate the complete heuristic with sequence logic
        heuristic_content = self.llm_model.generate_heuristic(
            dicom_summary=mapped_with_bids,
            study_info=study_info,
            additional_context=custom_context,
        )

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(heuristic_content)
            logger.info("Heuristic saved to: %s", output_file)

        logger.info("Heuristic generation completed successfully")
        return heuristic_content
