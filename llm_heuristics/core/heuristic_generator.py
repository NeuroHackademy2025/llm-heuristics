"""Main heuristic generator that orchestrates DICOM analysis and LLM generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from llm_heuristics.core.heudiconv_extractor import HeuDiConvExtractor
from llm_heuristics.core.series_grouper import SeriesGrouper
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
            LLM model name to use (only used for generate command)
        use_quantization : bool
            Whether to use quantization for the model (only used for generate command)
        n_cpus : int | None
            Number of CPU cores to use for parallel processing (only used for analyze command)
        slurm : bool
            Whether to generate SLURM job scripts instead of running directly
            (only used for analyze command)
        **model_kwargs
            Additional arguments for the model (only used for generate command)
        """
        # Only initialize components based on what's needed
        # For analyze command: only need dicom_extractor
        # For generate command: only need llm_model and template
        self.dicom_extractor = HeuDiConvExtractor(n_cpus=n_cpus, slurm=slurm)
        self.series_grouper = SeriesGrouper()
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
        additional_context: str | None = None,
    ) -> str:
        """
        Generate a heuristic file from pre-mapped dicominfo_mapped.tsv file.

        Parameters
        ----------
        mapped_dicominfo_path : Path
            Path to aggregated_dicominfo_mapped.tsv file created by map command
        output_file : Path | None
            Path to save the generated heuristic file
        additional_context : str | None
            Additional context for heuristic generation

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

        # Generate heuristic using LLM with pre-mapped data
        logger.info("Generating heuristic with LLM using pre-mapped data...")
        heuristic_content = self.llm_model.generate_heuristic(
            dicom_summary=mapped_with_bids,
            study_info=study_info,
            additional_context=additional_context,
        )

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(heuristic_content)
            logger.info("Heuristic saved to: %s", output_file)

        logger.info("Heuristic generation completed successfully")
        return heuristic_content

    def analyze_and_save_dicom_info(
        self, dicom_dir: Path, output_file: Path | None = None
    ) -> tuple[pd.DataFrame, str]:
        """
        Analyze DICOM directory and return both DataFrame and report.

        Parameters
        ----------
        dicom_dir : Path
            Path to DICOM directory
        output_file : Path | None
            Path to save the report

        Returns
        -------
        tuple[pd.DataFrame, str]
            Tuple of (dicom_dataframe, human_readable_report)
        """
        logger.info("Analyzing DICOM directory: %s", dicom_dir)

        # Extract DICOM information using HeuDiConv
        dicom_df = self.dicom_extractor.extract_dicom_info(dicom_dir)

        # Generate human-readable report
        summary = self.dicom_extractor.generate_summary(dicom_df)

        # Add detailed series information
        detailed_report = [
            summary,
            "\n" + "=" * 80,
            "\nDetailed Series Information:",
            "=" * 80,
        ]

        for _idx, row in dicom_df.iterrows():
            # Use HeuDiConv's column names
            series_info = [f"\nSeries {row.get('series_id', 'Unknown')}:"]

            # Add available information with safe access
            for field, label in [
                ("protocol_name", "Protocol"),
                ("series_description", "Description"),
                ("dim1", "Matrix X"),
                ("dim2", "Matrix Y"),
                ("dim3", "Slices"),
                ("dim4", "Timepoints"),
                ("TR", "TR (ms)"),
                ("TE", "TE (ms)"),
                ("is_motion_corrected", "Motion Corrected"),
                ("subject_id", "Subject"),
                ("session_id", "Session"),
            ]:
                if field in row and pd.notna(row[field]):
                    series_info.append(f"  {label}: {row[field]}")

            detailed_report.extend(series_info)

        report = "\n".join(detailed_report)

        # Save report to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report)
            logger.info("Report saved to: %s", output_file)

        return dicom_df, report
