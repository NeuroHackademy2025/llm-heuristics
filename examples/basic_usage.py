#!/usr/bin/env python3
"""
Basic usage example for LLM-Heuristics.

This script demonstrates how to use LLM-Heuristics to:
1. Analyze DICOM files
2. Generate heuristic files

Privacy Note: All processing happens locally on your machine.
No DICOM data is sent to external services or shared with third parties.
"""

import logging
from pathlib import Path

from llm_heuristics import HeuristicGenerator
from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration
from llm_heuristics.utils.logging import setup_logging


def main():
    """Main example function."""
    # Set up logging
    setup_logging(logging.INFO)

    # Example DICOM directory (replace with your path)
    dicom_dir = Path("./sample_dicom_data")
    output_dir = Path("./analysis_output")

    if not dicom_dir.exists():
        return

    # Initialize the heuristic generator

    # Example 1: Default privacy (recommended for clinical data)
    generator_default = HeuristicGenerator(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_quantization=True,
    )

    # Use the default generator for the rest of the examples
    generator = generator_default

    # Example 1: Analyze DICOM directory (required first step)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        dicom_df, report = generator.analyze_and_save_dicom_info(
            dicom_dir=dicom_dir, output_file=None
        )

        # Save aggregated_dicominfo.tsv for generate command
        aggregated_dicominfo_path = output_dir / "aggregated_dicominfo.tsv"
        dicom_df.to_csv(aggregated_dicominfo_path, sep="\t", index=False)

    except Exception:
        return

    # Example 2: Generate heuristic file (uses output from step 1)

    try:
        heuristic_content = generator.generate_from_dicominfo_tsv(
            dicominfo_path=aggregated_dicominfo_path,
            output_file=Path("generated_heuristic.py"),
            additional_context=(
                "Multi-modal study with T1w, T2w, and resting-state fMRI. "
                "Prefer raw anatomical sequences and motion-corrected functional data."
            ),
        )

    except Exception:
        return

    # Example 3: BIDS validation

    heuristic_file = Path("generated_heuristic.py")
    if heuristic_file.exists():
        try:
            # Initialize BIDS schema integration
            bids_schema = BIDSSchemaIntegration()

            # Load the heuristic content
            heuristic_content = heuristic_file.read_text()

            # Validate using the template
            from llm_heuristics.utils.templates import HeuristicTemplate

            template = HeuristicTemplate()
            validation = template.validate_generated_heuristic_paths(heuristic_content)

            if validation.get("overall_valid", False):
                pass
            else:
                pass

            if validation.get("valid_paths"):
                pass
            if validation.get("invalid_paths"):
                pass
            if validation.get("warnings"):
                pass

        except Exception:
            pass

    # Example 4: Show BIDS schema info

    try:
        bids_schema = BIDSSchemaIntegration()
        schema_info = bids_schema.get_schema_version_info()

        for _key, _value in schema_info.items():
            pass

        for _modality in bids_schema.modalities:
            pass

    except Exception:
        pass


if __name__ == "__main__":
    main()
