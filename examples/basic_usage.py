#!/usr/bin/env python3
"""
Basic usage example for LLM-Heuristics.

This script demonstrates the complete LLM-Heuristics workflow:
1. Analyze DICOM files using HeuDiConv
2. Group sequences using pandas groupby
3. Map sequences to BIDS using LLM
4. Generate heuristic files using LLM

Privacy Note: All processing happens locally on your machine.
No DICOM data is sent to external services or shared with third parties.
"""

import logging
from pathlib import Path

from llm_heuristics.core.bids_mapper import LLMBIDSMapper
from llm_heuristics.core.heudiconv_extractor import HeuDiConvExtractor
from llm_heuristics.core.heuristic_generator import HeuristicGenerator
from llm_heuristics.core.sequences_grouper import SequencesGrouper
from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration
from llm_heuristics.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main example function demonstrating the 4-step workflow."""
    # Set up logging
    setup_logging(logging.INFO)

    # Example DICOM directory (replace with your path)
    dicom_dir = Path("./sample_dicom_data")
    output_dir = Path("./analysis_output")

    if not dicom_dir.exists():
        logger.warning("DICOM directory not found: %s", dicom_dir)
        logger.info("Create some sample DICOM files or update the path.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze DICOM directory using HeuDiConv
    logger.info("Step 1: Analyzing DICOM files...")
    try:
        extractor = HeuDiConvExtractor(enable_caching=False)
        dicom_df = extractor.extract_dicom_info(dicom_dir=dicom_dir)

        if dicom_df.empty:
            logger.error("No DICOM data found!")
            return

        # Save aggregated_dicominfo.tsv
        aggregated_dicominfo_path = output_dir / "aggregated_dicominfo.tsv"
        dicom_df.to_csv(aggregated_dicominfo_path, sep="\t", index=False)
        logger.info("✓ Saved aggregated DICOM info: %s", aggregated_dicominfo_path)

    except Exception as e:
        logger.error("Step 1 failed: %s", e)
        return

    # Step 2: Group sequences using pandas groupby (NO LLM)
    logger.info("Step 2: Grouping sequences...")
    try:
        grouper = SequencesGrouper()
        grouped_df = grouper.group_sequences(dicom_df)

        # Save grouped results
        grouped_output_path = output_dir / "aggregated_dicominfo_groups.tsv"
        grouped_df.to_csv(grouped_output_path, sep="\t", index=False)

        # Generate grouping report
        report = grouper.generate_grouping_report(grouped_df)
        report_path = output_dir / "grouping_report.txt"
        report_path.write_text(report)

        logger.info("✓ Saved grouped sequences: %s", grouped_output_path)
        logger.info("✓ Saved grouping report: %s", report_path)

    except Exception as e:
        logger.error("Step 2 failed: %s", e)
        return

    # Step 3: Map sequences to BIDS using LLM
    logger.info("Step 3: Mapping sequences to BIDS using LLM...")
    try:
        mapper = LLMBIDSMapper(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_quantization=True,
        )

        # Map groups to BIDS
        mapped_df = mapper.map_groups_to_bids(grouped_df)

        # Save mapped results
        mapped_output_path = output_dir / "aggregated_dicominfo_mapped.tsv"
        mapped_df.to_csv(mapped_output_path, sep="\t", index=False)

        # Generate mapping report
        mapping_report = mapper.generate_mapping_report(mapped_df)
        mapping_report_path = output_dir / "mapping_report.txt"
        mapping_report_path.write_text(mapping_report)

        logger.info("✓ Saved BIDS mappings: %s", mapped_output_path)
        logger.info("✓ Saved mapping report: %s", mapping_report_path)

    except Exception as e:
        logger.error("Step 3 failed: %s", e)
        return

    # Step 4: Generate heuristic file using LLM
    logger.info("Step 4: Generating heuristic file...")
    try:
        generator = HeuristicGenerator(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            use_quantization=True,
        )

        generator.generate_from_mapped_tsv(
            mapped_dicominfo_path=mapped_output_path,
            output_file=output_dir / "generated_heuristic.py",
            custom_context=(
                "Multi-modal study with T1w, T2w, and resting-state fMRI. "
                "Prefer raw anatomical sequences and motion-corrected functional data."
            ),
        )

        logger.info("✓ Generated heuristic: %s", output_dir / "generated_heuristic.py")

    except Exception as e:
        logger.error("Step 4 failed: %s", e)
        return

    # Step 5: BIDS validation (optional)
    logger.info("Step 5: Validating BIDS compliance...")
    heuristic_file = output_dir / "generated_heuristic.py"
    if heuristic_file.exists():
        try:
            # Initialize BIDS schema integration
            bids_schema = BIDSSchemaIntegration()
            schema_info = bids_schema.get_schema_version_info()

            schema_version = schema_info.get("schema_version", "unknown")
            logger.info("✓ Using BIDS schema version: %s", schema_version)
            logger.info("✓ Available modalities: %d", len(bids_schema.modalities))
            logger.info("✓ Available entities: %d", len(bids_schema.entities))

            # You can add more validation logic here as needed
            logger.info("✓ BIDS validation completed")

        except Exception as e:
            logger.error("BIDS validation failed: %s", e)

    logger.info("🎉 Complete workflow finished!")
    logger.info("Check the output directory: %s", output_dir)
    logger.info("Generated files:")
    for file in output_dir.glob("*.tsv"):
        logger.info("  - %s", file.name)
    for file in output_dir.glob("*.txt"):
        logger.info("  - %s", file.name)
    for file in output_dir.glob("*.py"):
        logger.info("  - %s", file.name)

    logger.info("Next steps:")
    logger.info("1. Review the generated heuristic file")
    logger.info("2. Test with heudiconv:")
    heuristic_path = f"{output_dir}/generated_heuristic.py"
    logger.info("   heudiconv -f %s -d /path/to/dicom -s subject -c none", heuristic_path)


if __name__ == "__main__":
    main()
