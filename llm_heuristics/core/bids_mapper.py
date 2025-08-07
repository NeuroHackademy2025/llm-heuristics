"""LLM-based BIDS mapping functionality."""

import json
import logging
from typing import Any

import pandas as pd

from llm_heuristics.models.llama_model import LlamaModel
from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration

logger = logging.getLogger(__name__)


class LLMBIDSMapper:
    """LLM-based BIDS mapping for grouped DICOM series."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        use_quantization: bool = True,
        **model_kwargs,
    ):
        """
        Initialize the LLM BIDS mapper.

        Parameters
        ----------
        model_name : str
            LLM model name to use for mapping
        use_quantization : bool
            Whether to use quantization for the model
        **model_kwargs
            Additional arguments for the model
        """
        self.llm_model = LlamaModel(
            model_name=model_name,
            use_quantization=use_quantization,
            **model_kwargs,
        )
        self.bids_schema = BIDSSchemaIntegration()

    def map_groups_to_bids(self, grouped_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map grouped DICOM series to BIDS using LLM analysis.

        Parameters
        ----------
        grouped_df : pd.DataFrame
            DataFrame with grouped series from SeriesGrouper

        Returns
        -------
        pd.DataFrame
            DataFrame with BIDS mapping columns added
        """
        logger.info("Mapping %d groups to BIDS using LLM analysis", len(grouped_df))

        # Prepare BIDS schema context
        bids_context = self._prepare_bids_context()

        # Process groups in batches to avoid token limits
        batch_size = 10
        mapped_groups = []

        for i in range(0, len(grouped_df), batch_size):
            batch = grouped_df.iloc[i : i + batch_size]
            logger.info(
                "Processing batch %d/%d",
                i // batch_size + 1,
                (len(grouped_df) + batch_size - 1) // batch_size,
            )

            batch_mappings = self._map_batch_to_bids(batch, bids_context)
            mapped_groups.extend(batch_mappings)

        # Create result DataFrame
        result_df = grouped_df.copy()

        # Add BIDS mapping columns
        mapping_df = pd.DataFrame(mapped_groups)
        bids_columns = [
            "bids_modality",
            "bids_suffix",
            "bids_entities",
            "bids_confidence",
            "bids_path",
        ]
        for col in bids_columns:
            if col in mapping_df.columns:
                result_df[col] = mapping_df[col]

        logger.info("Successfully mapped %d groups to BIDS", len(result_df))
        return result_df

    def _get_mri_modalities(self) -> list[str]:
        """Dynamically get MRI modalities from BIDS schema."""
        try:
            # Get MRI datatypes from BIDS schema
            mri_datatypes = self.bids_schema.bids_schema["rules"]["modalities"]["mri"]["datatypes"]
            if isinstance(mri_datatypes, list):
                return mri_datatypes
            else:
                logger.warning("MRI datatypes not found in schema, using fallback")
                return ["anat", "func", "fmap", "dwi", "perf"]
        except (KeyError, TypeError) as e:
            logger.warning("Could not extract MRI modalities from schema: %s, using fallback", e)
            return ["anat", "func", "fmap", "dwi", "perf"]

    def _prepare_bids_context(self) -> str:
        """Prepare comprehensive BIDS context for LLM."""
        # Get BIDS schema information
        schema_info = self.bids_schema.get_schema_version_info()

        # Get MRI-specific modalities dynamically from schema
        mri_modality_list = self._get_mri_modalities()
        mri_modalities = {}
        for modality_name, modality_info in self.bids_schema.modalities.items():
            if modality_name in mri_modality_list:
                mri_modalities[modality_name] = {
                    "name": modality_info.get("name", ""),
                    "description": modality_info.get("description", ""),
                    "suffixes": list(modality_info.get("suffixes", {}).keys()),
                }

        # Get entity information and order
        entities_info = {}
        for entity_name, entity_info in self.bids_schema.entities.items():
            entities_info[entity_name] = {
                "name": entity_info.get("name", ""),
                "description": entity_info.get("description", ""),
                "format": entity_info.get("format", "string"),
            }

        # Get entity order from schema
        entity_order = self.bids_schema.get_entity_order()

        context = f"""
BIDS Schema Information (v{schema_info.get("version", "unknown")}):

MRI MODALITIES AND SUFFIXES:
{json.dumps(mri_modalities, indent=2)}

ENTITIES (for path construction):
{json.dumps(entities_info, indent=2)}

ENTITY ORDER (for path construction):
{json.dumps(entity_order, indent=2)}

TASK: Map DICOM series groups to appropriate BIDS modality, suffix, and entities.

CONSTRAINTS:
- Only use MRI-related modalities: {", ".join(mri_modality_list)}
- Use only suffixes that exist in the BIDS schema
- Assign entities based on DICOM information when possible
- Follow entity order when constructing BIDS paths
- Generate confidence score (0.0-1.0) for each mapping
- Create BIDS path template: modality/sub-{{subject}}_ses-{{session}}_[entities]_suffix

EXAMPLE OUTPUT FORMAT:
{{
  "bids_modality": "anat",
  "bids_suffix": "T1w",
  "bids_entities": {{"acq": "MPRAGE"}},
  "bids_confidence": 0.95,
  "bids_path": "anat/sub-{{subject}}_ses-{{session}}_acq-MPRAGE_T1w"
}}
"""
        return context

    def _map_batch_to_bids(
        self, batch_df: pd.DataFrame, bids_context: str
    ) -> list[dict[str, Any]]:
        """Map a batch of groups to BIDS using LLM."""
        # Prepare batch data for LLM
        batch_data = []
        for _, row in batch_df.iterrows():
            group_info = {
                "protocol_name": str(row.get("protocol_name", "Unknown")),
                "series_description": str(row.get("series_description", "Unknown")),
                "sequence_name": str(row.get("sequence_name", "Unknown")),
                "dimensions": (
                    f"{row.get('dim1', 0)}x{row.get('dim2', 0)}x"
                    f"{row.get('dim3', 0)}x{row.get('dim4', 1)}"
                ),
                "TR": row.get("TR", 0),
                "TE": row.get("TE", 0),
                "is_motion_corrected": row.get("is_motion_corrected", False),
                "image_type": str(row.get("image_type", "Unknown")),
                "series_count": row.get("series_count", 1),
            }
            batch_data.append(group_info)

        # Create prompt for LLM
        mri_modality_list = self._get_mri_modalities()
        prompt = f"""
{bids_context}

DICOM SERIES GROUPS TO MAP:
{json.dumps(batch_data, indent=2)}

Please map each group to BIDS format. Return a JSON array with one mapping per group,
in the same order as the input groups.

Each mapping should include:
- bids_modality: one of {mri_modality_list}
- bids_suffix: appropriate suffix from BIDS schema
- bids_entities: dictionary of entities (e.g., {{"acq": "value", "task": "value"}})
- bids_confidence: confidence score (0.0-1.0)
- bids_path: full BIDS path template

Return only the JSON array, no additional text.
"""

        try:
            # Get LLM response
            response = self.llm_model._call_model(prompt)

            # Parse JSON response
            mappings = json.loads(response.strip())

            # Validate and clean up mappings
            validated_mappings = []
            for i, mapping in enumerate(mappings):
                validated_mapping = self._validate_mapping(mapping, batch_data[i])
                validated_mappings.append(validated_mapping)

            return validated_mappings

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Failed to parse LLM response for batch, using fallback: %s", e)
            # Fallback to basic mapping
            return [self._fallback_mapping(group) for group in batch_data]

    def _validate_mapping(self, mapping: dict, group_info: dict) -> dict[str, Any]:
        """Validate and clean up LLM mapping result."""
        # Ensure required fields exist
        validated = {
            "bids_modality": mapping.get("bids_modality", "unknown"),
            "bids_suffix": mapping.get("bids_suffix", "unknown"),
            "bids_entities": mapping.get("bids_entities", {}),
            "bids_confidence": float(mapping.get("bids_confidence", 0.5)),
            "bids_path": mapping.get("bids_path", "unmapped"),
        }

        # Validate modality
        valid_modalities = self._get_mri_modalities()
        if validated["bids_modality"] not in valid_modalities:
            validated["bids_modality"] = "unknown"
            validated["bids_confidence"] *= 0.5

        # Ensure entities is a dict
        if not isinstance(validated["bids_entities"], dict):
            validated["bids_entities"] = {}

        # Validate path format
        if not validated["bids_path"] or validated["bids_path"] == "unmapped":
            modality = validated["bids_modality"]
            suffix = validated["bids_suffix"]
            entities = validated["bids_entities"]

            if modality != "unknown" and suffix != "unknown":
                entity_str = "_".join([f"{k}-{v}" for k, v in entities.items()])
                entity_part = f"_{entity_str}" if entity_str else ""
                validated["bids_path"] = (
                    f"{modality}/sub-{{subject}}_ses-{{session}}{entity_part}_{suffix}"
                )
            else:
                validated["bids_path"] = "unmapped"

        return validated

    def _fallback_mapping(self, group_info: dict) -> dict[str, Any]:
        """Provide fallback mapping when LLM fails."""
        protocol = group_info["protocol_name"].lower()
        description = group_info["series_description"].lower()
        mri_modalities = self._get_mri_modalities()

        # Simple fallback rules
        if any(term in protocol + description for term in ["t1", "mprage", "anatomical"]):
            return {
                "bids_modality": "anat" if "anat" in mri_modalities else "unknown",
                "bids_suffix": "T1w",
                "bids_entities": {},
                "bids_confidence": 0.6,
                "bids_path": (
                    "anat/sub-{subject}_ses-{session}_T1w"
                    if "anat" in mri_modalities
                    else "unmapped"
                ),
            }
        elif any(term in protocol + description for term in ["t2", "flair"]):
            return {
                "bids_modality": "anat" if "anat" in mri_modalities else "unknown",
                "bids_suffix": "T2w",
                "bids_entities": {},
                "bids_confidence": 0.6,
                "bids_path": (
                    "anat/sub-{subject}_ses-{session}_T2w"
                    if "anat" in mri_modalities
                    else "unmapped"
                ),
            }
        elif any(term in protocol + description for term in ["bold", "fmri", "functional"]):
            return {
                "bids_modality": "func" if "func" in mri_modalities else "unknown",
                "bids_suffix": "bold",
                "bids_entities": {"task": "unknown"},
                "bids_confidence": 0.6,
                "bids_path": (
                    "func/sub-{subject}_ses-{session}_task-unknown_bold"
                    if "func" in mri_modalities
                    else "unmapped"
                ),
            }
        else:
            return {
                "bids_modality": "unknown",
                "bids_suffix": "unknown",
                "bids_entities": {},
                "bids_confidence": 0.3,
                "bids_path": "unmapped",
            }

    def generate_mapping_report(self, mapped_df: pd.DataFrame) -> str:
        """Generate a summary report of the BIDS mapping results."""
        mapped_count = len(mapped_df[mapped_df["bids_path"] != "unmapped"])
        total_count = len(mapped_df)
        mapping_rate = (mapped_count / total_count) * 100 if total_count > 0 else 0

        # Count by modality
        modality_counts = mapped_df["bids_modality"].value_counts()

        report_lines = [
            "BIDS Mapping Report",
            "=" * 40,
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total groups: {total_count}",
            f"Successfully mapped: {mapped_count}",
            f"Mapping success rate: {mapping_rate:.1f}%",
            "",
            "Mapping by Modality:",
            "-" * 20,
        ]

        for modality, count in modality_counts.items():
            report_lines.append(f"  {modality}: {count} groups")

        # Show top mapped groups
        successful_mappings = mapped_df[mapped_df["bids_path"] != "unmapped"]
        if not successful_mappings.empty:
            report_lines.extend(
                [
                    "",
                    "Top Successful Mappings:",
                    "-" * 25,
                ]
            )

            top_mappings = successful_mappings.nlargest(5, "series_count")
            for _, row in top_mappings.iterrows():
                protocol = str(row.get("protocol_name", "Unknown"))[:20]
                bids_path = str(row.get("bids_path", "Unknown"))[:40]
                confidence = row.get("bids_confidence", 0)
                count = row.get("series_count", 0)

                report_lines.append(
                    f"  {count:3d} series: {protocol:20} → {bids_path} (conf: {confidence:.2f})"
                )

        report_lines.extend(
            [
                "",
                "Next Steps:",
                "-----------",
                "1. Review mapping results and adjust if needed",
                "2. Run 'llm-heuristics generate <output_dir>' to create heuristics",
                "",
                "Note: Mapped groups are ready for heuristic generation.",
            ]
        )

        return "\n".join(report_lines)
