"""Templates and utilities for heuristic file generation."""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import Template

from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration

logger = logging.getLogger(__name__)


class HeuristicTemplate:
    """Template manager for generating heuristic files."""

    def __init__(self) -> None:
        """Initialize the template manager."""
        self.base_template = Template(self._get_base_heuristic_template())
        self.prompt_template = Template(self._get_prompt_template())

        # Initialize BIDS schema integration
        try:
            self.bids_schema = BIDSSchemaIntegration()
            logger.info("BIDS schema integration initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize BIDS schema integration: %s", e)
            self.bids_schema = None

        # Cache for common BIDS patterns to avoid regenerating them
        self._bids_patterns_cache = None
        self._all_schema_patterns_cache = None

    def create_heuristic_prompt(
        self,
        sequences_info: list[dict[str, Any]],
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """
        Create a prompt for LLM heuristic generation using mapped BIDS data.

        Parameters
        ----------
        sequences_info : list[dict[str, Any]]
            Mapped BIDS information for each series group
        study_info : dict[str, str]
            Study-level information
        additional_context : str | None
            Additional context or requirements

        Returns
        -------
        str
            Formatted prompt for the LLM
        """
        return self.prompt_template.render(
            sequences_info=sequences_info,
            study_info=study_info,
            additional_context=additional_context or "",
            num_series=len(sequences_info),
        )

    def generate_heuristic_skeleton(
        self,
        series_mappings: dict[str, str],
        study_info: dict[str, str],
        custom_context: str | None = None,
    ) -> str:
        """
        Generate a heuristic file skeleton.

        Parameters
        ----------
        series_mappings : dict[str, str]
            Mapping of series identifiers to BIDS patterns
        study_info : dict[str, str]
            Study information
        custom_context : str | None
            Custom context for sequence selection rules

        Returns
        -------
        str
            Heuristic file skeleton
        """
        return self.base_template.render(
            series_mappings=series_mappings,
            study_info=study_info,
            custom_context=custom_context or "",
        )

    def get_create_key_function(self) -> str:
        """Get the standard create_key function definition."""
        return '''def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    """Create a key for heudiconv template mapping."""
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes'''

    def _get_base_heuristic_template(self) -> str:
        """Get the base heuristic file template by importing from heudiconv.

        Raises
        ------
        ImportError
            If heudiconv is not installed
        RuntimeError
            If heudiconv template cannot be extracted
        """
        return self._load_heudiconv_template()

    def _load_heudiconv_template(self) -> str:
        """Return the heuristic skeleton dynamically from heudiconv's convertall.py.

        This template extracts the actual source code from heudiconv's convertall.py
        and modifies it using Jinja templating, ensuring full compatibility with
        heudiconv standards while allowing customization.

        Raises
        ------
        ImportError
            If heudiconv is not installed
        RuntimeError
            If convertall.py source cannot be extracted
        """
        try:
            import inspect

            import heudiconv.heuristics.convertall
        except ImportError as e:
            raise ImportError(f"heudiconv is required but not installed: {e}") from e

        try:
            # Get the actual source code from heudiconv's convertall.py
            convertall_source = inspect.getsource(heudiconv.heuristics.convertall)
        except Exception as e:
            raise RuntimeError(f"Failed to extract convertall.py source: {e}") from e

        # Extract header (everything before infotodict function)
        lines = convertall_source.split("\n")
        infotodict_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def infotodict("):
                infotodict_start = i
                break

        if infotodict_start is None:
            raise RuntimeError("Could not find infotodict function in convertall.py")

        # Header includes everything up to infotodict
        header_lines = lines[:infotodict_start]
        convertall_source_header = "\n".join(header_lines).strip()

        # Extract SeqInfo documentation from original infotodict docstring
        seqinfo_doc_lines = []
        in_seqinfo_doc = False
        for line in lines[infotodict_start:]:
            if "The namedtuple `s` contains the following fields:" in line:
                in_seqinfo_doc = True
                seqinfo_doc_lines.append(line.strip())
                continue
            if in_seqinfo_doc:
                if line.strip().startswith("*") or line.strip().startswith("-"):
                    seqinfo_doc_lines.append(line.strip())
                elif line.strip() == '"""' or (line.strip() and not line.strip().startswith("*")):
                    break

        seqinfo_documentation = (
            "\n    ".join(seqinfo_doc_lines)
            if seqinfo_doc_lines
            else "See heudiconv documentation for SeqInfo fields."
        )

        # Create a Jinja template that modifies the convertall source with placeholders
        template = f'''"""
Heuristic file generated automatically by llm-heuristics

Based on heudiconv's convertall.py with modifications for BIDS mapping.
Generated from BIDS mapped groups: {{{{ study_info.dicom_dir }}}}
Total mapped groups: {{{{ study_info.num_unique_groups }}}}
{{% if custom_context -%}}

Custom sequence selection rules:
{{{{ custom_context }}}}
{{% endif %}}

Original convertall.py structure preserved with the following modifications:
- infotodict() function customized with BIDS key mappings
- Logic added for sequence-specific BIDS assignment
"""

{convertall_source_header}

def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where.

    This function follows the heudiconv standard interface and uses the
    SeqInfo namedtuple structure. Customized for BIDS mapping based on
    sequence grouping and mapping analysis.
    {{% if custom_context -%}}

    Custom rules applied:
    {{{{ custom_context }}}}
    {{% endif %}}

    Parameters
    ----------
    seqinfo : list[SeqInfo]
        List of SeqInfo namedtuples containing DICOM metadata for each series.

    Returns
    -------
    dict[tuple[str, tuple[str, ...], None], list[str]]
        Dictionary mapping BIDS keys to lists of series IDs.

    Notes
    -----
    {seqinfo_documentation}
    """

    # Define BIDS key templates for each mapped group
    {{% for key, pattern in series_mappings.items() -%}}
    {{{{ key }}}} = create_key('{{{{ pattern }}}}')
    {{% endfor %}}

    # Initialize info dictionary following heudiconv standard
    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {{
        {{% for key in series_mappings.keys() -%}}
        {{{{ key }}}}: [],
        {{% endfor %}}
    }}

    # Iterate over sequences in the study and apply mapping logic
    for s in seqinfo:
        # TODO: Fill in mapping logic using sequence attributes from `s`
        # Standard SeqInfo attributes available:
        # s.series_id, s.protocol_name, s.series_description, s.dim1-4,
        # s.TR, s.TE, s.is_motion_corrected, s.series_files, etc.

        {{% if custom_context -%}}
        # Apply custom sequence selection rules as specified above
        {{% endif %}}

        # Example mapping logic (customize based on your mapped groups):
        # if "T1w" in s.series_description.upper():
        #     info[anat_T1w].append(s.series_id)
        # elif "bold" in s.series_description.lower():
        #     info[func_task_rest_bold].append(s.series_id)

        pass  # Remove this pass statement and add your mapping logic

    return info
'''

        return template

    def _get_prompt_template(self) -> str:
        """Get the LLM prompt template for mapped BIDS data."""
        return """You are an expert neuroimaging researcher who creates heudiconv
heuristic files for converting DICOM data to BIDS format.

We already ran a mapping step that grouped DICOM series and assigned BIDS
modalities, suffixes, and entities. Use ONLY the mapped output below to
produce a concise and correct heuristic that follows heudiconv conventions.

STUDY INFORMATION:
- DICOM Directory: {{ study_info.dicom_dir }}
- Total Mapped Groups: {{ study_info.num_unique_groups }}

MAPPED BIDS ASSIGNMENTS:
{% for group in sequences_info -%}
Group {{ loop.index }}:
  - BIDS Path: {{ group.bids_path }}
  - Modality: {{ group.bids_modality }}
  - Suffix: {{ group.bids_suffix }}
  - Entities: {{ group.bids_entities }}
  - Series Count: {{ group.series_count }}
  - Protocol: {{ group.protocol_name }}
  - Description: {{ group.series_description }}
  - Sequence: {{ group.sequence_name }}
  - Dimensions: {{ group.dim1 }}x{{ group.dim2 }}x{{ group.dim3 }}x{{ group.dim4 }}
  - TR: {{ group.TR }} ms
  - Motion Corrected: {{ group.is_motion_corrected }}

{% endfor %}

{% if additional_context -%}
CUSTOM CONTEXT:
{{ additional_context }}
{% endif %}

Please generate a complete heuristic.py file that:

1. Imports create_key from heudiconv.utils
2. Defines the infotodict function following heudiconv standards
3. Creates BIDS keys for each mapped group using the exact paths above
4. Implements conditional logic to assign sequences to the correct BIDS keys
5. Uses standard heudiconv SeqInfo fields (s.protocol_name, s.series_description, etc.)
6. Applies the custom context rules if provided

Important:
- Use the exact BIDS paths provided in the mappings
- Make sure the logic correctly identified each sequence type
- Follow heudiconv conventions for the infotodict function
- Be specific in your conditions to avoid misclassification

Generate the complete Python heuristic file:"""

    def get_common_bids_patterns(self) -> dict[str, str]:
        """Get ALL BIDS naming patterns directly from schema rules (no hardcoding)."""
        # Return cached patterns if available
        if self._bids_patterns_cache is not None:
            return self._bids_patterns_cache

        # Use existing BIDS schema integration if available, otherwise create one
        if hasattr(self, "bids_schema") and self.bids_schema is not None:
            bids_schema = self.bids_schema
        else:
            from ..utils.bids_integration import BIDSSchemaIntegration

            bids_schema = BIDSSchemaIntegration()

        patterns = {}

        # Generate patterns using ONLY schema-defined rules
        for modality in bids_schema.modalities:
            # Get all valid entity-suffix combinations from schema rules
            combinations = bids_schema.get_all_valid_entity_suffix_combinations(modality)

            for combination in combinations:
                entities_list = combination.get("entities", [])
                suffixes_list = combination.get("suffixes", [])

                for suffix in suffixes_list:
                    # Basic pattern with no entities
                    pattern_key = f"{modality}_{suffix.lower()}"
                    patterns[pattern_key] = bids_schema.generate_bids_path_template(
                        modality=modality, suffix=suffix, entities={}, include_session=True
                    )

                    # Generate patterns for each entity in the schema rule
                    for entity in entities_list:
                        if entity not in [
                            "subject",
                            "session",
                        ]:  # These are handled by path template
                            # Single entity pattern
                            entity_dict = {entity: f"{{{entity}}}"}
                            pattern_key = f"{modality}_{suffix.lower()}_{entity}"
                            patterns[pattern_key] = bids_schema.generate_bids_path_template(
                                modality=modality,
                                suffix=suffix,
                                entities=entity_dict,
                                include_session=True,
                            )

        # Cache the patterns for future use
        self._bids_patterns_cache = patterns

        return patterns

    def get_all_schema_bids_patterns(self) -> dict[str, str]:
        """
        Get ALL possible BIDS patterns directly from schema rules.

        Returns
        -------
        dict[str, str]
            Dictionary with all schema-valid BIDS path patterns
        """
        if (
            hasattr(self, "_all_schema_patterns_cache")
            and self._all_schema_patterns_cache is not None
        ):
            return self._all_schema_patterns_cache

        # Use existing BIDS schema integration if available
        if hasattr(self, "bids_schema") and self.bids_schema is not None:
            bids_schema = self.bids_schema
        else:
            from ..utils.bids_integration import BIDSSchemaIntegration

            bids_schema = BIDSSchemaIntegration()

        patterns = {}

        # Generate patterns using ALL schema rules for each modality
        for modality in bids_schema.modalities:
            # Get the complete set of schema rules for this modality
            schema_rules = bids_schema.get_schema_rules_for_modality(modality)

            for rule in schema_rules:
                entities_list = rule.get("entities", [])
                suffixes_list = rule.get("suffixes", [])

                for suffix in suffixes_list:
                    # Pattern without additional entities
                    pattern_key = f"full_{modality}_{suffix.lower()}"
                    patterns[pattern_key] = bids_schema.generate_bids_path_template(
                        modality=modality, suffix=suffix, entities={}, include_session=True
                    )

                    # Pattern with all entities from this rule
                    for entity in entities_list:
                        if entity not in ["subject", "session"]:
                            entity_dict = {entity: f"{{{entity}}}"}
                            pattern_key = f"full_{modality}_{suffix.lower()}_{entity}"
                            patterns[pattern_key] = bids_schema.generate_bids_path_template(
                                modality=modality,
                                suffix=suffix,
                                entities=entity_dict,
                                include_session=True,
                            )

                    # Pattern with combinations of entities (if multiple entities in rule)
                    if len(entities_list) > 1:
                        # Generate combinations of 2+ entities
                        from itertools import combinations

                        for r in range(
                            2, min(4, len(entities_list) + 1)
                        ):  # Limit to reasonable combinations
                            for entity_combo in combinations(entities_list, r):
                                # Skip if it includes subject/session (handled by template)
                                filtered_combo = [
                                    e for e in entity_combo if e not in ["subject", "session"]
                                ]
                                if len(filtered_combo) >= 2:
                                    entity_dict = {
                                        entity: f"{{{entity}}}" for entity in filtered_combo
                                    }
                                    combo_str = "_".join(sorted(filtered_combo))
                                    pattern_key = f"full_{modality}_{suffix.lower()}_{combo_str}"
                                    patterns[pattern_key] = (
                                        bids_schema.generate_bids_path_template(
                                            modality=modality,
                                            suffix=suffix,
                                            entities=entity_dict,
                                            include_session=True,
                                        )
                                    )

        # Cache for future use
        self._all_schema_patterns_cache = patterns
        return patterns
