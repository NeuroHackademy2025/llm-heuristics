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
        series_info: list[dict[str, Any]],
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """
        Create a prompt for LLM heuristic generation.

        Parameters
        ----------
        series_info : list[dict[str, Any]]
            Information about each DICOM series
        study_info : dict[str, str]
            Study-level information
        additional_context : str | None
            Additional context or requirements

        Returns
        -------
        str
            Formatted prompt for the LLM
        """
        # Enhance series info with BIDS schema classifications
        enhanced_series_info = self._enhance_series_with_bids_schema(series_info)

        return self.prompt_template.render(
            series_info=enhanced_series_info,
            study_info=study_info,
            additional_context=additional_context or "",
            num_series=len(series_info),
            bids_entities=self._get_bids_entities_info() if self.bids_schema else {},
            bids_modalities=self._get_bids_modalities_info() if self.bids_schema else {},
        )

    def generate_heuristic_skeleton(
        self,
        series_mappings: dict[str, str],
        study_info: dict[str, str],
    ) -> str:
        """
        Generate a heuristic file skeleton.

        Parameters
        ----------
        series_mappings : dict[str, str]
            Mapping of series identifiers to BIDS patterns
        study_info : dict[str, str]
            Study information

        Returns
        -------
        str
            Heuristic file skeleton
        """
        return self.base_template.render(
            series_mappings=series_mappings,
            study_info=study_info,
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
        """Load heudiconv's convertall.py template and customize it for our use.

        This method requires heudiconv to be installed and will not provide any fallback.

        Raises
        ------
        ImportError
            If heudiconv is not installed
        RuntimeError
            If template extraction fails
        """
        try:
            import inspect

            import heudiconv.heuristics.convertall as convertall

            # Get the source code of the convertall module
            source = inspect.getsource(convertall)

            # Create our customized template based on the heudiconv source
            template = f'''"""
Heuristic file generated automatically by llm-heuristics

This file is based on heudiconv's convertall.py template and follows
the standard heudiconv format for maximum compatibility.

Based on DICOM analysis from: {{{{ study_info.dicom_dir }}}}
Number of series analyzed: {{{{ study_info.num_series }}}}
"""

{source}

# Override the default infotodict function with our custom logic
def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where

    This function follows the heudiconv standard format and uses
    the same function signature as heudiconv.heuristics.convertall.

    The namedtuple `s` contains the standard SeqInfo fields from heudiconv.
    """

    # Define BIDS key templates for each sequence type
    {{% for key, pattern in series_mappings.items() -%}}
    {{{{ key }}}} = create_key('{{{{ pattern }}}}')
    {{% endfor %}}

    # Initialize info dictionary following heudiconv standard
    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {{
        {{%- for key in series_mappings.keys() %}}
        {{{{ key }}}}: [],
        {{%- endfor %}}
    }}

    # Process each DICOM series using heudiconv's SeqInfo structure
    for s in seqinfo:
        # Map sequences to BIDS based on DICOM metadata
        # Logic generated by LLM analysis of your DICOM data

        {{% for key, pattern in series_mappings.items() -%}}
        # Add sequence identification logic for {{{{ key }}}}
        # Example: if conditions_for_{{{{ key }}}}:
        #     info[{{{{ key }}}}].append(s.series_id)
        {{% endfor %}}

    return info
'''
            return template

        except ImportError as e:
            # heudiconv is required - no fallback
            raise ImportError(f"heudiconv is required but not installed: {e}") from e
        except Exception as e:
            # Source inspection failed - this is also an error since heudiconv is required
            raise RuntimeError(f"Failed to extract template from heudiconv: {e}") from e

    def _get_prompt_template(self) -> str:
        """Get the LLM prompt template."""
        return """You are an expert neuroimaging researcher who helps create heudiconv
heuristic files for converting DICOM data to BIDS format.

IMPORTANT: The template below is based on the actual heudiconv.heuristics.convertall module.
Generate a complete heuristic file by filling in the infotodict function with sequence logic.

I need you to generate a complete, working heuristic file based on the following DICOM analysis:

STUDY INFORMATION:
- DICOM Directory: {{ study_info.dicom_dir }}
- Total Series: {{ num_series }}

DICOM SERIES ANALYSIS:
{% for series in series_info -%}
Series {{ series.series_number }}:
  - Description: {{ series.description }}
  - Protocol: {{ series.protocol }}
  - Modality: {{ series.modality }}
  - Dimensions: {{ series.dimensions }}
  - Files: {{ series.num_files }}
  - Scanning Sequence: {{ series.sequence }}
  - Sequence Variant: {{ series.variant }}
  - TR: {{ series.tr }} ms
  - TE: {{ series.te }} ms
  - Data Type: {{ "Derived" if series.is_derived else "Raw" }}
{% if series.bids_modality -%}
  - BIDS Suggestion: {{ series.bids_modality }}/{{ series.bids_suffix }}
    (confidence: {{ "%.1f"|format(series.bids_confidence) }})
  - Suggested Path: {{ series.suggested_bids_path }}
{% if series.bids_entities -%}
  - Suggested Entities: {{ series.bids_entities }}
{% endif -%}
{% endif -%}

{% endfor %}

{% if bids_modalities -%}
BIDS SCHEMA REFERENCE:
Valid BIDS Modalities:
{% for modality, info in bids_modalities.items() -%}
  - {{ modality }}: {{ info.description }}
{% endfor %}

{% endif -%}
{% if bids_entities -%}
Valid BIDS Entities:
{% for entity, info in bids_entities.items() -%}
  - {{ entity }} ({{ info.entity }}): {{ info.description }}
{% endfor %}

{% endif -%}

{% if additional_context -%}
ADDITIONAL REQUIREMENTS:
{{ additional_context }}
{% endif %}

Please generate a complete heuristic.py file:

1. USE THE TEMPLATE STRUCTURE ABOVE (which comes from heudiconv.heuristics.convertall)
2. Fill in the infotodict function with proper sequence identification logic
3. Use the standard SeqInfo fields available in the template comments
4. Map each DICOM series to appropriate BIDS naming

IMPORTANT HEUDICONV GUIDELINES:
- The template already includes the correct heudiconv imports and functions
- Reference SeqInfo fields correctly (s.protocol_name, s.series_description, etc.)
- Use BIDS-compliant naming conventions
- Be specific in sequence identification to avoid misclassification
- Consider anatomical (anat), functional (func), diffusion (dwi),
  fieldmap (fmap), and perfusion (perf) data types
- For functional data, include task names (e.g., task-rest, task-[taskname])
- Use acquisition labels (acq-) when needed to distinguish similar sequences
- Include run numbers (run-) for repeated sequences

Generate the complete Python heuristic file by customizing the template above:"""

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

    def _enhance_series_with_bids_schema(
        self, series_info: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Enhance series information with BIDS schema classifications.

        Parameters
        ----------
        series_info : list[dict[str, Any]]
            Original series information

        Returns
        -------
        list[dict[str, Any]]
            Enhanced series information with BIDS suggestions
        """
        if not self.bids_schema:
            return series_info

        enhanced_series = []
        for series in series_info:
            enhanced = series.copy()

            try:
                # Get dimensions tuple
                dimensions = series.get("dimensions", "1x1x1x1").split("x")
                dimensions = tuple(int(d) for d in dimensions if d.isdigit())
                if len(dimensions) < 4:
                    dimensions = dimensions + (1,) * (4 - len(dimensions))

                # Classify using BIDS schema
                classification = self.bids_schema.classify_sequence_to_bids(
                    series_description=series.get("description", ""),
                    protocol_name=series.get("protocol", ""),
                    sequence_type=series.get("sequence", ""),
                    dimensions=dimensions,
                )

                # Add BIDS suggestions to series info
                enhanced["bids_modality"] = classification["modality"]
                enhanced["bids_suffix"] = classification["suffix"]
                enhanced["bids_entities"] = classification["entities"]
                enhanced["bids_confidence"] = classification["confidence"]

                # Generate suggested BIDS path
                enhanced["suggested_bids_path"] = self.bids_schema.generate_bids_path_template(
                    modality=classification["modality"],
                    suffix=classification["suffix"],
                    entities=classification["entities"],
                    include_session=True,
                )

            except Exception as e:
                logger.warning(
                    "Failed to classify series %s: %s", series.get("series_number", "unknown"), e
                )
                # Add schema-driven fallback classifications
                fallback = self._get_schema_fallback_classification()
                enhanced.update(fallback)

            enhanced_series.append(enhanced)

        return enhanced_series

    def _get_bids_entities_info(self) -> dict[str, Any]:
        """Get BIDS entities information for template context."""
        if not self.bids_schema:
            return {}

        entities_info = {}
        for entity_key, entity_data in self.bids_schema.entities.items():
            entities_info[entity_key] = {
                "name": entity_data.get("name", entity_key),
                "description": entity_data.get("description", ""),
                "entity": entity_data.get("entity", entity_key),
            }

        return entities_info

    def _get_bids_modalities_info(self) -> dict[str, Any]:
        """Get BIDS modalities information for template context."""
        if not self.bids_schema:
            return {}

        modalities_info = {}
        for modality_key, modality_data in self.bids_schema.modalities.items():
            modalities_info[modality_key] = {
                "name": modality_data.get("name", modality_key),
                "description": modality_data.get("description", ""),
            }

        return modalities_info
