"""Integration with BIDS schema and specification for enhanced heuristic generation."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

from bidsschematools import schema

logger = logging.getLogger(__name__)


class BIDSSchemaIntegration:
    """Integration with official BIDS schema for enhanced heuristic generation."""

    def __init__(self, schema_version: str = "latest") -> None:
        """
        Initialize BIDS schema integration.

        Parameters
        ----------
        schema_version : str
            BIDS schema version to use ('latest' or specific version)
            Can be:
            - 'latest': Use the most recent schema from bidsschematools
            - 'master': Use the development version from bids-schema repo
            - 'x.y.z': Use a specific released version (e.g., '1.8.0')
        """
        self.schema_version = schema_version
        self._schema = None
        self._entities = None
        self._modalities = None
        self._suffixes = None

        # Cache directory for schema files from bids-schema repository
        self.cache_dir = Path.home() / ".cache" / "llm-heuristics" / "bids-schema"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # BIDS schema repository URLs
        self.schema_repo_base = "https://raw.githubusercontent.com/bids-standard/bids-schema/main"

    @property
    def bids_schema(self) -> dict[str, Any]:
        """Get the BIDS schema dictionary."""
        if self._schema is None:
            self._load_schema()
        return self._schema

    @property
    def entities(self) -> dict[str, Any]:
        """Get BIDS entities definitions."""
        if self._entities is None:
            entities_obj = self.bids_schema.get("objects", {}).get("entities", {})
            # Convert Namespace to dict if needed
            if hasattr(entities_obj, "_name") or hasattr(entities_obj, "__dict__"):
                self._entities = (
                    dict(entities_obj) if hasattr(entities_obj, "items") else entities_obj.__dict__
                )
            else:
                self._entities = entities_obj
        return self._entities

    @property
    def modalities(self) -> dict[str, Any]:
        """Get BIDS modalities definitions."""
        if self._modalities is None:
            modalities_obj = self.bids_schema.get("objects", {}).get("modalities", {})
            # Convert Namespace to dict if needed
            if hasattr(modalities_obj, "_name") or hasattr(modalities_obj, "__dict__"):
                self._modalities = (
                    dict(modalities_obj)
                    if hasattr(modalities_obj, "items")
                    else modalities_obj.__dict__
                )
            else:
                self._modalities = modalities_obj
        return self._modalities

    @property
    def suffixes(self) -> dict[str, Any]:
        """Get BIDS suffixes definitions."""
        if self._suffixes is None:
            suffixes_obj = self.bids_schema.get("objects", {}).get("suffixes", {})
            # Convert Namespace to dict if needed
            if hasattr(suffixes_obj, "_name") or hasattr(suffixes_obj, "__dict__"):
                self._suffixes = (
                    dict(suffixes_obj) if hasattr(suffixes_obj, "items") else suffixes_obj.__dict__
                )
            else:
                self._suffixes = suffixes_obj
        return self._suffixes

    def _load_schema(self) -> None:
        """Load BIDS schema from official sources."""
        try:
            logger.info("Loading BIDS schema version: %s", self.schema_version)

            if self.schema_version == "latest":
                # Use bidsschematools for cutting-edge schema
                self._schema = schema.load_schema()
                logger.info("Loaded latest cutting-edge BIDS schema via bidsschematools")

            elif self.schema_version == "master":
                # Load stable version from bids-schema repository
                self._schema = self._load_schema_from_repository("master")
                logger.info("Loaded stable master BIDS schema from repository")

            elif self._is_version_string(self.schema_version):
                # Load specific version from bids-schema repository
                self._schema = self._load_schema_from_repository(f"versions/{self.schema_version}")
                logger.info("Loaded BIDS schema version %s from repository", self.schema_version)

            else:
                # Try bidsschematools as fallback
                logger.warning(
                    "Unknown schema version %s, using bidsschematools default", self.schema_version
                )
                self._schema = schema.load_schema()

            if not self._schema:
                raise RuntimeError("Failed to load any BIDS schema")

            logger.info("BIDS schema loaded successfully")

        except Exception as e:
            logger.error("Failed to load BIDS schema: %s", e)
            raise RuntimeError(f"BIDS schema is required but failed to load: {e}") from e

    def _is_version_string(self, version: str) -> bool:
        """Check if string looks like a version number (e.g., '1.8.0')."""
        import re

        return bool(re.match(r"^\d+\.\d+\.\d+", version))

    def _load_schema_from_repository(self, schema_path: str) -> dict[str, Any]:
        """
        Load BIDS schema from the official bids-schema repository.

        Parameters
        ----------
        schema_path : str
            Path within the repository (e.g., 'master', 'versions/1.8.0')

        Returns
        -------
        dict[str, Any]
            Loaded BIDS schema
        """
        # Check cache first
        cache_file = self.cache_dir / f"{schema_path.replace('/', '_')}_schema.json"

        if cache_file.exists():
            try:
                with cache_file.open("r") as f:
                    cached_schema = json.load(f)
                logger.info("Loaded BIDS schema from cache: %s", cache_file)
                return cached_schema
            except Exception as e:
                logger.warning("Failed to load cached schema, will download: %s", e)

        # Download from repository
        try:
            # The bids-schema repository structure from the GitHub page
            if schema_path == "master":
                # For master branch, we need to construct the proper URL
                # Based on repository structure, schema files are in the root or tools/
                schema_url = f"{self.schema_repo_base}/src/schema.json"
            else:
                # For versioned schemas: versions/x.y.z/schema.json
                schema_url = f"{self.schema_repo_base}/{schema_path}/schema.json"

            logger.info("Downloading BIDS schema from: %s", schema_url)

            with urllib.request.urlopen(schema_url) as response:
                schema_data = json.loads(response.read().decode("utf-8"))

            # Cache the downloaded schema
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w") as f:
                json.dump(schema_data, f, indent=2)

            logger.info("Downloaded and cached BIDS schema: %s", cache_file)
            return schema_data

        except Exception as e:
            logger.error("Failed to download BIDS schema from %s: %s", schema_url, e)
            # Try alternative approach using bidsschematools as backup
            try:
                logger.info("Falling back to bidsschematools")
                return schema.load_schema()
            except Exception as backup_e:
                raise RuntimeError(
                    f"Failed to load schema from repository and bidsschematools: {e}, {backup_e}"
                ) from e

    def get_valid_entities_for_modality(self, modality: str) -> list[str]:
        """
        Get valid BIDS entities for a specific modality.

        Parameters
        ----------
        modality : str
            BIDS modality (e.g., 'anat', 'func', 'dwi', 'fmap', 'perf')

        Returns
        -------
        list[str]
            List of valid entity names for the modality
        """
        try:
            # Get valid entities for modality from schema
            # Get entities directly from the raw file rules for this modality
            raw_files = self.bids_schema["rules"]["files"]["raw"]

            if modality not in raw_files:
                available_modalities = list(raw_files.keys())
                raise RuntimeError(
                    f"Modality '{modality}' not found in BIDS schema. "
                    f"Available modalities: {available_modalities}"
                )

            valid_entities = []
            modality_rules = raw_files[modality]

            # Extract entities from all rule categories within this modality
            for rule_name, rule in modality_rules.items():
                if isinstance(rule, dict) and "entities" in rule:
                    entities = rule["entities"]
                    if isinstance(entities, dict):
                        valid_entities.extend(entities.keys())
                    else:
                        logger.warning(
                            "Unexpected entities format for %s/%s: %s",
                            modality,
                            rule_name,
                            type(entities),
                        )

            # Remove duplicates while preserving order
            unique_entities = list(dict.fromkeys(valid_entities))

            logger.debug(
                "Found %d entities for modality '%s': %s",
                len(unique_entities),
                modality,
                unique_entities,
            )

            return unique_entities

        except Exception as e:
            logger.error("Failed to get entities for modality %s: %s", modality, e)
            raise RuntimeError(
                f"Cannot extract BIDS entities for modality '{modality}': {e}"
            ) from e

    def get_schema_rules_for_modality(self, modality: str) -> list[dict[str, Any]]:
        """
        Get all schema rules that apply to a specific modality.

        Parameters
        ----------
        modality : str
            BIDS modality

        Returns
        -------
        list[dict[str, Any]]
            List of schema rules for the modality
        """
        try:
            rules = []

            # Get rules from schema
            all_rules = self.bids_schema.get("rules", {}).get("files", {}).get("common", [])

            for rule in all_rules:
                if modality in rule.get("datatypes", []):
                    rules.append(
                        {
                            "entities": rule.get("entities", []),
                            "suffixes": rule.get("suffixes", []),
                            "extensions": rule.get("extensions", []),
                            "datatypes": rule.get("datatypes", []),
                        }
                    )

            return rules

        except Exception as e:
            logger.warning("Failed to get schema rules for modality %s: %s", modality, e)
            return []

    def get_all_valid_entity_suffix_combinations(self, modality: str) -> list[dict[str, Any]]:
        """
        Get all valid entity-suffix combinations for a modality from schema rules.

        Parameters
        ----------
        modality : str
            BIDS modality

        Returns
        -------
        list[dict[str, Any]]
            List of valid combinations with entities and suffixes
        """
        try:
            combinations = []
            rules = self.get_schema_rules_for_modality(modality)

            for rule in rules:
                entities = rule.get("entities", [])
                suffixes = rule.get("suffixes", [])

                # Each rule defines a valid entity-suffix combination
                combinations.append({"entities": entities, "suffixes": suffixes, "rule": rule})

            return combinations

        except Exception as e:
            logger.warning("Failed to get entity-suffix combinations for %s: %s", modality, e)
            return []

    def get_valid_suffixes_for_modality(self, modality: str) -> list[str]:
        """
        Get valid BIDS suffixes for a specific modality from the actual BIDS schema.

        Parameters
        ----------
        modality : str
            BIDS modality (e.g., 'anat', 'func', 'dwi', 'fmap', 'perf')

        Returns
        -------
        list[str]
            List of valid suffixes for the modality

        Raises
        ------
        RuntimeError
            If BIDS schema is not properly loaded or modality not found
        """
        if self._schema is None:
            raise RuntimeError("BIDS schema not loaded. Cannot extract suffixes.")

        try:
            # For the real schema, all datatypes (anat, func, etc.) are under files/raw
            raw_files = self.bids_schema.get("rules", {}).get("files", {}).get("raw", {})

            # Handle "mri" by reading its constituent datatypes from BIDS schema
            if modality == "mri":
                # Get MRI datatypes from BIDS schema rules
                try:
                    mri_datatypes = (
                        self.bids_schema.get("rules", {})
                        .get("modalities", {})
                        .get("mri", {})
                        .get("datatypes", [])
                    )
                    if not isinstance(mri_datatypes, list):
                        # Fallback to common MRI modalities
                        mri_datatypes = ["anat", "func", "dwi", "fmap", "perf"]

                    logger.debug("MRI datatypes from BIDS schema: %s", mri_datatypes)

                except Exception as e:
                    logger.warning(
                        "Cannot extract MRI datatypes from BIDS schema: %s, using fallback", e
                    )
                    mri_datatypes = ["anat", "func", "dwi", "fmap", "perf"]

                valid_suffixes = []

                for datatype in mri_datatypes:
                    datatype_suffixes = self.get_valid_suffixes_for_modality(datatype)
                    valid_suffixes.extend(datatype_suffixes)

                # Remove duplicates while preserving order
                unique_suffixes = list(dict.fromkeys(valid_suffixes))

                logger.debug(
                    "Found %d combined MRI suffixes from %d datatypes: %s",
                    len(unique_suffixes),
                    len(mri_datatypes),
                    unique_suffixes,
                )

                return unique_suffixes

            # Check if modality exists in raw files
            if modality not in raw_files:
                available_modalities = list(raw_files.keys())
                logger.warning(
                    "Modality '%s' not found in BIDS schema. Available: %s",
                    modality,
                    available_modalities,
                )
                # Return empty list instead of raising error for better compatibility
                return []

            modality_rules = raw_files[modality]
            valid_suffixes = []

            # Handle different ways the schema might store suffixes
            if hasattr(modality_rules, "items"):
                # If it's a dict-like object
                items = modality_rules.items()
            elif hasattr(modality_rules, "__dict__"):
                # If it's a Namespace-like object
                items = modality_rules.__dict__.items()
            else:
                # Try to iterate directly
                try:
                    items = [
                        (k, getattr(modality_rules, k))
                        for k in dir(modality_rules)
                        if not k.startswith("_")
                    ]
                except Exception as e:
                    logger.warning("Cannot iterate over modality rules for %s: %s", modality, e)
                    return []

            # Extract suffixes from all rule categories within this modality
            for _rule_name, rule in items:
                if hasattr(rule, "suffixes"):
                    # Handle Namespace-like suffixes
                    suffixes = getattr(rule, "suffixes", [])
                    if isinstance(suffixes, list):
                        valid_suffixes.extend(suffixes)
                    elif hasattr(suffixes, "values"):
                        # If suffixes is a dict-like object
                        valid_suffixes.extend(suffixes.values())
                elif isinstance(rule, dict) and "suffixes" in rule:
                    # Handle dict-like suffixes
                    suffixes = rule["suffixes"]
                    if isinstance(suffixes, list):
                        valid_suffixes.extend(suffixes)

            # Remove duplicates while preserving order
            unique_suffixes = list(dict.fromkeys(valid_suffixes))

            logger.debug(
                "Found %d suffixes for modality '%s': %s",
                len(unique_suffixes),
                modality,
                unique_suffixes,
            )
            return unique_suffixes

        except Exception as e:
            logger.warning("Failed to extract suffixes for modality %s: %s", modality, e)
            # Return empty list instead of raising error for better compatibility
            return []

    def classify_sequence_to_bids(
        self,
        series_description: str,
        protocol_name: str,
        sequence_type: str,
        dimensions: tuple,
    ) -> dict[str, str]:
        """
        Classify a sequence to BIDS modality and suffix using schema knowledge.

        Parameters
        ----------
        series_description : str
            DICOM series description
        protocol_name : str
            DICOM protocol name
        sequence_type : str
            MR sequence type
        dimensions : tuple
            Image dimensions (rows, cols, slices, timepoints)

        Returns
        -------
        dict[str, str]
            Dictionary with suggested 'modality', 'suffix', and 'entities'
        """
        # Combine text fields for analysis
        text_to_analyze = f"{series_description} {protocol_name} {sequence_type}".lower()

        # Get schema-driven defaults instead of hardcoding
        default_modality, default_suffix = self._get_schema_defaults()

        # Classification logic based on BIDS schema
        classification = {
            "modality": default_modality,
            "suffix": default_suffix,
            "entities": {},
            "confidence": 0.0,
        }

        # Anatomical sequences
        if any(keyword in text_to_analyze for keyword in ["t1", "mprage", "anatomy"]):
            suggested_suffix = "T1w"
            if self._validate_suffix_for_modality("anat", suggested_suffix):
                classification.update(
                    {"modality": "anat", "suffix": suggested_suffix, "confidence": 0.9}
                )
        elif any(keyword in text_to_analyze for keyword in ["t2", "space", "cube"]):
            if "flair" in text_to_analyze:
                suggested_suffix = "FLAIR"
                if self._validate_suffix_for_modality("anat", suggested_suffix):
                    classification.update(
                        {"modality": "anat", "suffix": suggested_suffix, "confidence": 0.9}
                    )
            else:
                suggested_suffix = "T2w"
                if self._validate_suffix_for_modality("anat", suggested_suffix):
                    classification.update(
                        {"modality": "anat", "suffix": suggested_suffix, "confidence": 0.9}
                    )

        # Functional sequences
        elif any(
            keyword in text_to_analyze for keyword in ["bold", "fmri", "func", "rest", "task"]
        ):
            suggested_suffix = "bold"
            if self._validate_suffix_for_modality("func", suggested_suffix):
                classification.update(
                    {"modality": "func", "suffix": suggested_suffix, "confidence": 0.9}
                )

            # Determine task
            if "rest" in text_to_analyze:
                classification["entities"]["task"] = "rest"
            elif "task" in text_to_analyze:
                # Try to extract task name
                words = text_to_analyze.split()
                for i, word in enumerate(words):
                    if "task" in word and i + 1 < len(words):
                        classification["entities"]["task"] = words[i + 1]
                        break
                else:
                    classification["entities"]["task"] = "unknown"

        # Diffusion sequences
        elif any(keyword in text_to_analyze for keyword in ["dwi", "dti", "diffusion", "tensor"]):
            suggested_suffix = "dwi"
            if self._validate_suffix_for_modality("dwi", suggested_suffix):
                classification.update(
                    {"modality": "dwi", "suffix": suggested_suffix, "confidence": 0.9}
                )

            # Check for direction encoding
            if any(keyword in text_to_analyze for keyword in ["ap", "anterior", "posterior"]):
                classification["entities"]["direction"] = "AP"
            elif any(keyword in text_to_analyze for keyword in ["pa", "posterior", "anterior"]):
                classification["entities"]["direction"] = "PA"

        # Fieldmap sequences
        elif any(keyword in text_to_analyze for keyword in ["fieldmap", "field", "b0", "shim"]):
            classification.update({"modality": "fmap", "confidence": 0.8})

            if "magnitude" in text_to_analyze:
                suggested_suffix = "magnitude"
                if self._validate_suffix_for_modality("fmap", suggested_suffix):
                    classification["suffix"] = suggested_suffix
            elif "phase" in text_to_analyze:
                suggested_suffix = "phasediff"
                if self._validate_suffix_for_modality("fmap", suggested_suffix):
                    classification["suffix"] = suggested_suffix
            else:
                suggested_suffix = "fieldmap"
                if self._validate_suffix_for_modality("fmap", suggested_suffix):
                    classification["suffix"] = suggested_suffix

        # ASL/Perfusion sequences
        elif any(keyword in text_to_analyze for keyword in ["asl", "perfusion", "pcasl", "pasl"]):
            suggested_suffix = "asl"
            if self._validate_suffix_for_modality("perf", suggested_suffix):
                classification.update(
                    {"modality": "perf", "suffix": suggested_suffix, "confidence": 0.9}
                )

        # Use dimensions for additional hints
        rows, cols, slices, timepoints = dimensions

        # Multi-timepoint suggests functional
        if timepoints > 1 and classification["modality"] == "anat":
            classification.update(
                {
                    "modality": "func",
                    "suffix": "bold",
                    "confidence": max(0.6, classification["confidence"]),
                }
            )

        # High resolution suggests anatomical
        if (rows >= 512 or cols >= 512) and classification["modality"] == "func":
            classification["confidence"] = max(0.3, classification["confidence"] - 0.2)

        return classification

    def _validate_suffix_for_modality(self, modality: str, suffix: str) -> bool:
        """
        Validate that a suffix is valid for a given modality according to BIDS schema.

        Parameters
        ----------
        modality : str
            BIDS modality
        suffix : str
            Proposed suffix

        Returns
        -------
        bool
            True if suffix is valid for modality
        """
        try:
            valid_suffixes = self.get_valid_suffixes_for_modality(modality)
            return suffix in valid_suffixes
        except Exception:
            logger.warning("Could not validate suffix '%s' for modality '%s'", suffix, modality)
            return False

    def generate_bids_path_template(
        self, modality: str, suffix: str, entities: dict[str, str], include_session: bool = True
    ) -> str:
        """
        Generate a BIDS-compliant path template.

        Parameters
        ----------
        modality : str
            BIDS modality
        suffix : str
            BIDS suffix
        entities : dict[str, str]
            Entity key-value pairs
        include_session : bool
            Whether to include session in the path

        Returns
        -------
        str
            BIDS path template
        """
        # Build entity string following BIDS entity order
        entity_order = self.get_entity_order()

        entity_parts = []

        # Always include subject
        entity_parts.append("sub-{subject}")

        # Include session if requested and available
        if include_session:
            entity_parts.append("{session}")

        # Add other entities in order
        for entity in entity_order[2:]:  # Skip subject and session
            if entity in entities:
                if entity == "task":
                    entity_parts.append(f"task-{entities[entity]}")
                elif entity == "run":
                    entity_parts.append("run-{item:02d}")
                else:
                    entity_parts.append(f"{entity}-{entities[entity]}")

        # Create the full path template
        if include_session:
            path_template = (
                f"sub-{{subject}}/{{session}}/{modality}/{'_'.join(entity_parts)}_{suffix}"
            )
        else:
            path_template = f"sub-{{subject}}/{modality}/{'_'.join(entity_parts)}_{suffix}"

        return path_template

    def validate_bids_path(self, path: str) -> dict[str, Any]:
        """
        Validate a BIDS path against the schema.

        Parameters
        ----------
        path : str
            BIDS path to validate

        Returns
        -------
        dict[str, Any]
            Validation results with 'valid', 'errors', 'warnings'
        """
        result = {"valid": True, "errors": [], "warnings": [], "entities": {}}

        try:
            # Basic path structure validation
            parts = Path(path).parts

            # Should have at least subject/modality/file
            if len(parts) < 3:
                result["errors"].append("Path too short - missing required components")
                result["valid"] = False
                return result

            # Check subject directory
            if not parts[0].startswith("sub-"):
                result["errors"].append("First directory must be subject (sub-*)")
                result["valid"] = False

            # Extract modality
            modality_idx = -2  # Second to last part
            if modality_idx >= len(parts):
                result["errors"].append("Cannot identify modality directory")
                result["valid"] = False
                return result

            modality = parts[modality_idx]

            # Validate modality against schema
            valid_modalities = list(self.modalities.keys())
            if modality not in valid_modalities:
                result["warnings"].append(
                    f"Modality '{modality}' not in standard list: {valid_modalities}"
                )

            # Parse filename
            filename = parts[-1]
            if not filename.endswith((".nii.gz", ".nii", ".json", ".tsv", ".bval", ".bvec")):
                result["warnings"].append(f"Unusual file extension in '{filename}'")

            # Extract entities and suffix from filename
            basename = filename.split(".")[0]  # Remove extensions
            filename_parts = basename.split("_")

            suffix = filename_parts[-1]
            entity_parts = filename_parts[:-1]

            # Validate suffix
            valid_suffixes = self.get_valid_suffixes_for_modality(modality)
            if suffix not in valid_suffixes:
                result["warnings"].append(
                    f"Suffix '{suffix}' not standard for modality '{modality}'. "
                    f"Valid: {valid_suffixes}"
                )

            # Parse entities
            for part in entity_parts:
                if "-" in part:
                    entity, value = part.split("-", 1)
                    result["entities"][entity] = value

                    # Validate entity
                    if entity not in self.entities:
                        result["warnings"].append(f"Entity '{entity}' not in BIDS schema")

        except Exception as e:
            result["errors"].append(f"Validation error: {e}")
            result["valid"] = False

        return result

    def _get_schema_defaults(self) -> tuple[str, str]:
        """
        Get schema-driven default modality and suffix.

        Returns
        -------
        tuple[str, str]
            Default modality and suffix from schema
        """
        try:
            # Get first available modality and its first available suffix
            available_modalities = list(self.modalities.keys())
            if available_modalities:
                default_modality = available_modalities[0]  # Usually 'anat'
                available_suffixes = self.get_valid_suffixes_for_modality(default_modality)
                if not available_suffixes:
                    raise RuntimeError(
                        f"BIDS schema has no valid suffixes for modality '{default_modality}'"
                    )
                default_suffix = available_suffixes[0]  # First valid suffix for modality
            else:
                # Schema has no modalities - this is a schema error
                raise RuntimeError("BIDS schema contains no modalities")

            return default_modality, default_suffix

        except Exception as e:
            logger.error("Failed to get schema defaults: %s", e)
            # Re-raise since schema is required
            raise RuntimeError(f"Failed to get BIDS schema defaults: {e}") from e

    def get_schema_version_info(self) -> dict[str, str]:
        """Get information about the loaded BIDS schema version."""
        return {
            "schema_version": self.schema_version,
            "num_entities": len(self.entities),
            "num_modalities": len(self.modalities),
            "num_suffixes": len(self.suffixes),
            "schema_source": self._get_schema_source(),
        }

    def _get_schema_source(self) -> str:
        """Get the source of the currently loaded schema."""
        if self.schema_version == "latest":
            return "bidsschematools (latest cutting-edge)"
        elif self.schema_version == "master":
            return "bids-schema repository (stable master branch)"
        elif self._is_version_string(self.schema_version):
            return f"bids-schema repository (version {self.schema_version})"
        else:
            return "bidsschematools (fallback)"

    def get_entity_order(self) -> list[str]:
        """
        Get the standard BIDS entity order from schema.

        Returns
        -------
        list[str]
            List of entity names in standard BIDS order
        """
        try:
            # Try to get entity order from schema
            if hasattr(self.bids_schema, "get") and isinstance(self.bids_schema, dict):
                # Look for entity order in schema rules or objects
                entity_order = self.bids_schema.get("rules", {}).get("entities", [])
                if entity_order:
                    return entity_order

                # Alternative: extract from entity definitions
                entities = self.entities
                if entities:
                    # Try to get order from entity definitions
                    ordered_entities = []
                    for entity_name, entity_info in entities.items():
                        if isinstance(entity_info, dict) and "order" in entity_info:
                            ordered_entities.append((entity_name, entity_info["order"]))

                    if ordered_entities:
                        # Sort by order and return names
                        ordered_entities.sort(key=lambda x: x[1])
                        return [entity[0] for entity in ordered_entities]

            # Fallback to standard BIDS entity order
            logger.warning("Could not extract entity order from schema, using fallback")
            return [
                "subject",
                "session",
                "task",
                "acquisition",
                "run",
                "echo",
                "direction",
                "reconstruction",
                "space",
                "description",
            ]

        except Exception as e:
            logger.warning("Error getting entity order from schema: %s, using fallback", e)
            return [
                "subject",
                "session",
                "task",
                "acquisition",
                "run",
                "echo",
                "direction",
                "reconstruction",
                "space",
                "description",
            ]
