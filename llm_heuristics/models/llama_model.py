"""Llama model implementation for heuristic generation."""

from __future__ import annotations

import logging

import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

from llm_heuristics.models.llm_interface import LLMInterface
from llm_heuristics.utils.templates import HeuristicTemplate

logger = logging.getLogger(__name__)


class LlamaModel(LLMInterface):
    """Llama 3.1 model for generating heuristic files.

    Privacy Note: All processing happens locally on your machine.
    No data is sent to external services or shared with third parties.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        use_quantization: bool = True,
        max_length: int = 8192,
        temperature: float = 0.7,
        device: str | None = None,
    ) -> None:
        """
        Initialize the Llama model.

        Parameters
        ----------
        model_name : str
            HuggingFace model name
        use_quantization : bool
            Whether to use 4-bit quantization
        max_length : int
            Maximum generation length
        temperature : float
            Sampling temperature
        device : str | None
            Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing Llama model: %s", model_name)
        logger.info("Device: %s", self.device)
        logger.info("Quantization: %s", use_quantization)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if requested
        model_kwargs = {}
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        if not use_quantization:
            self.model = self.model.to(self.device)

        self.model.eval()

        # Initialize template helper
        self.template = HeuristicTemplate()

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
            Additional study information
        additional_context : str | None
            Additional context or requirements

        Returns
        -------
        str
            Generated heuristic file content
        """
        logger.info("Generating heuristic file with Llama model")

        # Try to generate using template-based approach first
        try:
            logger.info("Attempting template-based heuristic generation")
            heuristic_code = self._generate_template_based_heuristic(
                dicom_summary, study_info, additional_context
            )
            logger.info("Template-based generation successful")
            return heuristic_code
        except Exception as e:
            logger.warning("Template-based generation failed: %s", e)
            logger.info("Falling back to LLM-based generation")

        # Fallback to LLM generation
        try:
            # Prepare context for the LLM
            context = self._prepare_heuristic_context(
                dicom_summary, study_info, additional_context
            )

            # Generate heuristic
            heuristic_code = self._generate_with_llm(context)

            # Post-process and validate
            heuristic_code = self._post_process_heuristic(heuristic_code)

            return heuristic_code
        except Exception as e:
            logger.error("Both template-based and LLM-based generation failed: %s", e)
            # If both approaches fail, generate a basic template-based version anyway
            return self._generate_basic_heuristic_fallback(
                dicom_summary, study_info, additional_context
            )

    def analyze_sequence_mapping(
        self,
        series_descriptions: list[str],
        protocol_names: list[str],
    ) -> dict[str, str]:
        """
        Analyze sequence descriptions and suggest BIDS mappings.

        Parameters
        ----------
        series_descriptions : List[str]
            List of series descriptions
        protocol_names : List[str]
            List of protocol names

        Returns
        -------
        Dict[str, str]
            Mapping of series to suggested BIDS names
        """
        logger.info("Analyzing sequence mappings")

        # Create prompt for sequence analysis
        prompt = self._create_sequence_mapping_prompt(series_descriptions, protocol_names)

        # Generate mappings
        response = self._generate_with_llm(prompt)

        # Parse response into dictionary
        mappings = self._parse_sequence_mappings(response)

        return mappings

    def explain_heuristic_rules(self, heuristic_content: str) -> str:
        """
        Explain the rules in a generated heuristic file.

        Parameters
        ----------
        heuristic_content : str
            Content of the heuristic file

        Returns
        -------
        str
            Human-readable explanation
        """
        prompt = f"""
Please explain the heuristic rules in this heudiconv heuristic file in simple terms.
Focus on:
1. What each sequence/series maps to in BIDS format
2. The criteria used to identify each sequence
3. Any special handling or considerations

Heuristic file:
```python
{heuristic_content}
```

Provide a clear, numbered explanation that a researcher could easily understand.
"""

        return self._generate_with_llm(prompt)

    def _prepare_heuristic_context(
        self,
        dicom_summary: pd.DataFrame,
        study_info: dict[str, str],
        additional_context: str | None,
    ) -> str:
        """Prepare context for heuristic generation."""

        # Create series information summary from mapped data using ALL groupby variables
        sequences_info = []
        for _, row in dicom_summary.iterrows():
            # Extract ALL variables used by SequencesGrouper
            bids_path = row.get("bids_path", "Unknown")

            # Clean up missing/NaN values
            def clean_value(val, default="Unknown"):
                if pd.isna(val) or val == "Unknown":
                    return default
                return str(val)

            def clean_numeric(val, default=0):
                if pd.isna(val):
                    return default
                return val

            def clean_bool(val, default=False):
                if pd.isna(val):
                    return default
                return bool(val)

            sequences_info.append(
                {
                    # BIDS mapping info
                    "bids_path": bids_path,
                    # Groupby variables from SequencesGrouper
                    "protocol_name": clean_value(row.get("protocol_name")),
                    "series_description": clean_value(row.get("series_description")),
                    "sequence_name": clean_value(row.get("sequence_name")),
                    "dim1": clean_numeric(row.get("dim1", 0)),
                    "dim2": clean_numeric(row.get("dim2", 0)),
                    "dim3": clean_numeric(row.get("dim3", 0)),
                    "dim4": clean_numeric(row.get("dim4", 1)),
                    "TR": clean_numeric(row.get("TR", 0)),
                    "TE": clean_numeric(row.get("TE", 0)),
                    "is_derived": clean_bool(row.get("is_derived", False)),
                    "is_motion_corrected": clean_bool(row.get("is_motion_corrected", False)),
                    "image_type": clean_value(row.get("image_type")),
                    # Aggregated variables from grouping
                    "series_count": clean_numeric(row.get("series_count", 1)),
                    "total_files": clean_numeric(row.get("total_files", 0)),
                    "representative_series_id": clean_value(
                        row.get("representative_series_id"), ""
                    ),
                    "representative_files": clean_numeric(row.get("representative_files", 0)),
                    "example_dcm_file": clean_value(row.get("example_dcm_file"), ""),
                    "dcm_dir_name": clean_value(row.get("dcm_dir_name"), ""),
                    "subject": clean_value(row.get("subject"), ""),
                    "session": clean_value(row.get("session"), ""),
                }
            )

        context = self.template.create_heuristic_prompt(
            sequences_info=sequences_info,
            study_info=study_info,
            additional_context=additional_context,
        )

        return context

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using the LLM."""

        # Log the prompt being sent to LLM
        logger.info("=== LLM PROMPT FOR HEURISTIC GENERATION ===")
        logger.info(prompt)
        logger.info("=== END PROMPT ===")

        # Prepare input
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in neuroimaging, DICOM format, and BIDS conversion. "
                    "You help researchers create heudiconv heuristic files."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Apply chat template if available, otherwise use simple format
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat templates
            input_text = messages[1]["content"]  # Use the user message directly

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - 1000,  # Leave room for generation
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        try:
            input_length = inputs["input_ids"].shape[1]
            if len(outputs[0]) <= input_length:
                logger.warning("Model generated no new tokens, returning empty response")
                return ""

            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )
            return response.strip()
        except (IndexError, RuntimeError) as e:
            logger.error("Error decoding model output: %s", e)
            logger.debug(
                "Outputs shape: %s, Input length: %d",
                getattr(outputs[0], "shape", "unknown"),
                inputs["input_ids"].shape[1] if "input_ids" in inputs else -1,
            )
            return ""

    def complete(self, prompt: str) -> str:
        """Return a plain text completion for an arbitrary prompt."""
        return self._generate_with_llm(prompt)

    def _post_process_heuristic(self, heuristic_code: str) -> str:
        """Post-process generated heuristic code."""

        # Extract Python code from markdown if present
        if "```python" in heuristic_code:
            start = heuristic_code.find("```python") + 9
            end = heuristic_code.find("```", start)
            if end != -1:
                heuristic_code = heuristic_code[start:end].strip()
        elif "```" in heuristic_code:
            start = heuristic_code.find("```") + 3
            end = heuristic_code.find("```", start)
            if end != -1:
                heuristic_code = heuristic_code[start:end].strip()

        # Check if we have a valid heuristic structure
        has_imports = "from heudiconv.utils import SeqInfo" in heuristic_code
        has_create_key = "def create_key(" in heuristic_code
        has_infotodict = "def infotodict(" in heuristic_code

        # If the LLM failed to generate proper code, log and raise an error
        if not (has_imports and has_create_key and has_infotodict):
            logger.error("LLM generated invalid heuristic code. Missing required components:")
            logger.error("- Has imports: %s", has_imports)
            logger.error("- Has create_key: %s", has_create_key)
            logger.error("- Has infotodict: %s", has_infotodict)
            logger.error("Generated content: %s", heuristic_code[:500])

            raise ValueError(
                "LLM failed to generate valid heuristic code. "
                "Please check the logs and try again with different parameters."
            )

        return heuristic_code

    def _create_sequence_mapping_prompt(
        self,
        series_descriptions: list[str],
        protocol_names: list[str],
    ) -> str:
        """Create prompt for sequence mapping analysis."""

        sequences = []
        for i, (desc, proto) in enumerate(zip(series_descriptions, protocol_names)):
            sequences.append(f"{i + 1}. Description: '{desc}', Protocol: '{proto}'")

        prompt = f"""
Analyze these MRI sequences and suggest appropriate BIDS naming conventions:

{chr(10).join(sequences)}

For each sequence, suggest:
1. The appropriate BIDS modality (anat, func, dwi, fmap, etc.)
2. A specific BIDS filename pattern
3. Any relevant BIDS entities (task, acq, run, dir, etc.)

Respond in this format:
1. [modality]/[filename_pattern] - [explanation]

Example:
1. anat/sub-{{subject}}_T1w - T1-weighted anatomical image
"""

        return prompt

    def _parse_sequence_mappings(self, response: str) -> dict[str, str]:
        """Parse LLM response into sequence mappings."""

        mappings = {}
        lines = response.strip().split("\n")

        for line in lines:
            if line.strip() and line.strip()[0].isdigit():
                try:
                    # Extract the BIDS pattern
                    parts = line.split(" - ", 1)
                    if len(parts) >= 1:
                        # Extract just the filename pattern
                        pattern_part = parts[0].split(". ", 1)[1]
                        if "/" in pattern_part:
                            pattern = pattern_part.split("/")[1]
                        else:
                            pattern = pattern_part

                        # Use sequence number as key for now
                        seq_num = line.strip().split(".")[0]
                        mappings[f"sequence_{seq_num}"] = pattern.strip()
                except (IndexError, ValueError):
                    continue

        return mappings

    def _generate_template_based_heuristic(
        self,
        dicom_summary: pd.DataFrame,
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """Generate heuristic using template-based approach without LLM."""

        # Extract mapping information from the mapped data using ALL groupby variables
        sequences_info = []

        for idx, row in dicom_summary.iterrows():
            bids_path = row.get("bids_path", "Unknown")

            if bids_path == "Unknown" or pd.isna(bids_path):
                continue

            # Helper functions for cleaning data
            def clean_value(val, default=""):
                if pd.isna(val) or val == "Unknown":
                    return default
                return str(val)

            def clean_numeric(val, default=0):
                if pd.isna(val):
                    return default
                return val

            def clean_bool(val, default=False):
                if pd.isna(val):
                    return default
                return bool(val)

            image_type = clean_value(row.get("image_type"))

            # Check if we need to create multiple BIDS keys for image_type variants
            # e.g., T2w with NORM vs non-NORM image types
            bids_variants = self._create_image_type_variants(
                bids_path, image_type, additional_context
            )

            for _variant_idx, (variant_bids_path, variant_suffix) in enumerate(bids_variants):
                sequences_info.append(
                    {
                        "key_name": f"key_{idx + 1}"
                        + (f"_{variant_suffix}" if variant_suffix else ""),
                        "bids_path": variant_bids_path,
                        # ALL groupby variables from SequencesGrouper
                        "protocol_name": clean_value(row.get("protocol_name")),
                        "series_description": clean_value(row.get("series_description")),
                        "sequence_name": clean_value(row.get("sequence_name")),
                        "dim1": clean_numeric(row.get("dim1", 0)),
                        "dim2": clean_numeric(row.get("dim2", 0)),
                        "dim3": clean_numeric(row.get("dim3", 0)),
                        "dim4": clean_numeric(row.get("dim4", 1)),
                        "TR": clean_numeric(row.get("TR", 0)),
                        "TE": clean_numeric(row.get("TE", 0)),
                        "is_derived": clean_bool(row.get("is_derived", False)),
                        "is_motion_corrected": clean_bool(row.get("is_motion_corrected", False)),
                        "image_type": image_type,
                        "image_type_filter": variant_suffix,  # Used for filtering logic
                        # Aggregated variables
                        "series_count": clean_numeric(row.get("series_count", 1)),
                        "total_files": clean_numeric(row.get("total_files", 0)),
                        "representative_series_id": clean_value(
                            row.get("representative_series_id")
                        ),
                    }
                )

        # Generate the heuristic using the convertall template
        heuristic_content = self._build_deterministic_heuristic(
            sequences_info, study_info, additional_context
        )

        return heuristic_content

    def _generate_basic_heuristic_fallback(
        self,
        dicom_summary: pd.DataFrame,
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """Generate a basic fallback heuristic when all else fails."""

        logger.warning("Generating basic fallback heuristic")

        # Create a simple convertall-style heuristic
        fallback_content = '''from __future__ import annotations

import logging
from typing import Optional

from heudiconv.utils import SeqInfo

lgr = logging.getLogger("heudiconv")


def create_key(
    template: Optional[str],
    outtype: tuple[str, ...] = ("nii.gz",),
    annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where

    This is a fallback heuristic generated when LLM generation failed.
    You may need to customize this based on your specific data.
    """

    # Default key for all series
    data = create_key("run{item:03d}")
    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {data: []}

    for s in seqinfo:
        info[data].append(s.series_id)

    return info
'''

        return fallback_content

    def _build_deterministic_heuristic(
        self,
        sequences_info: list[dict],
        study_info: dict[str, str],
        additional_context: str | None = None,
    ) -> str:
        """Build heuristic deterministically from mapped data."""

        # Build the heuristic file content step by step
        heuristic_lines = []

        # Add standard imports
        heuristic_lines.extend(
            [
                "from __future__ import annotations",
                "",
                "import logging",
                "from typing import Optional",
                "",
                "from heudiconv.utils import SeqInfo",
                "",
                'lgr = logging.getLogger("heudiconv")',
                "",
                "",
                "def create_key(",
                "    template: Optional[str],",
                '    outtype: tuple[str, ...] = ("nii.gz",),',
                "    annotation_classes: None = None,",
                ") -> tuple[str, tuple[str, ...], None]:",
                "    if template is None or not template:",
                '        raise ValueError("Template must be a valid format string")',
                "    return (template, outtype, annotation_classes)",
                "",
                "",
                "def infotodict(",
                "    seqinfo: list[SeqInfo],",
                ") -> dict[tuple[str, tuple[str, ...], None], list[str]]:",
                '    """Heuristic evaluator for determining which runs belong where"""',
                "",
            ]
        )

        # Add context comment if provided
        if additional_context:
            heuristic_lines.extend(
                [
                    f"    # Custom context: {additional_context}",
                    "",
                ]
            )

        # Add key definitions with comprehensive descriptions
        heuristic_lines.append("    # Define BIDS keys for each mapped group")
        for seq_info in sequences_info:
            key_name = seq_info["key_name"]
            bids_path = seq_info["bids_path"]

            # Create detailed description using all available variables
            desc_parts = []
            if seq_info["series_description"]:
                desc_parts.append(seq_info["series_description"])
            if (
                seq_info["protocol_name"]
                and seq_info["protocol_name"] != seq_info["series_description"]
            ):
                desc_parts.append(f"({seq_info['protocol_name']})")
            if seq_info["TR"] > 0:
                desc_parts.append(f"TR={seq_info['TR']}ms")
            if seq_info["TE"] > 0:
                desc_parts.append(f"TE={seq_info['TE']}ms")
            if seq_info["series_count"] > 1:
                desc_parts.append(f"n={seq_info['series_count']}")

            description = " ".join(desc_parts) if desc_parts else f"Group {key_name}"
            heuristic_lines.append(f'    {key_name} = create_key("{bids_path}")  # {description}')

        heuristic_lines.extend(["", "    # Initialize info dictionary"])

        # Build info dict initialization
        info_dict_lines = ["    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {"]
        for seq_info in sequences_info:
            key_name = seq_info["key_name"]
            info_dict_lines.append(f"        {key_name}: [],")
        info_dict_lines.append("    }")
        heuristic_lines.extend(info_dict_lines)

        heuristic_lines.extend(
            ["", "    # Mapping logic based on protocol names and descriptions"]
        )
        heuristic_lines.append("    for s in seqinfo:")

        # Add mapping logic for each sequence using ALL groupby variables
        for seq_info in sequences_info:
            key_name = seq_info["key_name"]
            protocol_name = seq_info["protocol_name"]
            series_description = seq_info["series_description"]
            sequence_name = seq_info["sequence_name"]
            image_type = seq_info["image_type"]
            is_motion_corrected = seq_info["is_motion_corrected"]
            is_derived = seq_info["is_derived"]
            TR = seq_info["TR"]
            TE = seq_info["TE"]

            # Build comprehensive condition using multiple variables for robust matching
            conditions = []

            # Primary matching on protocol and description
            if protocol_name:
                conditions.append(f'"{protocol_name.lower()}" in s.protocol_name.lower()')
            if series_description:
                conditions.append(
                    f'"{series_description.lower()}" in s.series_description.lower()'
                )
            if sequence_name:
                conditions.append(f'"{sequence_name.lower()}" in s.sequence_name.lower()')

            # Add TR/TE matching for more specificity (with tolerance)
            if TR > 0:
                conditions.append(f"abs(s.TR - {TR}) < 50")  # 50ms tolerance
            if TE > 0:
                conditions.append(f"abs(s.TE - {TE}) < 5")  # 5ms tolerance

            if conditions:
                condition_str = " and ".join(conditions)

                # Create descriptive comment
                comment_parts = []
                if series_description:
                    comment_parts.append(series_description)
                if protocol_name and protocol_name != series_description:
                    comment_parts.append(f"({protocol_name})")
                comment = " ".join(comment_parts) if comment_parts else f"Group {key_name}"

                heuristic_lines.append(f"        # {comment}")
                heuristic_lines.append(f"        if ({condition_str}):")

                # Add context-based filters
                filters_applied = False

                # Add derived filter if needed
                if additional_context and "derived" in additional_context.lower():
                    if is_derived:
                        heuristic_lines.append("            if s.is_derived:")
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    else:
                        heuristic_lines.append("            if not s.is_derived:")
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    filters_applied = True

                # Add motion correction filter if needed
                elif additional_context and "motion" in additional_context.lower():
                    if is_motion_corrected:
                        heuristic_lines.append("            if s.is_motion_corrected:")
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    else:
                        heuristic_lines.append("            if not s.is_motion_corrected:")
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    filters_applied = True

                # Add image_type filter if needed (e.g., for NORM vs non-NORM scans)
                elif additional_context and "image_type" in additional_context.lower():
                    image_type_filter = seq_info.get("image_type_filter", "")

                    if image_type_filter == "norm":
                        heuristic_lines.append('            if "NORM" in str(s.image_type):')
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    elif image_type_filter == "non_norm":
                        heuristic_lines.append('            if "NORM" not in str(s.image_type):')
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    elif image_type:
                        # Generic image_type matching
                        image_type_clean = (
                            image_type.replace("'", "").replace("[", "").replace("]", "")
                        )
                        heuristic_lines.append(
                            f'            if "{image_type_clean}" in str(s.image_type):'
                        )
                        heuristic_lines.append(
                            f"                info[{key_name}].append(s.series_id)"
                        )
                    else:
                        # Fallback - no specific image_type filter
                        heuristic_lines.append(f"            info[{key_name}].append(s.series_id)")
                    filters_applied = True

                # Default case - no specific filters
                if not filters_applied:
                    heuristic_lines.append(f"            info[{key_name}].append(s.series_id)")

                heuristic_lines.append("")

        heuristic_lines.extend(["    return info", ""])

        return "\n".join(heuristic_lines)

    def _create_image_type_variants(
        self, bids_path: str, image_type: str, additional_context: str | None = None
    ) -> list[tuple[str, str]]:
        """
        Create BIDS path variants based on image_type and context.

        Returns list of (bids_path, filter_suffix) tuples.
        """
        if not additional_context:
            return [(bids_path, "")]

        context_lower = additional_context.lower()

        # Handle T2w NORM case specifically
        if (
            "t2w" in context_lower
            and "norm" in context_lower
            and "_rec-norm_" in context_lower
            and "_T2w" in bids_path
        ):
            # Create two variants: one for NORM, one for non-NORM
            norm_path = bids_path.replace("_T2w", "_rec-norm_T2w")
            return [
                (norm_path, "norm"),  # NORM variant
                (bids_path, "non_norm"),  # Non-NORM variant
            ]

        # Handle other image_type cases generically
        elif "image_type" in context_lower and image_type:
            # For now, return the original path unless we have specific patterns
            return [(bids_path, "")]

        # Default case
        return [(bids_path, "")]
