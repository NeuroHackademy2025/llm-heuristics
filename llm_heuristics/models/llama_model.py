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

        # Prepare context for the LLM
        context = self._prepare_heuristic_context(dicom_summary, study_info, additional_context)

        # Generate heuristic
        heuristic_code = self._generate_with_llm(context)

        # Post-process and validate
        heuristic_code = self._post_process_heuristic(heuristic_code)

        return heuristic_code

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

        # Create series information summary from grouped data
        sequences_info = []
        for _, row in dicom_summary.iterrows():
            # Handle both grouped and ungrouped DataFrames
            if "series_count" in row:
                # This is grouped data from SeriesGrouper
                sequences_info.append(
                    {
                        "series_number": row.get("representative_series_id", "Unknown"),
                        "description": row.get("series_description", "Unknown"),
                        "protocol": row.get("protocol_name", "Unknown"),
                        "modality": "MRI",  # Default to MRI for neuroimaging data
                        "dimensions": (
                            f"{row.get('dim1', 0)}x{row.get('dim2', 0)}x"
                            f"{row.get('dim3', 0)}x{row.get('dim4', 1)}"
                        ),
                        "num_files": row.get("representative_files", 0),
                        "sequence": row.get("sequence_name", "Unknown"),
                        "variant": "Unknown",
                        "tr": row.get("TR", 0),
                        "te": row.get("TE", 0),
                        "is_derived": row.get("is_derived", False),
                        "series_count": row.get("series_count", 1),
                        "bids_modality": row.get("bids_modality", "Unknown"),
                        "bids_suffix": row.get("bids_suffix", "Unknown"),
                        "bids_path": row.get("bids_path", "Unknown"),
                        "bids_confidence": row.get("bids_confidence", 0.0),
                    }
                )
            else:
                # This is individual series data
                sequences_info.append(
                    {
                        "series_number": row.get("series_id", row.get("series_number", "Unknown")),
                        "description": row.get("series_description", "Unknown"),
                        "protocol": row.get("protocol_name", "Unknown"),
                        "modality": "MRI",  # Default to MRI for neuroimaging data
                        "dimensions": (
                            f"{row.get('dim1', 0)}x{row.get('dim2', 0)}x"
                            f"{row.get('dim3', 0)}x{row.get('dim4', 1)}"
                        ),
                        "num_files": row.get("series_files", row.get("num_files", 0)),
                        "sequence": row.get(
                            "scanning_sequence", row.get("sequence_name", "Unknown")
                        ),
                        "variant": row.get("sequence_variant", "Unknown"),
                        "tr": row.get("TR", row.get("repetition_time", 0)),
                        "te": row.get("TE", row.get("echo_time", 0)),
                        "is_derived": row.get("is_derived", False),
                        "series_count": 1,
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
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response.strip()

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

        # Ensure proper imports
        if "import os" not in heuristic_code:
            heuristic_code = "import os\n\n" + heuristic_code

        # Ensure create_key function is present
        if "def create_key" not in heuristic_code:
            heuristic_code = self.template.get_create_key_function() + "\n\n" + heuristic_code

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
