"""Tests for BIDS integration functionality."""

import pytest

from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration


@pytest.fixture
def bids_integration():
    """Create a BIDSSchemaIntegration instance with actual BIDS schema for testing."""
    # Use actual BIDS schema - this will download/load the real schema
    try:
        integration = BIDSSchemaIntegration()
        # Ensure schema is loaded by accessing it
        _ = integration.bids_schema
        return integration
    except Exception as e:
        pytest.skip(f"Could not load actual BIDS schema: {e}")


def test_initialization(bids_integration):
    """Test BIDSSchemaIntegration initialization."""
    assert bids_integration.schema_version == "latest"
    assert bids_integration._schema is not None


def test_entities_property(bids_integration):
    """Test entities property access."""
    entities = bids_integration.entities
    assert isinstance(entities, dict)
    assert "subject" in entities
    assert "session" in entities


def test_modalities_property(bids_integration):
    """Test modalities property access."""
    modalities = bids_integration.modalities
    assert isinstance(modalities, dict)
    assert "mri" in modalities  # Real BIDS schema uses 'mri' as top-level modality
    assert "eeg" in modalities
    assert "meg" in modalities


def test_suffixes_property(bids_integration):
    """Test suffixes property access."""
    suffixes = bids_integration.suffixes
    assert isinstance(suffixes, dict)
    assert "T1w" in suffixes
    assert "bold" in suffixes


def test_get_valid_entities_for_modality(bids_integration):
    """Test getting valid entities for a modality."""
    entities = bids_integration.get_valid_entities_for_modality("anat")
    assert isinstance(entities, list)


def test_get_valid_suffixes_for_modality(bids_integration):
    """Test getting valid suffixes for a modality."""
    suffixes = bids_integration.get_valid_suffixes_for_modality("anat")
    assert isinstance(suffixes, list)


def test_classify_sequence_to_bids_t1w(bids_integration):
    """Test sequence classification for T1w."""
    classification = bids_integration.classify_sequence_to_bids(
        series_description="T1_MPRAGE",
        protocol_name="T1w_MPR",
        sequence_type="MPRAGE",
        dimensions=(256, 256, 176, 1),
    )

    assert classification["modality"] == "anat"
    assert classification["suffix"] == "T1w"
    assert classification["confidence"] > 0.8


def test_classify_sequence_to_bids_bold(bids_integration):
    """Test sequence classification for BOLD."""
    classification = bids_integration.classify_sequence_to_bids(
        series_description="BOLD_rest",
        protocol_name="resting_state_fmri",
        sequence_type="EPI",
        dimensions=(64, 64, 35, 200),
    )

    assert classification["modality"] == "func"
    assert classification["suffix"] == "bold"
    assert "task" in classification["entities"]


def test_classify_sequence_to_bids_dwi(bids_integration):
    """Test sequence classification for DWI."""
    classification = bids_integration.classify_sequence_to_bids(
        series_description="DTI_30dir",
        protocol_name="diffusion_AP",
        sequence_type="DWI",
        dimensions=(128, 128, 64, 31),
    )

    assert classification["modality"] == "dwi"
    assert classification["suffix"] == "dwi"


def test_generate_bids_path_template(bids_integration):
    """Test BIDS path template generation."""
    path = bids_integration.generate_bids_path_template(
        modality="anat", suffix="T1w", entities={}, include_session=True
    )

    assert "sub-{subject}" in path
    assert "{session}" in path
    assert "/anat/" in path
    assert "_T1w" in path


def test_generate_bids_path_template_no_session(bids_integration):
    """Test BIDS path template generation without session."""
    path = bids_integration.generate_bids_path_template(
        modality="func", suffix="bold", entities={"task": "rest"}, include_session=False
    )

    assert "sub-{subject}" in path
    assert "{session}" not in path
    assert "/func/" in path
    assert "_bold" in path


def test_validate_bids_path_valid(bids_integration):
    """Test BIDS path validation for valid path."""
    valid_path = "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz"

    result = bids_integration.validate_bids_path(valid_path)

    assert result["valid"] is True
    assert len(result["errors"]) == 0
    assert "sub" in result["entities"]
    assert "ses" in result["entities"]


def test_validate_bids_path_invalid(bids_integration):
    """Test BIDS path validation for invalid path."""
    invalid_path = "subject-01/T1w.nii.gz"

    result = bids_integration.validate_bids_path(invalid_path)

    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_get_schema_version_info(bids_integration):
    """Test schema version information retrieval."""
    info = bids_integration.get_schema_version_info()

    assert isinstance(info, dict)
    assert "schema_version" in info
    assert "num_entities" in info
    assert "num_modalities" in info


def test_integration_with_dicom_metadata(bids_integration):
    """Test integration with DICOM analyzer metadata format."""
    # Simulate metadata that would come from DicomAnalyzer
    # (using non-PHI tags that would pass through exclusion filter)
    simulated_metadata = {
        "series_description": "T1_MPRAGE",
        "protocol_name": "T1w_MPR",
        "sequence_name": "MPRAGE",
        "modality": "MR",
        "manufacturer": "Siemens",
        "repetition_time": 2000.0,
        "echo_time": 2.98,
        "flip_angle": 9.0,
        "rows": 256,
        "columns": 256,
        "number_of_slices": 176,
        "number_of_temporal_positions": 1,
    }

    # Test classification using the metadata format from DicomAnalyzer
    classification = bids_integration.classify_sequence_to_bids(
        series_description=simulated_metadata["series_description"],
        protocol_name=simulated_metadata["protocol_name"],
        sequence_type=simulated_metadata["sequence_name"],
        dimensions=(
            simulated_metadata["rows"],
            simulated_metadata["columns"],
            simulated_metadata["number_of_slices"],
            simulated_metadata["number_of_temporal_positions"],
        ),
    )

    assert classification["modality"] == "anat"
    assert classification["suffix"] == "T1w"
    assert classification["confidence"] > 0.8
