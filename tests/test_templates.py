"""Tests for template functionality."""

import ast

import pytest

from llm_heuristics.utils.templates import HeuristicTemplate


@pytest.fixture
def heuristic_template():
    """Create a HeuristicTemplate instance for testing."""
    return HeuristicTemplate()


def test_template_initialization(heuristic_template):
    """Test HeuristicTemplate initialization."""
    assert heuristic_template.base_template is not None
    assert heuristic_template.prompt_template is not None


def test_generate_heuristic_skeleton_basic(heuristic_template):
    """Test basic heuristic skeleton generation."""
    mappings = {"anat_T1w": "sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w"}
    study_info = {"dicom_dir": "/test/dicom", "num_unique_groups": "1"}

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info)

    # Check that essential heudiconv elements are present
    assert "from heudiconv.utils import SeqInfo" in skeleton
    assert "def create_key(" in skeleton
    assert "def infotodict(" in skeleton
    assert "anat_T1w = create_key(" in skeleton
    assert study_info["dicom_dir"] in skeleton


def test_generate_heuristic_skeleton_with_context(heuristic_template):
    """Test heuristic skeleton generation with custom context."""
    mappings = {
        "anat_T1w": "sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w",
        "func_task_rest_bold": (
            "sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest_bold"
        ),
    }
    study_info = {"dicom_dir": "/test/dicom", "num_unique_groups": "2"}
    custom_context = "Use MPRAGE for T1w and EPI for BOLD sequences"

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info, custom_context)

    # Check that custom context is included
    assert custom_context in skeleton
    assert "Custom sequence selection rules:" in skeleton

    # Check that both mappings are included
    assert "anat_T1w = create_key(" in skeleton
    assert "func_task_rest_bold = create_key(" in skeleton


def test_generate_heuristic_skeleton_empty_mappings(heuristic_template):
    """Test heuristic skeleton generation with empty mappings."""
    mappings = {}
    study_info = {"dicom_dir": "/test/dicom", "num_unique_groups": "0"}

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info)

    # Should still generate valid skeleton with heudiconv imports
    assert "from heudiconv.utils import SeqInfo" in skeleton
    assert "def create_key(" in skeleton
    assert "def infotodict(" in skeleton


def test_generated_skeleton_is_valid_python(heuristic_template):
    """Test that generated skeleton is syntactically valid Python."""
    mappings = {
        "anat_T1w": "sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w",
        "anat_T2w": "sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T2w",
        "func_task_rest_bold": (
            "sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest_bold"
        ),
    }
    study_info = {"dicom_dir": "/data/test/dicom", "num_unique_groups": "3"}
    context = "Test context with special characters: @#$%^&*()"

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info, context)

    # Test that the generated code is syntactically valid
    try:
        ast.parse(skeleton)
    except SyntaxError as e:
        pytest.fail(f"Generated skeleton has syntax error: {e}")


def test_dynamic_convertall_import(heuristic_template):
    """Test that convertall.py content is dynamically imported."""
    mappings = {"test_key": "test_pattern"}
    study_info = {"dicom_dir": "/test", "num_unique_groups": "1"}

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info)

    # Check that actual heudiconv convertall imports are present (not hardcoded)
    assert "from __future__ import annotations" in skeleton
    assert "import logging" in skeleton
    assert "from typing import Optional" in skeleton
    assert 'lgr = logging.getLogger("heudiconv")' in skeleton

    # Check that the create_key function signature matches heudiconv's
    assert "def create_key(" in skeleton
    assert "template: Optional[str]" in skeleton
    assert 'outtype: tuple[str, ...] = ("nii.gz",)' in skeleton
    assert "annotation_classes: None = None" in skeleton


def test_seqinfo_documentation_extraction(heuristic_template):
    """Test that SeqInfo documentation is extracted from convertall.py."""
    mappings = {"test_key": "test_pattern"}
    study_info = {"dicom_dir": "/test", "num_unique_groups": "1"}

    skeleton = heuristic_template.generate_heuristic_skeleton(mappings, study_info)

    # Check that SeqInfo field documentation is included
    # This should come from the actual convertall.py docstring
    expected_fields = [
        "total_files_till_now",
        "example_dcm_file",
        "series_id",
        "dcm_dir_name",
        "dim1",
        "dim2",
        "dim3",
        "dim4",
        "TR",
        "TE",
        "protocol_name",
        "is_motion_corrected",
        "series_description",
    ]

    # At least some of these fields should be documented
    fields_found = sum(1 for field in expected_fields if field in skeleton)
    assert fields_found >= 8, f"Expected SeqInfo documentation, found only {fields_found} fields"


def test_create_heuristic_prompt(heuristic_template):
    """Test heuristic prompt creation."""
    sequences_info = [
        {
            "bids_path": "sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w",
            "bids_modality": "anat",
            "bids_suffix": "T1w",
            "series_count": 1,
            "protocol_name": "T1_MPRAGE",
        }
    ]
    study_info = {"dicom_dir": "/test/dicom", "num_unique_groups": "1"}
    additional_context = "Use high-resolution sequences"

    prompt = heuristic_template.create_heuristic_prompt(
        sequences_info, study_info, additional_context
    )

    assert "T1_MPRAGE" in prompt
    assert "Use high-resolution sequences" in prompt
    assert "/test/dicom" in prompt


def test_get_create_key_function(heuristic_template):
    """Test create_key function generation."""
    create_key_func = heuristic_template.get_create_key_function()

    assert "def create_key(" in create_key_func
    assert "Template must be a valid format string" in create_key_func
    assert "return template, outtype, annotation_classes" in create_key_func
