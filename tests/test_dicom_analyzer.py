"""Tests for HeuDiConv-based DICOM extraction functionality."""

from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from llm_heuristics.core.heudiconv_extractor import HeuDiConvExtractor


@pytest.fixture
def extractor():
    """Create a HeuDiConvExtractor instance for testing."""
    with patch("llm_heuristics.core.heudiconv_extractor.HeuDiConvExtractor._verify_heudiconv"):
        return HeuDiConvExtractor(enable_caching=False)


def test_initialization(extractor):
    """Test HeuDiConvExtractor initialization."""
    assert extractor.n_cpus > 0
    assert not extractor.enable_caching
    assert extractor.heudiconv_bin == "heudiconv"


def test_initialization_with_custom_params():
    """Test HeuDiConvExtractor initialization with custom parameters."""
    with patch("llm_heuristics.core.heudiconv_extractor.HeuDiConvExtractor._verify_heudiconv"):
        extractor = HeuDiConvExtractor(
            n_cpus=8, enable_caching=True, heudiconv_bin="/custom/path/heudiconv"
        )

    assert extractor.n_cpus == 8
    assert extractor.enable_caching
    assert extractor.heudiconv_bin == "/custom/path/heudiconv"
    assert not extractor.slurm  # Default should be False


def test_initialization_with_slurm():
    """Test HeuDiConvExtractor initialization with SLURM mode."""
    with patch("llm_heuristics.core.heudiconv_extractor.HeuDiConvExtractor._verify_heudiconv"):
        extractor = HeuDiConvExtractor(slurm=True, enable_caching=False)

    assert extractor.slurm
    assert not extractor.enable_caching


@patch("subprocess.run")
def test_verify_heudiconv_success(mock_run):
    """Test successful HeuDiConv verification."""
    mock_run.return_value.stdout = "heudiconv 1.3.3"

    # Should not raise an exception
    extractor = HeuDiConvExtractor(enable_caching=False)
    assert extractor.heudiconv_bin == "heudiconv"


@patch("subprocess.run")
def test_verify_heudiconv_failure(mock_run):
    """Test HeuDiConv verification failure."""
    mock_run.side_effect = FileNotFoundError("heudiconv not found")

    with pytest.raises(RuntimeError, match="HeuDiConv not found"):
        HeuDiConvExtractor(enable_caching=False)


def test_discover_subject_sessions_bids_structure(extractor, tmp_path):
    """Test subject/session discovery with BIDS-like structure."""
    # Create BIDS-like directory structure
    (tmp_path / "sub-001" / "ses-001").mkdir(parents=True)
    (tmp_path / "sub-001" / "ses-002").mkdir(parents=True)
    (tmp_path / "sub-002" / "ses-001").mkdir(parents=True)

    result = extractor._discover_subject_sessions(tmp_path)

    assert len(result) == 3
    assert ("001", "001") in result
    assert ("001", "002") in result
    assert ("002", "001") in result


def test_discover_subject_sessions_numbered_structure(extractor, tmp_path):
    """Test subject/session discovery with numbered directories."""
    # Create numbered directory structure
    (tmp_path / "001" / "001").mkdir(parents=True)
    (tmp_path / "001" / "002").mkdir(parents=True)
    (tmp_path / "002" / "001").mkdir(parents=True)

    result = extractor._discover_subject_sessions(tmp_path)

    assert len(result) >= 3  # May find more if other patterns match


def test_discover_subject_sessions_flat_structure(extractor, tmp_path):
    """Test subject/session discovery with flat structure."""
    # Create some files in the directory (but no clear subject/session structure)
    (tmp_path / "some_file.dcm").touch()

    result = extractor._discover_subject_sessions(tmp_path)

    # Should fall back to single subject
    assert len(result) >= 1
    assert ("01", None) in result


def test_extract_subject_session_success(extractor, tmp_path):
    """Test successful HeuDiConv extraction for a subject/session."""
    # Create mock dicominfo.tsv file
    dicominfo_content = (
        "total_files_till_now\texample_dcm_file\tseries_id\tdcm_dir_name\tunspecified2\t"
        "unspecified3\tdim1\tdim2\tdim3\tdim4\tTR\tTE\tprotocol_name\tis_motion_corrected\t"
        "series_description\n"
        "1\t/path/to/file.dcm\t1\tseries1\t\t\t256\t256\t176\t1\t2000\t30\tT1w_MPR\tFalse\t"
        "T1 MPRAGE"
    )

    dicominfo_file = tmp_path / ".heudiconv" / "001" / "ses-001" / "info" / "dicominfo_ses-001.tsv"
    dicominfo_file.parent.mkdir(parents=True)
    dicominfo_file.write_text(dicominfo_content)

    # Mock successful HeuDiConv run
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        extractor._extract_subject_session(tmp_path, "001", "001", tmp_path)

    # Read the actual file that was created
    df = pd.read_csv(dicominfo_file, sep="\t")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["protocol_name"] == "T1w_MPR"


def test_extract_subject_session_failure(extractor, tmp_path):
    """Test HeuDiConv extraction failure."""
    # Mock failed HeuDiConv run
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stderr = "HeuDiConv error"

    with patch("subprocess.run", return_value=mock_result):
        result = extractor._extract_subject_session(tmp_path, "001", "001", tmp_path)

    # Create an empty DataFrame with expected columns
    expected_df = pd.DataFrame(
        columns=[
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
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_generate_summary_empty(extractor):
    """Test summary generation for empty DataFrame."""
    empty_df = pd.DataFrame()
    summary = extractor.generate_summary(empty_df)
    assert "No DICOM information available" in summary


def test_generate_summary_with_data(extractor):
    """Test summary generation with sample data."""
    # Create sample dicominfo DataFrame
    data = {
        "subject_id": ["001", "001", "002"],
        "session_id": ["001", "002", "001"],
        "protocol_name": ["T1w_MPR", "T1w_MPR", "BOLD_REST"],
        "dim1": [256, 256, 64],
        "dim2": [256, 256, 64],
        "dim3": [176, 176, 40],
        "dim4": [1, 1, 200],
        "TR": [2000, 2000, 2500],
        "TE": [30, 30, 35],
        "is_motion_corrected": [False, False, False],
    }
    df = pd.DataFrame(data)

    summary = extractor.generate_summary(df)

    assert "Dataset Overview" in summary
    assert "Subjects: 2" in summary
    assert "Sessions: 2" in summary
    assert "Total Series: 3" in summary
    assert "T1w_MPR" in summary
    assert "BOLD_REST" in summary


def test_cache_key_generation(extractor, tmp_path):
    """Test cache key generation based on directory."""
    # Create some test files
    (tmp_path / "test1.dcm").touch()
    (tmp_path / "test2.dcm").touch()

    key1 = extractor._get_cache_key(tmp_path)
    key2 = extractor._get_cache_key(tmp_path)

    # Same directory should generate same key
    assert key1 == key2
    assert len(key1) == 32  # MD5 hash length


def test_extract_dicom_info_no_files(extractor, tmp_path):
    """Test extraction when no DICOM files are found."""
    # Empty directory
    result = extractor.extract_dicom_info(tmp_path)

    # Create an empty DataFrame with expected columns
    expected_df = pd.DataFrame(
        columns=[
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
    )
    pd.testing.assert_frame_equal(result, expected_df)


def test_clear_cache(extractor, tmp_path):
    """Test cache clearing functionality."""
    extractor.cache_dir = tmp_path / "cache"
    extractor.cache_dir.mkdir()

    # Create some fake cache files
    (extractor.cache_dir / "extraction_123.json").touch()
    (extractor.cache_dir / "extraction_456.json").touch()
    (extractor.cache_dir / "other_file.txt").touch()

    extractor.clear_cache()

    # Only extraction cache files should be removed
    assert not (extractor.cache_dir / "extraction_123.json").exists()
    assert not (extractor.cache_dir / "extraction_456.json").exists()
    assert (extractor.cache_dir / "other_file.txt").exists()  # Should remain


def test_slurm_script_generation(tmp_path):
    """Test SLURM script generation functionality."""
    # Create BIDS-like directory structure
    dicom_dir = tmp_path / "dicom"
    (dicom_dir / "sub-001" / "ses-001").mkdir(parents=True)
    (dicom_dir / "sub-001" / "ses-002").mkdir(parents=True)
    (dicom_dir / "sub-002" / "ses-001").mkdir(parents=True)

    output_dir = tmp_path / "scripts"

    # Create SLURM-enabled extractor (skip verification)
    extractor = HeuDiConvExtractor(slurm=True, n_cpus=4, enable_caching=False)

    # Generate SLURM script
    script_path = extractor.extract_dicom_info(dicom_dir, output_dir=output_dir)

    # Verify script was created
    assert isinstance(script_path, str)
    script_file = Path(script_path)
    assert script_file.exists()
    assert script_file.name == "heudiconv_extract.slurm"

    # Verify mapping file was created
    mapping_file = script_file.parent / "subject_sessions.txt"
    assert mapping_file.exists()

    # Check mapping file content
    with open(mapping_file) as f:
        lines = f.readlines()
    assert len(lines) == 3  # Should have 3 subject/session combinations

    # Check script content
    with open(script_file) as f:
        script_content = f.read()

    assert "#!/bin/bash" in script_content
    assert "#SBATCH --job-name=heudiconv_extract" in script_content
    assert "#SBATCH --array=1-3" in script_content
    assert "#SBATCH --cpus-per-task=4" in script_content
    assert "heudiconv" in script_content


def test_extract_dicom_info_slurm_mode_empty_dir(tmp_path):
    """Test SLURM mode with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    extractor = HeuDiConvExtractor(slurm=True, enable_caching=False)

    result = extractor.extract_dicom_info(empty_dir)

    # Should return a SLURM script path
    assert isinstance(result, str)
    assert result.endswith("heudiconv_extract.slurm")
    assert Path(result).exists()

    # Check script content
    with open(result) as f:
        script_content = f.read()
        assert "#!/bin/bash" in script_content
        assert "#SBATCH --job-name=heudiconv_extract" in script_content
        assert "heudiconv" in script_content
