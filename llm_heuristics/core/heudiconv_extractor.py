"""HeuDiConv-based DICOM information extraction."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class HeuDiConvExtractor:
    """Extract DICOM information using HeuDiConv's optimized scanning.

    This class leverages HeuDiConv's proven DICOM analysis capabilities
    instead of reinventing the wheel with custom DICOM parsing.

    Privacy Note: All analysis happens locally using HeuDiConv.
    No DICOM data is sent to external services.
    """

    def __init__(
        self,
        n_cpus: int | None = None,
        enable_caching: bool = True,
        cache_dir: Path | None = None,
        heudiconv_bin: str = "heudiconv",
        slurm: bool = False,
    ) -> None:
        """
        Initialize the HeuDiConv DICOM extractor.

        Parameters
        ----------
        n_cpus : int | None
            Number of CPU cores for parallel processing. Passed to HeuDiConv.
        enable_caching : bool
            Whether to cache extraction results to avoid re-processing.
        cache_dir : Path | None
            Directory for cache files. If None, uses system temp directory.
        heudiconv_bin : str
            Path to heudiconv binary. Default: "heudiconv" (assumes in PATH).
        slurm : bool
            Whether to generate SLURM job scripts instead of running directly.
        """
        self.n_cpus = n_cpus or min(32, (os.cpu_count() or 1) + 4)
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir or Path.home() / ".cache" / "llm_heuristics" / "heudiconv"
        self.heudiconv_bin = heudiconv_bin
        self.slurm = slurm

        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Verify HeuDiConv is available (skip if using SLURM mode)
        if not self.slurm:
            self._verify_heudiconv()

        logger.info(
            "HeuDiConv Extractor initialized: n_cpus=%d, caching=%s, binary=%s, slurm=%s",
            self.n_cpus,
            self.enable_caching,
            self.heudiconv_bin,
            self.slurm,
        )

    def _verify_heudiconv(self) -> None:
        """Verify HeuDiConv is installed and accessible."""
        try:
            result = subprocess.run(
                [self.heudiconv_bin, "--version"], capture_output=True, text=True, timeout=10
            )
            logger.info("Found HeuDiConv version: %s", result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"HeuDiConv not found or not working (binary: {self.heudiconv_bin}). "
                "Please install HeuDiConv: pip install heudiconv[all] or use Docker. "
                f"Error: {e}"
            ) from e

    def extract_dicom_info(
        self, dicom_dir: Path, output_dir: Path | None = None
    ) -> pd.DataFrame | str:
        """
        Extract DICOM information using HeuDiConv's optimized scanner.

        This method discovers subjects and sessions automatically and either runs
        HeuDiConv directly or generates SLURM job scripts for cluster execution.

        Parameters
        ----------
        dicom_dir : Path
            Root directory containing DICOM files. Can be organized in any structure.
        output_dir : Path | None
            Output directory for SLURM scripts (only used in SLURM mode).

        Returns
        -------
        pd.DataFrame | str
            If slurm=False: Consolidated DICOM information from all discovered sessions.
            If slurm=True: Path to generated SLURM script file.
        """
        logger.info("Extracting DICOM information from: %s", dicom_dir)

        # Discover subjects and sessions
        subject_sessions = self._discover_subject_sessions(dicom_dir)

        if not subject_sessions:
            logger.warning("No subjects/sessions discovered in %s", dicom_dir)
            if self.slurm:
                return "# No subjects/sessions found - no SLURM script generated"
            return pd.DataFrame(
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

        logger.info("Discovered %d subject/session combinations", len(subject_sessions))

        # Handle SLURM mode
        if self.slurm:
            return self._generate_slurm_script(dicom_dir, subject_sessions, output_dir)

        # Check cache for direct execution
        if self.enable_caching:
            cached_result = self._load_cached_extraction(dicom_dir)
            if cached_result is not None:
                logger.info("Using cached DICOM extraction results")
                return cached_result

        # Create temporary working directory for HeuDiConv
        with tempfile.TemporaryDirectory(prefix="heudiconv_extract_") as temp_dir:
            temp_path = Path(temp_dir)
            all_dicominfo = []

            # Process each subject/session combination
            for subject_id, session_id in subject_sessions:
                try:
                    dicominfo_df = self._extract_subject_session(
                        dicom_dir, subject_id, session_id, temp_path
                    )
                    if not dicominfo_df.empty:
                        # Add subject/session context
                        dicominfo_df["subject_id"] = subject_id
                        dicominfo_df["session_id"] = session_id or "single_session"
                        all_dicominfo.append(dicominfo_df)

                except Exception as e:
                    logger.warning(
                        "Failed to extract DICOM info for subject=%s, session=%s: %s",
                        subject_id,
                        session_id,
                        e,
                    )
                    continue

            # Consolidate all results
            if all_dicominfo:
                consolidated_df = pd.concat(all_dicominfo, ignore_index=True)
                logger.info(
                    "Successfully extracted DICOM info: %d total series across %d subjects",
                    len(consolidated_df),
                    len(subject_sessions),
                )
            else:
                consolidated_df = pd.DataFrame(
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
                logger.warning("No DICOM information could be extracted")

            # Cache results
            if self.enable_caching and not consolidated_df.empty:
                self._save_cached_extraction(dicom_dir, consolidated_df)

            return consolidated_df

    def _generate_slurm_script(
        self,
        dicom_dir: Path,
        subject_sessions: list[tuple[str, str | None]],
        output_dir: Path | None,
    ) -> str:
        """
        Generate SLURM job array script for HeuDiConv extraction.

        Parameters
        ----------
        dicom_dir : Path
            Root DICOM directory
        subject_sessions : list[tuple[str, str | None]]
            List of (subject_id, session_id) combinations
        output_dir : Path | None
            Directory to save SLURM script

        Returns
        -------
        str
            Path to generated SLURM script file
        """
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        script_path = output_dir / "heudiconv_extract.slurm"

        # Create subject/session mapping file
        mapping_file = output_dir / "subject_sessions.txt"
        with open(mapping_file, "w") as f:
            for i, (subject, session) in enumerate(subject_sessions, 1):
                session_str = session if session else "NONE"
                f.write(f"{i}\t{subject}\t{session_str}\n")

        # Generate SLURM script
        script_content = self._create_slurm_script_content(
            dicom_dir, len(subject_sessions), mapping_file, output_dir
        )

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        script_path.chmod(0o755)

        logger.info("Generated SLURM script: %s", script_path)
        logger.info("Subject/session mapping: %s", mapping_file)
        logger.info("Run with: sbatch %s", script_path)

        return str(script_path)

    def _create_slurm_script_content(
        self, dicom_dir: Path, num_jobs: int, mapping_file: Path, output_dir: Path
    ) -> str:
        """Create the content of the SLURM script."""
        return f"""#!/bin/bash
#SBATCH --job-name=heudiconv_extract
#SBATCH --output={output_dir}/heudiconv_extract_%A_%a.out
#SBATCH --error={output_dir}/heudiconv_extract_%A_%a.err
#SBATCH --array=1-{num_jobs}
#SBATCH --cpus-per-task={self.n_cpus}
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00

# HeuDiConv DICOM extraction job array
# Generated by llm-heuristics

# Set up environment
module load python/3.9  # Adjust as needed for your cluster
source activate heudiconv  # Adjust as needed

# Get subject and session for this array job
MAPPING_FILE="{mapping_file}"
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" $MAPPING_FILE)
TASK_ID=$(echo $LINE | cut -f1)
SUBJECT=$(echo $LINE | cut -f2)
SESSION=$(echo $LINE | cut -f3)

echo "Task ${{SLURM_ARRAY_TASK_ID}}: Processing subject=$SUBJECT, session=$SESSION"

# Prepare HeuDiConv command
DICOM_DIR="{dicom_dir}"
OUTPUT_DIR="{output_dir}/heudiconv_output"
HEUDICONV_BIN="{self.heudiconv_bin}"

# Build command
CMD="$HEUDICONV_BIN --files $DICOM_DIR/**/*.dcm -o $OUTPUT_DIR -f convertall \\
    -s $SUBJECT -c none --overwrite"

# Add session if not NONE
if [ "$SESSION" != "NONE" ]; then
    CMD="$CMD -ss $SESSION"
fi

# Add parallel jobs
if [ {self.n_cpus} -gt 1 ]; then
    CMD="$CMD --jobs {self.n_cpus}"
fi

echo "Running: $CMD"

# Execute HeuDiConv
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Task ${{SLURM_ARRAY_TASK_ID}} completed successfully"
else
    echo "Task ${{SLURM_ARRAY_TASK_ID}} failed with exit code $?"
    exit 1
fi
"""

    def _discover_subject_sessions(self, dicom_dir: Path) -> list[tuple[str, str | None]]:
        """
        Discover subject IDs and session IDs from directory structure.

        This tries to be smart about different directory organizations:
        - sub-001/ses-001/... (BIDS-like)
        - subject_001/session_001/...
        - 001/001/...
        - Or just flat structure with numbered directories

        Returns list of (subject_id, session_id) tuples.
        """
        subject_sessions = []

        # Look for BIDS-style directories first
        bids_subjects = list(dicom_dir.glob("sub-*"))
        if bids_subjects:
            for subj_dir in bids_subjects:
                subject_id = subj_dir.name[4:]  # Remove 'sub-' prefix

                # Look for sessions
                sessions = list(subj_dir.glob("ses-*"))
                if sessions:
                    for ses_dir in sessions:
                        session_id = ses_dir.name[4:]  # Remove 'ses-' prefix
                        subject_sessions.append((subject_id, session_id))
                else:
                    # No sessions, just subject
                    subject_sessions.append((subject_id, None))
        else:
            # Try other common patterns
            # Look for numbered directories that might be subjects
            numbered_dirs = [
                d for d in dicom_dir.iterdir() if d.is_dir() and re.match(r"\d+", d.name)
            ]

            if numbered_dirs:
                for subj_dir in numbered_dirs[:10]:  # Limit to first 10 for safety
                    subject_id = subj_dir.name

                    # Check if this directory has subdirectories (sessions)
                    subdirs = [d for d in subj_dir.iterdir() if d.is_dir()]
                    session_dirs = [d for d in subdirs if re.match(r"\d+|ses", d.name)]

                    if session_dirs:
                        for ses_dir in session_dirs:
                            session_id = ses_dir.name
                            subject_sessions.append((subject_id, session_id))
                    else:
                        # No clear sessions
                        subject_sessions.append((subject_id, None))
            else:
                # Fallback: treat the entire directory as one subject
                subject_sessions.append(("01", None))

        return subject_sessions[:50]  # Reasonable limit for discovery

    def _extract_subject_session(
        self, dicom_dir: Path, subject_id: str, session_id: str | None, temp_dir: Path
    ) -> pd.DataFrame:
        """Extract DICOM info for a specific subject/session using HeuDiConv."""

        # Build HeuDiConv command
        cmd = [
            self.heudiconv_bin,
            "--files",
            str(dicom_dir / "**" / "*.dcm"),  # Find all .dcm files
            "-o",
            str(temp_dir),
            "-f",
            "convertall",
            "-s",
            subject_id,
            "-c",
            "none",  # Don't convert, just extract info
            "--overwrite",
        ]

        # Add session if specified
        if session_id:
            cmd.extend(["-ss", session_id])

        # Add parallel processing
        if self.n_cpus > 1:
            cmd.extend(["--jobs", str(self.n_cpus)])

        logger.debug("Running HeuDiConv: %s", " ".join(cmd))

        try:
            # Run HeuDiConv
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=temp_dir,
            )

            if result.returncode != 0:
                logger.warning(
                    "HeuDiConv failed for subject=%s, session=%s. stderr: %s",
                    subject_id,
                    session_id,
                    result.stderr,
                )
                return pd.DataFrame(
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

            # Find the generated dicominfo.tsv file
            if session_id:
                info_path = temp_dir / ".heudiconv" / subject_id / f"ses-{session_id}" / "info"
                dicominfo_file = info_path / f"dicominfo_ses-{session_id}.tsv"
            else:
                info_path = temp_dir / ".heudiconv" / subject_id / "info"
                dicominfo_file = info_path / "dicominfo.tsv"

            if not dicominfo_file.exists():
                # Try alternative locations
                info_files = list(temp_dir.rglob("dicominfo*.tsv"))
                if info_files:
                    dicominfo_file = info_files[0]
                else:
                    logger.warning(
                        "No dicominfo.tsv file found for subject=%s, session=%s",
                        subject_id,
                        session_id,
                    )
                    return pd.DataFrame(
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

            # Load the TSV file
            try:
                df = pd.read_csv(dicominfo_file, sep="\t")
                logger.debug("Loaded dicominfo.tsv with %d series", len(df))
                return df
            except Exception as e:
                logger.warning("Failed to read dicominfo.tsv: %s", e)
                return pd.DataFrame(
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

        except subprocess.TimeoutExpired:
            logger.warning(
                "HeuDiConv timed out for subject=%s, session=%s", subject_id, session_id
            )
            return pd.DataFrame(
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
        except Exception as e:
            logger.warning(
                "Error running HeuDiConv for subject=%s, session=%s: %s", subject_id, session_id, e
            )
            return pd.DataFrame(
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

    def _get_cache_key(self, dicom_dir: Path) -> str:
        """Generate cache key based on directory structure and modification times."""
        import hashlib

        # Get directory stats
        dir_stat = dicom_dir.stat()

        # Sample a few files to detect changes
        dcm_files = list(dicom_dir.rglob("*.dcm"))[:10]  # Sample first 10
        file_info = []

        for f in dcm_files:
            try:
                stat = f.stat()
                file_info.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
            except OSError:
                continue

        # Create hash
        cache_data = f"{dicom_dir}:{dir_stat.st_mtime}:{len(dcm_files)}:{'|'.join(file_info)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _load_cached_extraction(self, dicom_dir: Path) -> pd.DataFrame | None:
        """Load cached extraction results if available and valid."""
        cache_key = self._get_cache_key(dicom_dir)
        cache_file = self.cache_dir / f"extraction_{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Convert back to DataFrame
            df = pd.DataFrame(data["dicom_info"])
            logger.debug("Loaded cached extraction with %d series", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to load cached extraction: %s", e)
            return None

    def _save_cached_extraction(self, dicom_dir: Path, df: pd.DataFrame) -> None:
        """Save extraction results to cache."""
        cache_key = self._get_cache_key(dicom_dir)
        cache_file = self.cache_dir / f"extraction_{cache_key}.json"

        try:
            # Convert DataFrame to JSON-serializable format
            data = {
                "dicom_info": df.to_dict("records"),
                "extraction_time": pd.Timestamp.now().isoformat(),
                "source_directory": str(dicom_dir),
            }

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Cached extraction results to %s", cache_file)
        except Exception as e:
            logger.warning("Failed to save cached extraction: %s", e)

    def clear_cache(self) -> None:
        """Clear all cached extraction results."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("extraction_*.json"):
                cache_file.unlink()
            logger.info("Cleared HeuDiConv extraction cache")

    def generate_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a human-readable summary of the DICOM dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DICOM information DataFrame from extract_dicom_info()

        Returns
        -------
        str
            Multi-line summary of the dataset characteristics
        """
        if df.empty:
            return "No DICOM information available."

        summary_lines = ["=== DICOM Dataset Summary (via HeuDiConv) ===", ""]

        # Overall statistics
        n_subjects = df["subject_id"].nunique() if "subject_id" in df else 1
        n_sessions = df["session_id"].nunique() if "session_id" in df else 1
        n_series = len(df)

        summary_lines.extend(
            [
                "Dataset Overview:",
                f"   Subjects: {n_subjects}",
                f"   Sessions: {n_sessions}",
                f"   Total Series: {n_series}",
                "",
            ]
        )

        # Protocol summary
        if "protocol_name" in df:
            protocols = df["protocol_name"].value_counts()
            summary_lines.extend(
                [
                    f"Scan Protocols ({len(protocols)} unique):",
                ]
            )
            for protocol, count in protocols.head(10).items():
                summary_lines.append(f"   {protocol}: {count} series")
            if len(protocols) > 10:
                summary_lines.append(f"   ... and {len(protocols) - 10} more")
            summary_lines.append("")

        # Dimension analysis
        if all(col in df for col in ["dim1", "dim2", "dim3", "dim4"]):
            summary_lines.extend(
                [
                    "Image Dimensions:",
                    f"   Matrix sizes: {df[['dim1', 'dim2']].drop_duplicates().shape[0]} "
                    f"unique combinations",
                    f"   Slice counts (dim3): {df['dim3'].min()}-{df['dim3'].max()}",
                    f"   Time points (dim4): {df['dim4'].min()}-{df['dim4'].max()}",
                    "",
                ]
            )

        # Sequence timing
        if all(col in df for col in ["TR", "TE"]):
            summary_lines.extend(
                [
                    "Sequence Timing:",
                    f"   TR range: {df['TR'].min():.1f}-{df['TR'].max():.1f} ms",
                    f"   TE range: {df['TE'].min():.1f}-{df['TE'].max():.1f} ms",
                    "",
                ]
            )

        # Motion correction status
        if "is_motion_corrected" in df:
            moco_counts = df["is_motion_corrected"].value_counts()
            summary_lines.extend(
                [
                    "Motion Correction:",
                    f"   Original series: {moco_counts.get(False, 0)}",
                    f"   Motion corrected: {moco_counts.get(True, 0)}",
                    "",
                ]
            )

        summary_lines.extend(
            [
                "This dataset is ready for heuristic generation!",
                "All analysis performed locally using HeuDiConv.",
            ]
        )

        return "\n".join(summary_lines)
