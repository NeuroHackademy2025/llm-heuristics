"""Series grouping functionality for DICOM data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SeriesGrouper:
    """Class for grouping DICOM sequences based on key characteristics."""

    def __init__(self):
        """Initialize the SeriesGrouper."""
        self.grouping_variables = [
            "protocol_name",
            "series_description",
            "sequence_name",
            "dim1",
            "dim2",
            "dim3",
            "dim4",
            "TR",
            "TE",
            "is_motion_corrected",
            "image_type",
        ]

    def group_sequences(self, dicom_df: pd.DataFrame) -> pd.DataFrame:
        """
        Group DICOM sequences based on key characteristics.

        Parameters
        ----------
        dicom_df : pd.DataFrame
            DataFrame containing DICOM metadata

        Returns
        -------
        pd.DataFrame
            Grouped sequences with aggregated information
        """
        logger.info(
            "Grouping %d sequences using variables: %s",
            len(dicom_df),
            ", ".join(self.grouping_variables),
        )

        # Ensure all grouping variables exist in the DataFrame
        missing_vars = [var for var in self.grouping_variables if var not in dicom_df.columns]
        if missing_vars:
            logger.warning("Missing grouping variables: %s", missing_vars)
            # Use only available variables
            available_vars = [var for var in self.grouping_variables if var in dicom_df.columns]
        else:
            available_vars = self.grouping_variables

        # Convert numeric columns to handle potential NaN values
        numeric_cols = ["dim1", "dim2", "dim3", "dim4", "TR", "TE"]
        for col in numeric_cols:
            if col in dicom_df.columns:
                dicom_df[col] = pd.to_numeric(dicom_df[col], errors="coerce").fillna(0)

        # Handle boolean column
        if "is_motion_corrected" in dicom_df.columns:
            dicom_df["is_motion_corrected"] = dicom_df["is_motion_corrected"].fillna(False)

        # Group by available variables
        grouped = (
            dicom_df.groupby(available_vars)
            .agg(
                {
                    "series_id": ["count", "first"],
                    "series_files": ["sum", "first"],
                    "example_dcm_file": "first",
                    "dcm_dir_name": "first",
                    "subject": "first",
                    "session": "first",
                }
            )
            .reset_index()
        )

        # Flatten multi-level column names
        grouped.columns = [
            "_".join(col).strip("_") if col[1] else col[0] for col in grouped.columns
        ]

        # Rename aggregated columns for clarity
        grouped = grouped.rename(
            columns={
                "series_id_count": "series_count",
                "series_id_first": "representative_series_id",
                "series_files_sum": "total_files",
                "series_files_first": "representative_files",
            }
        )

        # Sort by series count (most common groups first)
        grouped = grouped.sort_values("series_count", ascending=False)

        logger.info("Created %d unique sequence groups", len(grouped))
        logger.info("Total sequences covered: %d", grouped["series_count"].sum())

        return grouped

    def generate_grouping_report(self, grouped_df: pd.DataFrame) -> str:
        """
        Generate a summary report of the grouping results.

        Parameters
        ----------
        grouped_df : pd.DataFrame
            DataFrame with grouped sequences

        Returns
        -------
        str
            Human-readable report of grouping results
        """
        report_lines = [
            "DICOM Sequence Grouping Report",
            "=" * 50,
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total unique groups: {len(grouped_df)}",
            f"Total sequences represented: {grouped_df['series_count'].sum()}",
            "",
            "Grouping Variables Used:",
            "- protocol_name, series_description, sequence_name",
            "- dim1, dim2, dim3, dim4, TR, TE",
            "- is_motion_corrected, image_type",
            "",
            "Top Groups by Series Count:",
            "-" * 30,
        ]

        # Show top 10 groups by series count
        top_groups = grouped_df.nlargest(10, "series_count")

        for _, row in top_groups.iterrows():
            protocol = str(row.get("protocol_name", "Unknown"))[:25]
            description = str(row.get("series_description", "Unknown"))[:25]
            count = row.get("series_count", 0)
            dims = f"{row.get('dim1', 0)}x{row.get('dim2', 0)}x{row.get('dim3', 0)}"

            report_lines.append(
                f"  {count:3d} sequences: {protocol:25} | {description:25} | {dims}"
            )

        if len(grouped_df) > 10:
            report_lines.append(f"  ... and {len(grouped_df) - 10} more groups")

        report_lines.extend(
            [
                "",
                "Next Steps:",
                "-----------",
                "1. Run 'llm-heuristics map <output_dir>' to map groups to BIDS",
                "2. Run 'llm-heuristics generate <output_dir>' to create heuristics",
                "",
                "Note: Groups are ready for BIDS mapping using LLM analysis.",
            ]
        )

        return "\n".join(report_lines)
