"""Command-line interface for llm-heuristics."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from llm_heuristics import __version__
from llm_heuristics.core.heuristic_generator import HeuristicGenerator
from llm_heuristics.utils.bids_integration import BIDSSchemaIntegration
from llm_heuristics.utils.logging import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--verbose", "-v", count=True, help="Increase verbosity (can be used multiple times)"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def main(verbose: int, quiet: bool) -> None:
    """LLM-Heuristics: Local DICOM analysis and heuristic generation.

    Privacy Note: All processing happens locally. No DICOM data is sent to external services.
    Models run entirely on your hardware (GPU/CPU). Your data never leaves your system.
    """
    """LLM-based DICOM header analysis and heuristic file generation for heudiconv."""

    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    setup_logging(log_level)


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for the generated heuristic",
)
@click.option(
    "--model",
    default="meta-llama/Meta-Llama-3.1-70B-Instruct",
    help="LLM model to use for heuristic generation",
)
@click.option(
    "--no-quantization", is_flag=True, help="Disable model quantization (requires more memory)"
)
@click.option("--context", help="Additional context or requirements for heuristic generation")
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum number of regeneration attempts for BIDS compliance (default: 5)",
)
def generate(
    output_dir: Path,
    output: Path | None,
    model: str,
    no_quantization: bool,
    context: str | None,
    max_iterations: int,
) -> None:
    """Generate heuristic file for heudiconv from grouped DICOM data.

    This command requires that 'group' has been run first. It reads the
    aggregated_dicominfo_groups.tsv file from the group output directory and uses
    LLM to generate the heuristic file with automatic BIDS validation.

    If the generated heuristic contains BIDS compliance issues, it will
    automatically regenerate improved versions until all paths are compliant
    or max iterations are reached. Files are saved as:
    - heuristic_v0.py (first attempt)
    - heuristic_v1.py (first regeneration)
    - heuristic_v2.py (second regeneration)
    - etc.

    OUTPUT_DIR: Directory containing group results (with aggregated_dicominfo_groups.tsv)

    Privacy Note: All LLM processing happens locally on your machine.
    No data is sent to external services or shared with third parties.
    """

    console.print(f"[bold green]Generating heuristic from grouped data:[/bold green] {output_dir}")
    console.print(
        "[dim]Privacy Note: All processing happens locally - no data leaves your system[/dim]"
    )

    # Check for required aggregated_dicominfo_mapped.tsv file
    mapped_dicominfo_path = output_dir / "aggregated_dicominfo_mapped.tsv"
    if not mapped_dicominfo_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Required file not found: {mapped_dicominfo_path}"
        )
        console.print("[yellow]Please run 'map' command first:[/yellow]")
        console.print(f"[dim]llm-heuristics map {output_dir}[/dim]")
        raise click.ClickException("Missing required aggregated_dicominfo_mapped.tsv file")

    if output is None:
        output = Path("heuristic.py")

    console.print(f"[bold blue]Model:[/bold blue] {model}")
    console.print(f"[bold blue]Using mapped data from:[/bold blue] {mapped_dicominfo_path}")

    try:
        # Initialize BIDS schema for enhanced context
        with console.status("Loading BIDS schema for enhanced generation..."):
            bids_schema = BIDSSchemaIntegration()

        # Prepare BIDS context from schema
        schema_info = bids_schema.get_schema_version_info()
        available_modalities = list(bids_schema.modalities.keys())

        # Create modalities list
        modality_list = ", ".join(available_modalities)

        bids_context = f"""
BIDS Schema Context (v{schema_info.get("version", "unknown")}):
- Available modalities: {modality_list}
- Ensure generated paths follow BIDS specification
- Use standard BIDS entities: sub-, ses-, task-, acq-, run-, echo-, etc.
"""

        # Combine user context with BIDS context
        full_context = bids_context
        if context:
            full_context += f"\n\nUser-specified context:\n{context}"

        # Initialize generator
        with console.status("Initializing LLM model..."):
            generator = HeuristicGenerator(
                model_name=model,
                use_quantization=not no_quantization,
            )

        # Iterative generation with automatic BIDS validation and regeneration
        console.print("[bold blue]Starting iterative BIDS-compliant generation...[/bold blue]")
        console.print(f"[dim]Max iterations: {max_iterations}[/dim]")

        heuristic_content, final_output_path, iterations_used = _iterative_heuristic_generation(
            generator=generator,
            mapped_dicominfo_path=mapped_dicominfo_path,
            base_output=output,
            full_context=full_context,
            bids_schema=bids_schema,
            max_iterations=max_iterations,
        )

        console.print("\n[bold green] Final heuristic generated![/bold green]")
        console.print(f"[bold blue]Final output file:[/bold blue] {final_output_path}")
        iterations_text = f"{iterations_used}/{max_iterations}"
        console.print(f"[bold blue]Iterations used:[/bold blue] {iterations_text}")

        # Final validation summary
        from llm_heuristics.utils.templates import HeuristicTemplate

        template = HeuristicTemplate()
        final_validation = template.validate_generated_heuristic_paths(heuristic_content)

        if final_validation.get("overall_valid", False):
            console.print("[bold green] Final heuristic is fully BIDS compliant![/bold green]")
        else:
            console.print("[bold yellow] Final heuristic has some remaining issues[/bold yellow]")

        valid_count = len(final_validation.get("valid_paths", []))
        invalid_count = len(final_validation.get("invalid_paths", []))
        console.print(f"[dim]BIDS-compliant paths: {valid_count}[/dim]")
        if invalid_count > 0:
            console.print(f"[yellow]Paths still needing review: {invalid_count}[/yellow]")

        # Show preview
        if len(heuristic_content) > 1000:
            preview = heuristic_content[:1000] + "\n... (truncated)"
        else:
            preview = heuristic_content

        console.print("\n[bold yellow]Preview:[/bold yellow]")
        console.print(f"[dim]{preview}[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("dicom_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--n-cpus",
    "--n_cpus",
    type=int,
    help="Number of CPU cores for parallel processing (default: CPU count + 4). "
    "Cannot be used with --slurm.",
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Generate SLURM job array script instead of running directly. "
    "Cannot be used with --n-cpus.",
)
def analyze(
    dicom_dir: Path,
    output_dir: Path,
    n_cpus: int | None,
    slurm: bool,
) -> None:
    """Analyze DICOM directory using HeuDiConv's optimized scanner.

    This command is REQUIRED before running 'generate'. It extracts DICOM metadata
    using HeuDiConv and saves results to the specified output directory:
    - .heudiconv/ folder with HeuDiConv working files
    - aggregated_dicominfo.tsv with consolidated metadata

    DICOM_DIR: Directory containing DICOM files
    OUTPUT_DIR: Directory where analysis results will be saved

    Privacy Note: All DICOM analysis happens locally using HeuDiConv.
    No data is sent to external services or shared with third parties.
    """

    # Validate mutually exclusive options
    if n_cpus is not None and slurm:
        raise click.ClickException(
            "--n-cpus and --slurm are mutually exclusive. "
            "Use --n-cpus for direct execution or --slurm for job script generation."
        )

    console.print(
        f"[bold green]Analyzing DICOM directory (via HeuDiConv):[/bold green] {dicom_dir}"
    )
    console.print("[dim]Leveraging HeuDiConv's optimized DICOM scanner[/dim]")

    try:
        # Initialize generator with optimization parameters
        generator = HeuristicGenerator(n_cpus=n_cpus, slurm=slurm)

        if slurm:
            # Generate SLURM script
            with console.status("Generating SLURM job array script..."):
                script_path = generator.dicom_extractor.extract_dicom_info(
                    dicom_dir=dicom_dir, output_dir=output_dir
                )

            console.print("[bold green] SLURM script generated![/bold green]")
            console.print(f"[bold blue]Script path:[/bold blue] {script_path}")
            console.print(f"[bold yellow]Run with:[/bold yellow] sbatch {script_path}")
            subject_sessions = generator.dicom_extractor._discover_subject_sessions(dicom_dir)
            console.print(
                f"[dim]Generated for {len(subject_sessions)} subject/session combinations[/dim]"
            )

        else:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

            # Run HeuDiConv analysis and save results to output directory
            with console.status("Running HeuDiConv DICOM analysis..."):
                dicom_df, report_content = generator.analyze_and_save_dicom_info(
                    dicom_dir=dicom_dir,
                    output_file=None,
                )

            console.print("[bold green] HeuDiConv analysis completed![/bold green]")

            # Save aggregated dicominfo.tsv to output directory
            aggregated_dicominfo_path = output_dir / "aggregated_dicominfo.tsv"
            dicom_df.to_csv(aggregated_dicominfo_path, sep="\t", index=False)
            console.print(
                f"[bold blue]Aggregated DICOM data saved to:[/bold blue] "
                f"{aggregated_dicominfo_path}"
            )
            console.print("[dim]This file will be used by the 'generate' command[/dim]")

            # Show preview of report
            lines = report_content.split("\n")
            preview_lines = lines[:20]
            if len(lines) > 20:
                preview_lines.append("... (truncated)")

            console.print("\n[bold yellow]Analysis Summary:[/bold yellow]")
            console.print("\n".join(preview_lines))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def group(output_dir: Path) -> None:
    """Group sequences from analyzed DICOM data.

    This command reads the aggregated_dicominfo.tsv file from the analyze output
    directory and creates grouped series data with BIDS mapping. It outputs:
    - aggregated_dicominfo_groups.tsv with grouped series and BIDS paths
    - grouping_report.txt with human-readable summary

    This step is required before running 'generate'. The grouped data helps users
    understand their DICOM structure and reduces complexity for heuristic generation.

    OUTPUT_DIR: Directory containing analyze results (with aggregated_dicominfo.tsv)

    Privacy Note: All processing happens locally. No data is sent to external services.
    """
    console.print(
        f"[bold green]Grouping sequences from analyze results:[/bold green] {output_dir}"
    )

    # Check for required input file or create it
    aggregated_dicominfo_path = output_dir / "aggregated_dicominfo.tsv"
    if not aggregated_dicominfo_path.exists():
        console.print(
            "[yellow]aggregated_dicominfo.tsv not found, attempting to create it...[/yellow]"
        )

        # Try to aggregate from .heudiconv subdirectories
        heudiconv_dir = output_dir / ".heudiconv"
        if heudiconv_dir.exists():
            with console.status("Aggregating dicominfo files from .heudiconv directories..."):
                aggregated_df = _aggregate_dicominfo_files(heudiconv_dir)

            if not aggregated_df.empty:
                aggregated_df.to_csv(aggregated_dicominfo_path, sep="\t", index=False)
                console.print(
                    f"[bold green]Created aggregated_dicominfo.tsv with {len(aggregated_df)} "
                    f"series[/bold green]"
                )
            else:
                console.print(
                    "[bold red]Error:[/bold red] No dicominfo files found in .heudiconv "
                    "directories!"
                )
                console.print("[yellow]Please run 'analyze' command first:[/yellow]")
                console.print(f"[dim]llm-heuristics analyze <DICOM_DIR> {output_dir}[/dim]")
                raise click.ClickException("No dicominfo files found. Run 'analyze' first.")
        else:
            console.print("[bold red]Error:[/bold red] No .heudiconv directory found!")
            console.print("[yellow]Please run 'analyze' command first:[/yellow]")
            console.print(f"[dim]llm-heuristics analyze <DICOM_DIR> {output_dir}[/dim]")
            raise click.ClickException("Missing .heudiconv directory. Run 'analyze' first.")

    console.print(f"[bold blue]Input file:[/bold blue] {aggregated_dicominfo_path}")

    try:
        # Initialize generator (only need sequences_grouper, not LLM)
        from llm_heuristics.core.sequences_grouper import SequencesGrouper

        grouper = SequencesGrouper()

        # Define output files
        grouped_output_path = output_dir / "aggregated_dicominfo_groups.tsv"
        report_output_path = output_dir / "grouping_report.txt"

        with console.status("Grouping DICOM sequences using pandas groupby..."):
            # Read the DICOM data
            dicom_df = pd.read_csv(aggregated_dicominfo_path, sep="\t")

            if dicom_df.empty:
                raise ValueError(f"No data found in {aggregated_dicominfo_path}")

            # Group sequences using simplified approach (NO BIDS mapping)
            grouped_series = grouper.group_sequences(dicom_df)

            # Generate summary report (without BIDS info)
            report = grouper.generate_grouping_report(grouped_series)

            # Save grouped results (without BIDS mapping)
            grouped_series.to_csv(grouped_output_path, sep="\t", index=False)
            report_output_path.write_text(report)

        console.print("[bold green] Series grouping completed![/bold green]")
        console.print(f"[bold blue]Grouped data saved to:[/bold blue] {grouped_output_path}")
        console.print(f"[bold blue]Grouping report saved to:[/bold blue] {report_output_path}")

        # Show summary stats
        total_series = (
            dicom_df["series_id"].nunique() if "series_id" in dicom_df.columns else len(dicom_df)
        )
        unique_groups = len(grouped_series)

        console.print("\n[bold yellow]Grouping Summary:[/bold yellow]")
        console.print(f"  Original sequences: {total_series}")
        console.print(f"  Unique groups: {unique_groups}")
        console.print("[dim]Use the 'map' command next to map groups to BIDS[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--model",
    default="meta-llama/Meta-Llama-3.1-70B-Instruct",
    help="LLM model to use for BIDS mapping",
)
@click.option(
    "--no-quantization", is_flag=True, help="Disable model quantization (requires more memory)"
)
def map_bids(output_dir: Path, model: str, no_quantization: bool) -> None:
    """Map grouped series to BIDS using LLM analysis.

    This command reads the aggregated_dicominfo_groups.tsv file from the group output
    directory and uses LLM with comprehensive BIDS schema knowledge to map each group
    to appropriate BIDS modalities, suffixes, and entities. It outputs:
    - aggregated_dicominfo_mapped.tsv with BIDS mappings
    - mapping_report.txt with human-readable summary

    This step is required before running 'generate'. The LLM provides intelligent
    BIDS mapping based on the official BIDS schema and DICOM characteristics.

    OUTPUT_DIR: Directory containing group results (with aggregated_dicominfo_groups.tsv)

    Privacy Note: All LLM processing happens locally. No data is sent to external services.
    """
    console.print(f"[bold green]Mapping series to BIDS using LLM:[/bold green] {output_dir}")

    # Check for required input file
    grouped_dicominfo_path = output_dir / "aggregated_dicominfo_groups.tsv"
    if not grouped_dicominfo_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Required file not found: {grouped_dicominfo_path}"
        )
        console.print("[dim]Run 'group' command first to generate this file[/dim]")
        raise click.ClickException("Missing required aggregated_dicominfo_groups.tsv file")

    console.print(f"[bold blue]Input file:[/bold blue] {grouped_dicominfo_path}")
    console.print(f"[bold blue]Model:[/bold blue] {model}")

    try:
        # Initialize LLM BIDS mapper
        from llm_heuristics.core.bids_mapper import LLMBIDSMapper

        with console.status("Initializing LLM model for BIDS mapping..."):
            mapper = LLMBIDSMapper(
                model_name=model,
                use_quantization=not no_quantization,
            )

        # Define output files
        mapped_output_path = output_dir / "aggregated_dicominfo_mapped.tsv"
        report_output_path = output_dir / "mapping_report.txt"

        with console.status("Mapping groups to BIDS using LLM analysis..."):
            # Read the grouped data
            grouped_df = pd.read_csv(grouped_dicominfo_path, sep="\t")

            if grouped_df.empty:
                raise ValueError(f"No data found in {grouped_dicominfo_path}")

            # Map groups to BIDS using LLM
            mapped_df = mapper.map_groups_to_bids(grouped_df)

            # Generate mapping report
            report = mapper.generate_mapping_report(mapped_df)

            # Save results
            mapped_df.to_csv(mapped_output_path, sep="\t", index=False)
            report_output_path.write_text(report)

        console.print("[bold green] BIDS mapping completed![/bold green]")
        console.print(f"[bold blue]Mapped data saved to:[/bold blue] {mapped_output_path}")
        console.print(f"[bold blue]Mapping report saved to:[/bold blue] {report_output_path}")

        # Show summary stats
        mapped_count = len(mapped_df[mapped_df["bids_path"] != "unmapped"])
        total_count = len(mapped_df)
        mapping_rate = (mapped_count / total_count) * 100 if total_count > 0 else 0

        console.print("\n[bold yellow]Mapping Summary:[/bold yellow]")
        console.print(f"  Total groups: {total_count}")
        console.print(f"  Successfully mapped: {mapped_count}")
        console.print(f"  Success rate: {mapping_rate:.1f}%")
        console.print("[dim]Use the 'generate' command next to create heuristic file[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.argument("heuristic_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_heuristic(heuristic_file: Path) -> None:
    """Validate a heuristic file using heudiconv's validation infrastructure."""

    console.print(
        f"[bold green]Validating heuristic with heudiconv:[/bold green] {heuristic_file}"
    )

    try:
        from llm_heuristics.utils.templates import HeuristicTemplate

        template = HeuristicTemplate()

        with console.status("Testing heuristic compatibility..."):
            validation_results = template.validate_heuristic_with_heudiconv(heuristic_file)

        console.print("[bold green] Heuristic validation completed![/bold green]")

        # Display results
        if validation_results.get("valid", False):
            console.print("[bold green] Heuristic is valid and heudiconv compatible![/bold green]")
        else:
            console.print("[bold yellow] Issues found with heuristic[/bold yellow]")

        if validation_results.get("heudiconv_compatible", False):
            console.print("[bold blue] Heuristic follows heudiconv standards[/bold blue]")

        if validation_results.get("errors"):
            console.print(f"\n[bold red]Errors ({len(validation_results['errors'])}):[/bold red]")
            for error in validation_results["errors"]:
                console.print(f"  ✗ {error}")

        if validation_results.get("warnings"):
            console.print(
                f"\n[bold yellow]Warnings ({len(validation_results['warnings'])}):[/bold yellow]"
            )
            for warning in validation_results["warnings"]:
                console.print(f"   {warning}")

        # Suggest next steps
        if validation_results.get("valid", False):
            console.print("\n[bold cyan]Next steps:[/bold cyan]")
            console.print("  You can now use this heuristic with heudiconv:")
            console.print(f"  heudiconv -f {heuristic_file} -d /path/to/dicom -s subject -c none")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
def bids_info() -> None:
    """Show information about the loaded BIDS schema."""

    console.print("[bold green]BIDS Schema Information[/bold green]")

    try:
        # Initialize BIDS schema integration
        with console.status("Loading BIDS schema..."):
            bids_schema = BIDSSchemaIntegration()

        # Get schema info
        schema_info = bids_schema.get_schema_version_info()

        console.print("\n[bold yellow]Schema Details:[/bold yellow]")
        for key, value in schema_info.items():
            console.print(f"  {key}: {value}")

        # Show available modalities
        console.print("\n[bold yellow]Available BIDS Modalities:[/bold yellow]")
        for modality, info in bids_schema.modalities.items():
            console.print(f"  {modality}: {info.get('description', 'No description')}")

        # Show common entities
        console.print("\n[bold yellow]Common BIDS Entities:[/bold yellow]")
        common_entities = bids_schema.get_entity_order()
        for entity in common_entities:
            if entity in bids_schema.entities:
                info = bids_schema.entities[entity]
                console.print(
                    f"  {entity} ({info.get('entity', entity)}): "
                    f"{info.get('description', 'No description')}"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.option(
    "--version",
    "-v",
    help="Specific BIDS schema version to use (e.g., 'latest', 'master', '1.8.0')",
)
def schema_versions(version: str | None) -> None:
    """Show available BIDS schema versions or details about a specific version."""

    console.print("[bold green]BIDS Schema Versions[/bold green]")

    try:
        if version:
            # Show details about specific version
            console.print(f"\n[bold yellow]Loading schema version: {version}[/bold yellow]")

            with console.status(f"Loading BIDS schema version {version}..."):
                bids_schema = BIDSSchemaIntegration(schema_version=version)

            schema_info = bids_schema.get_schema_version_info()

            console.print("\n[bold yellow]Schema Version Details:[/bold yellow]")
            for key, value in schema_info.items():
                console.print(f"  {key}: {value}")

            console.print(
                f"\n[bold yellow]Modalities ({len(bids_schema.modalities)}):[/bold yellow]"
            )
            for modality in bids_schema.modalities:
                console.print(f"  {modality}")

        else:
            # Show available versions
            bids_schema = BIDSSchemaIntegration()
            available_versions = bids_schema.get_available_schema_versions()

            console.print("\n[bold yellow]Available Schema Sources:[/bold yellow]")
            for source, versions in available_versions.items():
                if source != "note":
                    console.print(f"\n  {source.title()}:")
                    for version_id in versions:
                        console.print(f"    {version_id}")

            if "note" in available_versions:
                console.print("\n[bold blue]Usage Note:[/bold blue]")
                for note in available_versions["note"]:
                    console.print(f"  {note}")

            console.print("\n[bold cyan]Example usage:[/bold cyan]")
            console.print("  llm-heuristics schema-versions --version master  # stable")
            console.print("  llm-heuristics schema-versions --version latest  # cutting-edge")
            console.print("  llm-heuristics schema-versions --version 1.8.0   # specific version")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
def clear_schema_cache() -> None:
    """Clear cached BIDS schema files."""

    console.print("[bold green]Clearing BIDS schema cache...[/bold green]")

    try:
        bids_schema = BIDSSchemaIntegration()
        bids_schema.clear_schema_cache()
        console.print("[bold green] Schema cache cleared successfully[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


@main.command()
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Specific cache directory to clear (default: ~/.cache/llm_heuristics)",
)
def clear_cache(cache_dir: Path | None) -> None:
    """Clear DICOM analysis cache files."""
    try:
        from llm_heuristics.core.dicom_analyzer import DicomAnalyzer

        # Create analyzer instance to get cache directory
        analyzer = DicomAnalyzer(cache_dir=cache_dir)
        cache_dir_path = analyzer.cache_dir

        console.print("[bold green]Clearing DICOM analysis cache...[/bold green]")

        if cache_dir_path.exists():
            cache_files = list(cache_dir_path.glob("dicom_analysis_*.json"))

            if cache_files:
                for cache_file in cache_files:
                    cache_file.unlink()
                console.print(
                    f"[bold green] Cleared {len(cache_files)} DICOM analysis cache files "
                    f"from {cache_dir_path}[/bold green]"
                )
            else:
                console.print(
                    f"[yellow]No DICOM analysis cache files found in {cache_dir_path}[/yellow]"
                )
        else:
            console.print(f"[yellow]Cache directory {cache_dir_path} does not exist[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.ClickException(str(e)) from e


def _get_versioned_filename(base_output: Path, version: int) -> Path:
    """Generate versioned filename for iterative heuristic generation.

    Parameters
    ----------
    base_output : Path
        Original output file path
    version : int
        Version number (0 for original, 1+ for iterations)

    Returns
    -------
    Path
        Versioned filename
    """
    if version == 0:
        # Original file gets _v0 suffix
        stem = base_output.stem
        suffix = base_output.suffix
        return base_output.parent / f"{stem}_v0{suffix}"
    else:
        # Iterations get _v1, _v2, etc.
        stem = base_output.stem
        suffix = base_output.suffix
        return base_output.parent / f"{stem}_v{version}{suffix}"


def _iterative_heuristic_generation(
    generator,
    mapped_dicominfo_path: Path,
    base_output: Path,
    full_context: str,
    bids_schema,
    max_iterations: int = 10,
) -> tuple[str, Path, int]:
    """Generate heuristic with iterative BIDS validation and regeneration.

    Parameters
    ----------
    generator : HeuristicGenerator
        Initialized heuristic generator
    mapped_dicominfo_path : Path
        Path to mapped DICOM info file (aggregated_dicominfo_mapped.tsv)
    base_output : Path
        Base output filename
    full_context : str
        Full context for LLM generation
    bids_schema : BIDSSchemaIntegration
        BIDS schema instance
    max_iterations : int
        Maximum number of regeneration attempts

    Returns
    -------
    tuple[str, Path, int]
        Final heuristic content, output path, and iteration count
    """
    from llm_heuristics.utils.templates import HeuristicTemplate

    template = HeuristicTemplate()

    for iteration in range(max_iterations):
        # Determine output filename
        output_path = _get_versioned_filename(base_output, iteration)

        attempt_text = f"Generation attempt {iteration + 1}/{max_iterations}"
        console.print(f"\n[bold blue]{attempt_text}[/bold blue]")

        # Add iteration-specific context for regeneration
        iteration_context = full_context
        if iteration > 0:
            iteration_context += f"\n\nIMPORTANT: This is regeneration attempt #{iteration + 1}."
            iteration_context += "\nThe previous heuristic had BIDS compliance issues."
            iteration_context += "\nPlease fix the path templates to be fully BIDS compliant."
            iteration_context += "\nEnsure all create_key() paths follow BIDS specification."

        # Generate heuristic
        with console.status(f"Generating heuristic (attempt {iteration + 1})..."):
            heuristic_content = generator.generate_from_mapped_tsv(
                mapped_dicominfo_path=mapped_dicominfo_path,
                output_file=output_path,
                additional_context=iteration_context,
            )

        console.print(f"[green]Generated:[/green] {output_path}")

        # Validate the generated heuristic
        with console.status(f"Validating heuristic (attempt {iteration + 1})..."):
            validation_results = template.validate_generated_heuristic_paths(heuristic_content)

        # Check if validation passed
        if validation_results.get("overall_valid", False):
            success_text = f" Validation successful on attempt {iteration + 1}!"
            console.print(f"[bold green]{success_text}[/bold green]")
            return heuristic_content, output_path, iteration + 1

        # Show validation issues
        invalid_count = len(validation_results.get("invalid_paths", []))
        console.print(f"[yellow] Validation failed: {invalid_count} invalid paths[/yellow]")

        # Show first few issues for debugging
        for i, path_info in enumerate(validation_results.get("invalid_paths", [])[:2]):
            console.print(f"   {path_info['template']}")
            if path_info.get("errors") and i == 0:
                for error in path_info["errors"][:1]:
                    console.print(f"    - {error}")

        if iteration < max_iterations - 1:
            console.print("[dim]Attempting regeneration...[/dim]")
        else:
            console.print(f"[bold red]Max iterations ({max_iterations}) reached.[/bold red]")

    # If we get here, all iterations failed
    output_path = _get_versioned_filename(base_output, max_iterations - 1)
    console.print(f"[bold yellow] Returning best attempt from {output_path}[/bold yellow]")
    return heuristic_content, output_path, max_iterations


def _aggregate_dicominfo_files(heudiconv_dir: Path) -> pd.DataFrame:
    """Aggregate all dicominfo*.tsv files from heudiconv output structure.

    This function searches for dicominfo files in the heudiconv directory structure:
    - .heudiconv/{subject_id}/info/dicominfo.tsv (no sessions)
    - .heudiconv/{subject_id}/ses-{session_id}/info/dicominfo_ses-{session_id}.tsv (with sessions)

    Parameters
    ----------
    heudiconv_dir : Path
        Path to the .heudiconv directory

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with all dicominfo data
    """
    all_dataframes = []

    # Find all dicominfo*.tsv files recursively
    dicominfo_files = list(heudiconv_dir.rglob("dicominfo*.tsv"))

    if not dicominfo_files:
        return pd.DataFrame()

    for dicominfo_file in dicominfo_files:
        try:
            df = pd.read_csv(dicominfo_file, sep="\t")
            if not df.empty:
                all_dataframes.append(df)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to read {dicominfo_file}: {e}[/yellow]")
            continue

    if not all_dataframes:
        return pd.DataFrame()

    # Concatenate all dataframes
    aggregated_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove duplicates if any (based on all columns)
    aggregated_df = aggregated_df.drop_duplicates().reset_index(drop=True)

    return aggregated_df


if __name__ == "__main__":
    main()
