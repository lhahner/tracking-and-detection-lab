def write_output(output_file, lines):
    """Append detection lines to the detector output file.

    Args:
        lines: MOT-format lines to append.

    Raises:
        ValueError: If the output directory does not exist.
    """
    if not output_file.parent.exists():
        raise ValueError(
            f"Output directory does not exist: {output_file.parent}")

    with open(output_file, "a", encoding="utf-8") as f:
        f.writelines(lines)
