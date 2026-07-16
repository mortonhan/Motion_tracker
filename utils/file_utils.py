"""
File handling utilities
"""

import os
from pathlib import Path
from typing import Union, List, Optional


def get_file_name(file_path: Union[str, Path], with_extension: bool = False) -> str:
    """Get filename without directory path

    Args:
        file_path: Path to file
        with_extension: Whether to include extension, default False

    Returns:
        Filename
    """
    path = Path(file_path)
    if with_extension:
        return path.name
    else:
        return path.stem


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension (including the dot)"""
    return Path(file_path).suffix


def ensure_extension(file_path: Union[str, Path], extension: str) -> str:
    """Ensure file path has the specified extension

    Args:
        file_path: Path to file
        extension: Extension (with or without leading dot)

    Returns:
        File path with correct extension
    """
    path = Path(file_path)
    if not extension.startswith('.'):
        extension = '.' + extension

    if path.suffix != extension:
        return str(path.with_suffix(extension))
    return str(path)


def list_files(directory: Union[str, Path],
               extensions: Optional[List[str]] = None,
               recursive: bool = False) -> List[Path]:
    """List files in a directory

    Args:
        directory: Directory path
        extensions: File extensions to filter, e.g. ['.jpg', '.png']
        recursive: Whether to search subdirectories

    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if extensions is None:
        extensions = ['*']

    files = []
    for ext in extensions:
        if not ext.startswith('.') and ext != '*':
            ext = '.' + ext

        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"

        files.extend(directory.glob(pattern))

    return sorted(files)


def create_output_filename(input_path: Union[str, Path],
                          output_dir: Union[str, Path],
                          suffix: str = "",
                          extension: str = None) -> Path:
    """Create output file path from input file path

    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Filename suffix
        extension: New extension (uses input extension if not specified)

    Returns:
        Output file path
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    if suffix:
        base_name += suffix

    if extension is None:
        extension = input_path.suffix
    elif not extension.startswith('.'):
        extension = '.' + extension

    return output_dir / (base_name + extension)


def safe_filename(filename: str) -> str:
    """Create a safe filename by replacing illegal characters

    Args:
        filename: Original filename

    Returns:
        Safe filename
    """
    illegal_chars = '<>:"/\\|?*'

    safe_name = filename
    for char in illegal_chars:
        safe_name = safe_name.replace(char, '_')

    safe_name = safe_name.strip(' .')

    if not safe_name:
        safe_name = 'unnamed'

    return safe_name


def get_relative_path(file_path: Union[str, Path],
                     base_path: Union[str, Path]) -> str:
    """Get relative path with respect to base path

    Args:
        file_path: File path
        base_path: Base path

    Returns:
        Relative path (or absolute if not computable)
    """
    try:
        return str(Path(file_path).relative_to(Path(base_path)))
    except ValueError:
        return str(Path(file_path).resolve())


def copy_file_structure(src_dir: Union[str, Path],
                       dst_dir: Union[str, Path],
                       create_dirs_only: bool = True) -> None:
    """Copy directory structure

    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        create_dirs_only: If True, only create directory structure without copying files
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        return

    for item in src_dir.rglob('*'):
        if item.is_dir():
            relative_path = item.relative_to(src_dir)
            target_dir = dst_dir / relative_path
            target_dir.mkdir(parents=True, exist_ok=True)
        elif not create_dirs_only and item.is_file():
            import shutil
            relative_path = item.relative_to(src_dir)
            target_file = dst_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_file)