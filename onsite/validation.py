"""
File validation module for onsite multi-file processing.

This module provides sanity checks for input/output file pairs.
"""

import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Exception raised for file validation errors."""
    pass


class FileValidator:
    """Validator for input/output file pairs."""

    # Supported file extensions
    SPECTRUM_EXTENSIONS = {'.mzml'}
    ID_EXTENSIONS = {'.idxml'}

    @staticmethod
    def validate_file_pairs(
        in_files: Tuple[str, ...],
        id_files: Tuple[str, ...]
    ) -> List[Tuple[str, str]]:
        """
        Validate and pair mzML and idXML files.

        Sanity checks:
        - Equal number of mzML and idXML files
        - All files exist and are readable
        - Correct file extensions (.mzML, .idXML)
        - Files can be opened by PyOpenMS (quick header check)

        Args:
            in_files: Tuple of input mzML file paths
            id_files: Tuple of input idXML file paths

        Returns:
            List of (mzML, idXML) file path tuples

        Raises:
            FileValidationError: If validation fails
        """
        # Check for empty inputs
        if not in_files:
            raise FileValidationError("No input mzML files provided")
        if not id_files:
            raise FileValidationError("No input idXML files provided")

        # Check equal number of files
        if len(in_files) != len(id_files):
            raise FileValidationError(
                f"Mismatched file counts: {len(in_files)} mzML files vs "
                f"{len(id_files)} idXML files. Each mzML file must have a "
                f"corresponding idXML file."
            )

        # Validate each file
        validated_pairs = []
        for i, (mzml_path, idxml_path) in enumerate(zip(in_files, id_files), 1):
            # Validate mzML file
            FileValidator._validate_spectrum_file(mzml_path, i)
            # Validate idXML file
            FileValidator._validate_id_file(idxml_path, i)
            validated_pairs.append((mzml_path, idxml_path))

        logger.info(f"Validated {len(validated_pairs)} file pair(s)")
        return validated_pairs

    @staticmethod
    def _validate_spectrum_file(file_path: str, pair_index: int) -> None:
        """
        Validate a spectrum (mzML) file.

        Args:
            file_path: Path to the mzML file
            pair_index: Index of the file pair (for error messages)

        Raises:
            FileValidationError: If validation fails
        """
        # Check file exists
        if not os.path.exists(file_path):
            raise FileValidationError(
                f"File pair {pair_index}: mzML file not found: {file_path}"
            )

        # Check file is readable
        if not os.access(file_path, os.R_OK):
            raise FileValidationError(
                f"File pair {pair_index}: mzML file not readable: {file_path}"
            )

        # Check file extension (case-insensitive)
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in FileValidator.SPECTRUM_EXTENSIONS:
            raise FileValidationError(
                f"File pair {pair_index}: Invalid spectrum file extension '{ext}'. "
                f"Expected .mzML: {file_path}"
            )

        # Quick header check with PyOpenMS
        try:
            from pyopenms import MzMLFile, MSExperiment
            exp = MSExperiment()
            # Load just metadata (fast check)
            mzml_file = MzMLFile()
            mzml_file.load(file_path, exp)
            if exp.empty():
                logger.warning(
                    f"File pair {pair_index}: mzML file appears empty: {file_path}"
                )
        except Exception as e:
            raise FileValidationError(
                f"File pair {pair_index}: Cannot read mzML file with PyOpenMS: "
                f"{file_path}. Error: {str(e)}"
            )

    @staticmethod
    def _validate_id_file(file_path: str, pair_index: int) -> None:
        """
        Validate an identification (idXML) file.

        Args:
            file_path: Path to the idXML file
            pair_index: Index of the file pair (for error messages)

        Raises:
            FileValidationError: If validation fails
        """
        # Check file exists
        if not os.path.exists(file_path):
            raise FileValidationError(
                f"File pair {pair_index}: idXML file not found: {file_path}"
            )

        # Check file is readable
        if not os.access(file_path, os.R_OK):
            raise FileValidationError(
                f"File pair {pair_index}: idXML file not readable: {file_path}"
            )

        # Check file extension (case-insensitive)
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in FileValidator.ID_EXTENSIONS:
            raise FileValidationError(
                f"File pair {pair_index}: Invalid ID file extension '{ext}'. "
                f"Expected .idXML: {file_path}"
            )

        # Quick header check with PyOpenMS
        try:
            from pyopenms import IdXMLFile, PeptideIdentificationList
            prot_ids = []
            pep_ids = PeptideIdentificationList()
            IdXMLFile().load(file_path, prot_ids, pep_ids)
            if len(pep_ids) == 0:
                logger.warning(
                    f"File pair {pair_index}: idXML file has no peptide IDs: {file_path}"
                )
        except Exception as e:
            raise FileValidationError(
                f"File pair {pair_index}: Cannot read idXML file with PyOpenMS: "
                f"{file_path}. Error: {str(e)}"
            )

    @staticmethod
    def validate_output_paths(
        out_files: Tuple[str, ...],
        in_files: Tuple[str, ...],
        count: int
    ) -> List[str]:
        """
        Generate or validate output file paths.

        If output files are not provided, generates them as <input_base>_localized.idXML.

        Args:
            out_files: Tuple of output file paths (may be empty)
            in_files: Tuple of input mzML file paths (for generating output names)
            count: Expected number of output files

        Returns:
            List of output file paths

        Raises:
            FileValidationError: If validation fails
        """
        if out_files and len(out_files) > 0:
            # User provided output files
            if len(out_files) != count:
                raise FileValidationError(
                    f"Mismatched output file count: expected {count}, "
                    f"got {len(out_files)}"
                )

            # Validate output paths are writable
            output_paths = []
            for i, out_path in enumerate(out_files, 1):
                FileValidator._validate_output_path(out_path, i)
                output_paths.append(out_path)
            return output_paths

        # Generate output paths from input files
        output_paths = []
        for i, in_file in enumerate(in_files, 1):
            # Replace extension with _localized.idXML
            base = os.path.splitext(in_file)[0]
            out_path = f"{base}_localized.idXML"
            FileValidator._validate_output_path(out_path, i)
            output_paths.append(out_path)
            logger.info(f"Generated output path for file {i}: {out_path}")

        return output_paths

    @staticmethod
    def _validate_output_path(file_path: str, file_index: int) -> None:
        """
        Validate an output file path is writable.

        Args:
            file_path: Path to the output file
            file_index: Index of the file (for error messages)

        Raises:
            FileValidationError: If validation fails
        """
        # Check parent directory exists and is writable
        parent_dir = os.path.dirname(file_path) or '.'
        if not os.path.exists(parent_dir):
            raise FileValidationError(
                f"Output file {file_index}: Parent directory does not exist: {parent_dir}"
            )

        if not os.access(parent_dir, os.W_OK):
            raise FileValidationError(
                f"Output file {file_index}: Parent directory not writable: {parent_dir}"
            )

        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext != '.idxml':
            raise FileValidationError(
                f"Output file {file_index}: Invalid output file extension '{ext}'. "
                f"Expected .idXML: {file_path}"
            )

        # Warn if file already exists
        if os.path.exists(file_path):
            logger.warning(
                f"Output file {file_index} already exists and will be overwritten: {file_path}"
            )
