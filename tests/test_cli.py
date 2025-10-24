"""
Test CLI functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from onsite.onsitec import cli, main

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cli_help():
    """Test CLI help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert (
        "OnSite: Mass spectrometry post-translational modification localization tool"
        in result.output
    )
    assert "ascore" in result.output
    assert "phosphors" in result.output
    assert "lucxor" in result.output


def test_cli_version():
    """Test CLI version output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.0.1" in result.output


def test_cli_ascore_help():
    """Test AScore CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ascore", "--help"])

    assert result.exit_code == 0
    assert (
        "Phosphorylation site localization scoring tool using AScore algorithm"
        in result.output
    )
    assert "--in-file" in result.output
    assert "--id-file" in result.output
    assert "--out-file" in result.output


def test_cli_phosphors_help():
    """Test PhosphoRS CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["phosphors", "--help"])

    assert result.exit_code == 0
    assert (
        "Phosphorylation site localization scoring tool using PhosphoRS algorithm"
        in result.output
    )
    assert "--in-file" in result.output
    assert "--id-file" in result.output
    assert "--out-file" in result.output


def test_cli_lucxor_help():
    """Test LucXor CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["lucxor", "--help"])

    assert result.exit_code == 0
    assert "Modification site localization using pyLuciPHOr2 algorithm" in result.output
    assert "--input-spectrum" in result.output
    assert "--input-id" in result.output
    assert "--output" in result.output


def test_cli_ascore_missing_required_args():
    """Test AScore CLI with missing required arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ascore"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_phosphors_missing_required_args():
    """Test PhosphoRS CLI with missing required arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["phosphors"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_lucxor_missing_required_args():
    """Test LucXor CLI with missing required arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["lucxor"])

    assert result.exit_code != 0
    assert "Missing option" in result.output


@patch("onsite.onsitec.ascore")
def test_cli_ascore_execution(mock_ascore):
    """Test AScore execution through CLI."""
    runner = CliRunner()

    # Create temporary test files
    with runner.isolated_filesystem():
        # Create dummy files for testing
        with open("test.mzML", "w") as f:
            f.write("dummy mzML content")
        with open("test.idXML", "w") as f:
            f.write("dummy idXML content")

        result = runner.invoke(
            cli,
            [
                "ascore",
                "--in-file",
                "test.mzML",
                "--id-file",
                "test.idXML",
                "--out-file",
                "result.idXML",
            ],
        )

        # Should not exit with error (though ascore is mocked)
        # The command may fail due to file validation, but the CLI should handle it gracefully
        assert result.exit_code == 0, f"CLI failed: {result.output}"  # Either success or expected failure
        mock_ascore.assert_called_once()


@patch("onsite.onsitec.phosphors")
def test_cli_phosphors_execution(mock_phosphors):
    """Test PhosphoRS execution through CLI."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        with open("test.mzML", "w") as f:
            f.write("dummy mzML content")
        with open("test.idXML", "w") as f:
            f.write("dummy idXML content")

        result = runner.invoke(
            cli,
            [
                "phosphors",
                "--in-file",
                "test.mzML",
                "--id-file",
                "test.idXML",
                "--out-file",
                "result.idXML",
            ],
        )

        # Should not exit with error (though phosphors is mocked)
        # The command may fail due to file validation, but the CLI should handle it gracefully
        assert result.exit_code == 0, f"CLI failed: {result.output}"  # Either success or expected failure
        mock_phosphors.assert_called_once()


@patch("onsite.onsitec.lucxor")
def test_cli_lucxor_execution(mock_lucxor):
    """Test LucXor execution through CLI."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        with open("test.mzML", "w") as f:
            f.write("dummy mzML content")
        with open("test.idXML", "w") as f:
            f.write("dummy idXML content")

        result = runner.invoke(
            cli,
            [
                "lucxor",
                "--input-spectrum",
                "test.mzML",
                "--input-id",
                "test.idXML",
                "--output",
                "result.idXML",
            ],
        )

        # Should not exit with error (though lucxor is mocked)
        # The command may fail due to file validation, but the CLI should handle it gracefully
        assert result.exit_code == 0, f"CLI failed: {result.output}"  # Either success or expected failure
        mock_lucxor.assert_called_once()

def test_cli_unknown_command():
    """Test CLI with unknown command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["unknown"])

    assert result.exit_code != 0
    assert "No such command" in result.output



