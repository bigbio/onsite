"""
Test CLI functionality.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from onsite.onsitec import cli, main

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_idparquet(dirpath):
    """Create a minimal idparquet directory with a psms.parquet."""
    os.makedirs(dirpath, exist_ok=True)
    df = pd.DataFrame({
        "sequence": ["PEPTIDE"],
        "peptidoform": ["PEPTIDE"],
        "peptide_identification_index": [0],
        "hit_index": [0],
        "score": [1.0],
        "score_type": ["X"],
        "higher_score_better": [True],
    })
    df.to_parquet(os.path.join(dirpath, "psms.parquet"), index=False)


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ascore" in result.output
    assert "phosphors" in result.output
    assert "lucxor" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_cli_ascore_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ascore", "--help"])
    assert result.exit_code == 0
    assert "--in-file" in result.output
    assert "--id-file" in result.output
    assert "--out-file" in result.output


def test_cli_phosphors_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["phosphors", "--help"])
    assert result.exit_code == 0
    assert "--in-file" in result.output
    assert "--id-file" in result.output
    assert "--out-file" in result.output


def test_cli_lucxor_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["lucxor", "--help"])
    assert result.exit_code == 0
    assert "--input-spectrum" in result.output
    assert "--input-id" in result.output
    assert "--output" in result.output


def test_cli_ascore_missing_required_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["ascore"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_phosphors_missing_required_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["phosphors"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_lucxor_missing_required_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["lucxor"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_unknown_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["unknown"])
    assert result.exit_code != 0
    assert "No such command" in result.output
