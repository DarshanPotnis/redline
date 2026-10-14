"""The Parquet export of the mock run: schema, one row per sample, and DuckDB can query it."""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import CONTEXT_LIMITED, MockRun
from redline.export import to_parquet
from redline.export.parquet import SCHEMA
from redline.puzzles.logic_grid import GENERATOR
from redline.puzzles.logic_grid.model import GENERATOR_VERSION


@pytest.fixture(scope="module")
def parquet(mock_run: MockRun, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return to_parquet(mock_run.run_dir, tmp_path_factory.mktemp("data"))


def test_the_file_lands_under_the_run_id(parquet: Path) -> None:
    assert parquet.parts[-2:] == ("mock", "samples.parquet")


def test_schema(parquet: Path) -> None:
    schema = pq.read_schema(parquet)
    fixed = schema.names[: len(SCHEMA)]
    assert fixed == SCHEMA.names
    assert all(schema.field(name).type == SCHEMA.field(name).type for name in fixed)
    features = set(schema.names[len(SCHEMA) :])
    expected = {f"feature_{name}" for name in GENERATOR.generate(1, 0).features}
    assert features == expected
    assert all(schema.field(name).type == pa.float64() for name in features)


def test_one_row_per_sample_of_every_successful_log(parquet: Path, mock_run: MockRun) -> None:
    expected = sum(level.n for ladder in mock_run.manifest.ladders for level in ladder.levels)
    assert pq.read_metadata(parquet).num_rows == expected == 12


def test_duckdb_reads_it_back(parquet: Path) -> None:
    rows = duckdb.sql(
        f"""
        SELECT strategy, level, count_if(solved), count_if(context_limited),
               count(translation_status), any_value(ladder_stop_reason)
        FROM '{parquet}' GROUP BY ALL ORDER BY ALL
        """
    ).fetchall()
    assert rows == [
        ("direct", 1, 2, 0, 0, "early_stop"),
        ("direct", 2, 0, 1, 0, "early_stop"),
        ("direct", 3, 0, 0, 0, "early_stop"),
        ("z3_translate", 1, 2, 0, 2, "early_stop"),
        ("z3_translate", 2, 0, 0, 2, "early_stop"),
        ("z3_translate", 3, 0, 0, 2, "early_stop"),
    ]


def test_sample_rows_carry_ids_tokens_and_translations(parquet: Path) -> None:
    table = duckdb.sql(
        f"""
        SELECT sample_id, strategy, translation_exact, translation_status, input_tokens,
               output_tokens, n_calls, stop_reason, run_id, seed
        FROM '{parquet}'
        WHERE level = 1 OR (sample_id = '{CONTEXT_LIMITED}' AND strategy = 'direct')
        ORDER BY strategy, sample_id
        """
    ).fetchall()
    assert table == [
        ("logic_grid/L1/s0", "direct", None, None, 300, 120, 1, "stop", "mock", 0),
        ("logic_grid/L1/s1", "direct", None, None, 300, 120, 1, "stop", "mock", 1),
        # cut off at 100 of 3,584 tokens, then forced: cut off again below the reserve
        (CONTEXT_LIMITED, "direct", None, None, 600, 200, 2, "max_tokens", "mock", 1),
        ("logic_grid/L1/s0", "z3_translate", True, "unique", 300, 120, 1, "stop", "mock", 0),
        ("logic_grid/L1/s1", "z3_translate", True, "unique", 300, 120, 1, "stop", "mock", 1),
    ]


def test_rows_record_the_puzzle_text_version(parquet: Path) -> None:
    versions = duckdb.sql(f"SELECT DISTINCT generator_version FROM '{parquet}'").fetchall()
    assert versions == [(GENERATOR_VERSION,)]
