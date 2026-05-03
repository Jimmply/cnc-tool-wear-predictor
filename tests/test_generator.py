"""Unit tests for CNC tool wear data generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from data_generator import FEATURE_COLS, ToolConfig, ToolWearGenerator


def test_output_shape():
    gen = ToolWearGenerator(n_tools=3, random_seed=0)
    df = gen.generate()
    assert len(df) > 0
    assert "tool_id" in df.columns
    assert "wear_state" in df.columns
    assert "rul_cycles" in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_wear_state_labels():
    gen = ToolWearGenerator(n_tools=5, random_seed=1)
    df = gen.generate()
    valid_states = {"Fresh", "Worn", "Critical"}
    assert set(df["wear_state"].unique()).issubset(valid_states)


def test_wear_index_bounds():
    gen = ToolWearGenerator(n_tools=5, random_seed=2)
    df = gen.generate()
    assert df["wear_index"].between(0, 1).all()


def test_rul_non_negative():
    gen = ToolWearGenerator(n_tools=5, random_seed=3)
    df = gen.generate()
    assert (df["rul_cycles"] >= 0).all()


def test_sensors_positive():
    gen = ToolWearGenerator(n_tools=5, random_seed=4)
    df = gen.generate()
    for col in FEATURE_COLS:
        assert (df[col] > 0).all(), f"Non-positive values in {col}"


def test_all_three_states_present():
    gen = ToolWearGenerator(n_tools=10, random_seed=5)
    df = gen.generate()
    states = set(df["wear_state"].unique())
    assert "Fresh" in states
    assert "Critical" in states


def test_tool_count():
    n = 7
    gen = ToolWearGenerator(n_tools=n, random_seed=6)
    df = gen.generate()
    assert df["tool_id"].nunique() == n


def test_custom_config():
    cfg = ToolConfig(max_cuts=200, fresh_threshold=0.25, critical_threshold=0.65)
    gen = ToolWearGenerator(n_tools=3, tool_config=cfg, random_seed=7)
    df = gen.generate()
    assert len(df) > 0


def test_reproducibility():
    df1 = ToolWearGenerator(n_tools=3, random_seed=99).generate()
    df2 = ToolWearGenerator(n_tools=3, random_seed=99).generate()
    assert df1["vibration_rms_mm_s"].equals(df2["vibration_rms_mm_s"])
