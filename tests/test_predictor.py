"""Leakage guard: no tool's cuts may appear in both train and test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import ToolWearGenerator
from predictor import split_by_tool


def test_split_keeps_each_tool_on_one_side():
    df = ToolWearGenerator(n_tools=10, random_seed=0).generate()
    tr, te = split_by_tool(df)
    train_tools = set(df["tool_id"].iloc[tr])
    test_tools = set(df["tool_id"].iloc[te])
    assert test_tools and train_tools
    assert train_tools.isdisjoint(test_tools)
    assert len(tr) + len(te) == len(df)
