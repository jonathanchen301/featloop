"""Feature join / materialize shape tests."""

from __future__ import annotations

import pandas as pd

from featloop.features import load_raw, materialize, point_in_time_join, time_split


def test_materialize_shape():
    table = materialize()
    assert "entity_id" in table.columns
    assert "event_ts" in table.columns
    assert "cnt" in table.columns
    assert "cnt_lag_1h" in table.columns
    assert table["cnt_lag_1h"].isna().sum() == 0
    assert len(table) > 1000


def test_time_split_is_chronological():
    table = materialize()
    train, valid, holdout = time_split(table)
    assert train["event_ts"].max() <= valid["event_ts"].min()
    assert valid["event_ts"].max() <= holdout["event_ts"].min()


def test_point_in_time_join_shape():
    table = materialize()
    entities = table[["entity_id", "event_ts"]].iloc[100:120].copy()
    # nudge timestamps slightly forward so asof finds prior rows
    entities["event_ts"] = entities["event_ts"] + pd.Timedelta(minutes=1)
    joined = point_in_time_join(entities, table[["entity_id", "event_ts", "temp", "cnt"]])
    assert len(joined) == len(entities)
    assert "temp" in joined.columns
