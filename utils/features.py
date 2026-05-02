"""Reusable feature transforms for ``compute_features`` in pipelines.

Each function returns a single ``pd.Series`` aligned to ``df.index``.
Assign with ``df["your_col"] = fn(df)``.

When ``cumulative=True``, totals are **before the current ball** (they exclude
this row's contribution). ``innings_legal_balls`` counts legal deliveries
completed before the current ball.
"""

from __future__ import annotations

import pandas as pd


def is_wicket(df: pd.DataFrame) -> pd.Series:
    return df["player_dismissed"].notna()


def _runs_series(df: pd.DataFrame) -> pd.Series:
    return df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)


def _cumsum_before_current(
    values: pd.Series,
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.Series:
    """Inclusive group cumsum minus current row → sum over **prior** balls in the group."""
    tmp = df.assign(_cum_val=values)
    inclusive = tmp.groupby(group_cols, sort=False)["_cum_val"].cumsum()
    return inclusive - tmp["_cum_val"]


def innings_runs(df: pd.DataFrame, cumulative: bool = True) -> pd.Series:
    tmp = df.assign(_total_runs=_runs_series(df))
    gcols = ["match_id", "innings"]
    if cumulative:
        return _cumsum_before_current(tmp["_total_runs"], tmp, gcols)
    return tmp.groupby(gcols, sort=False)["_total_runs"].transform("sum")


def innings_wickets(df: pd.DataFrame, cumulative: bool = True) -> pd.Series:
    tmp = df.assign(_is_wicket=is_wicket(df).astype("int8"))
    gcols = ["match_id", "innings"]
    if cumulative:
        return _cumsum_before_current(tmp["_is_wicket"], tmp, gcols)
    return tmp.groupby(gcols, sort=False)["_is_wicket"].transform("sum")


def innings_legal_balls(df: pd.DataFrame) -> pd.Series:
    """Legal deliveries completed in this innings **before** the current ball."""
    is_extra = df["wides"].notna() | df["noballs"].notna()
    legal = (~is_extra).astype("int8")
    gcols = ["match_id", "innings"]
    return _cumsum_before_current(legal, df, gcols)


def batting_team_won(df: pd.DataFrame) -> pd.Series:
    ball_runs = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    per_team = (
        df.assign(_ball_runs=ball_runs)
        .groupby(["match_id", "batting_team"], as_index=False)["_ball_runs"]
        .sum()
        .rename(columns={"_ball_runs": "team_runs"})
    )
    max_runs = per_team.groupby("match_id")["team_runs"].transform("max")
    at_max = per_team["team_runs"] == max_runs
    n_tied = per_team.groupby("match_id")["team_runs"].transform(lambda s: (s == s.max()).sum())
    per_team = per_team.assign(_win=at_max & (n_tied == 1))
    merged = df[["match_id", "batting_team"]].merge(
        per_team[["match_id", "batting_team", "_win"]],
        on=["match_id", "batting_team"],
        how="left",
    )
    s = merged["_win"].fillna(False).astype(int)
    s.index = df.index
    return s


def with_target(
    df: pd.DataFrame,
    *,
    first_innings: int = 1,
    chase_innings: int = 2,
    runs_above_first_innings: int = 1,
) -> pd.Series:
    ball_runs = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    first_totals = (
        df.assign(_ball_runs=ball_runs)
        .loc[df["innings"] == first_innings]
        .groupby("match_id")["_ball_runs"]
        .sum()
    )
    chase_value = first_totals + runs_above_first_innings
    s = pd.Series(0, index=df.index, dtype="int64")
    chase_mask = df["innings"] == chase_innings
    s.loc[chase_mask] = df.loc[chase_mask, "match_id"].map(chase_value).fillna(0).astype(int)
    return s


def batter_runs(df: pd.DataFrame, cumulative: bool = True) -> pd.Series:
    tmp = df.assign(_total_runs=_runs_series(df))
    gcols = ["match_id", "innings", "striker"]
    if cumulative:
        return _cumsum_before_current(tmp["_total_runs"], tmp, gcols)
    return tmp.groupby(gcols, sort=False)["_total_runs"].transform("sum")


def bowler_runs(df: pd.DataFrame, cumulative: bool = True) -> pd.Series:
    tmp = df.assign(_total_runs=_runs_series(df))
    gcols = ["match_id", "innings", "bowler"]
    if cumulative:
        return _cumsum_before_current(tmp["_total_runs"], tmp, gcols)
    return tmp.groupby(gcols, sort=False)["_total_runs"].transform("sum")


def bowler_wickets(df: pd.DataFrame, cumulative: bool = True) -> pd.Series:
    tmp = df.assign(_is_wicket=is_wicket(df).astype("int8"))
    gcols = ["match_id", "innings", "bowler"]
    if cumulative:
        return _cumsum_before_current(tmp["_is_wicket"], tmp, gcols)
    return tmp.groupby(gcols, sort=False)["_is_wicket"].transform("sum")
