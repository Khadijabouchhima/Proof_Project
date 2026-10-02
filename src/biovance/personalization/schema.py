"""Input normalization: map an upstream table (any reasonable column names) to the engine's canonical
wide table: patient_id, timestamp, context, and per signal: <signal> and <signal>__q (quality)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from biovance.personalization.config import PersonalizationConfig


def _find(df_cols: dict, aliases) -> str | None:
    for a in aliases:
        if a.lower() in df_cols:
            return df_cols[a.lower()]
    return None


def normalize_observations(df: pd.DataFrame, cfg: PersonalizationConfig) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    pid = _find(cols, cfg.column_aliases["patient_id"])
    ts = _find(cols, cfg.column_aliases["timestamp"])
    if pid is None or ts is None:
        raise ValueError(
            f"input needs a patient id column {cfg.column_aliases['patient_id']} and a timestamp column "
            f"{cfg.column_aliases['timestamp']}; got {list(df.columns)}")
    out = pd.DataFrame({"patient_id": df[pid].astype(str).to_numpy(),
                        "timestamp": pd.to_datetime(df[ts]).to_numpy()})

    ctx_col = _find(cols, cfg.column_aliases["context"])
    if ctx_col is None:
        out["context"] = np.nan
    else:
        lookup = {raw.lower(): canon for canon, raws in cfg.context_map.items() for raw in raws}
        out["context"] = df[ctx_col].astype(str).str.lower().map(lookup).to_numpy()

    gq_col = _find(cols, cfg.column_aliases["signal_quality"])
    n_found = 0
    for s in cfg.signals:
        vcol = _find(cols, s.aliases)
        if vcol is None:
            continue
        n_found += 1
        out[s.name] = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
        qcol = _find(cols, s.quality_aliases) or gq_col
        out[s.name + "__q"] = (pd.to_numeric(df[qcol], errors="coerce").to_numpy(dtype=float)
                               if qcol else np.ones(len(df)))
    if n_found == 0:
        raise ValueError(f"no known signal column found; expected one of "
                         f"{[a for s in cfg.signals for a in s.aliases]}; got {list(df.columns)}")
    # Preserve upstream sample counts for pre-aggregated observations.
    for s in cfg.signals:
        count_col = f"{s.name}__n_source"
        if count_col in df.columns:
            out[count_col] = pd.to_numeric(
                df[count_col], errors="coerce"
            ).to_numpy(dtype=float)
    return out.drop_duplicates().sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def signals_present(obs: pd.DataFrame, cfg: PersonalizationConfig):
    return tuple(s for s in cfg.signals if s.name in obs.columns)
