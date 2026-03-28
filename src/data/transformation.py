# ============================================================
# AeroGuard — Data Transformation Pipeline
#
# Yeh file EDA notebook ke exact steps ko modular,
# production-grade code mein implement karti hai.
#
# Notebook mein jo exactly hua — wahi yahan bhi hoga.
# Koi naya logic nahi — sirf modular + logged version.
#
# PHASE 1 — Header Cleaning (Steps 1-4):
#   Step 1 : Duplicate rows drop
#   Step 2 : Hierarchy imputation from label
#   Step 3 : Flight length filter (>= 1800s)
#   Step 4 : Dtype conversions + checkpoint save
#
# PHASE 2 — Label Construction (Steps 5-7):
#   Step 5 : Same day flights drop
#   Step 6 : Extreme date_diff drop (±30 days)
#   Step 7 : RUL binary label + regression target
#
# PHASE 3 — Sensor Cleaning (Steps 8-11):
#   Step 8 : Filter sensor IDs to header_clean
#   Step 9 : Drop cluster column
#   Step 10: Sort timesteps per flight
#   Step 11: NaN treatment (ffill → bfill → zero)
#
# Usage:
#   from src.data.transformation import run_transformation
#   header_clean, sensor_clean = run_transformation()
# ============================================================

import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import yaml
from src.logger import logger
from src.exception import DataTransformationException


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config() -> dict:
    """config.yaml load karta hai."""
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.debug("config.yaml loaded")
        return config
    except Exception as e:
        raise DataTransformationException(
            e, context="Loading config.yaml in transformation"
        )


# ============================================================
# LABEL → HIERARCHY MAPPING
# Exact same map jo notebook Cell 5 mein use hua tha
# ============================================================

LABEL_HIERARCHY_MAP = {
    # BAFFLE
    'baffle crack/damage/loose/miss'                   : 'baffle',
    'baffle plug need repair/replace'                  : 'baffle',
    'baffle screw miss/loose'                          : 'baffle',
    'baffle seal loose/damage'                         : 'baffle',
    'baffle tie/tie rod loose or damage'               : 'baffle',
    'baffle rivet loose/miss/damage'                   : 'baffle',
    'baffle spring damage'                             : 'baffle',
    'baffle mount loose/damage'                        : 'baffle',
    'baffle bracket loose/damage'                      : 'baffle',
    # ENGINE
    'engine run rough'                                 : 'engine',
    'engine failure/fire/time out'                     : 'engine',
    'engine idle/rpm issue'                            : 'engine',
    'engine need repair/reinstall/clean'               : 'engine',
    'engine seal/tube/bolt loose or damage'            : 'engine',
    'engine/propeller overspeed or damage'             : 'engine',
    'engine crankcase/crankshaft/firewall near repair' : 'engine',
    'intake gasket leak/damage'                        : 'engine',
    'intake tube/bolt/seal/boot loose or damage'       : 'engine',
    'induction damage/hardware fail'                   : 'engine',
    'magneto failure'                                  : 'engine',
    'spark plug need repair/replace'                   : 'engine',
    'mixture fail/need adjust'                         : 'engine',
    'aircraft start/external issue'                    : 'engine',
    # CYLINDER
    'cylinder compression issue'                       : 'cylinder',
    'cylinder crack/fail/need part repair'             : 'cylinder',
    'cylinder/exhaust push rod/tube damage'            : 'cylinder',
    'cylinder head/exhaust gas temperature issue'      : 'cylinder',
    'cylinder exhaust valve/stuck valve issue'         : 'cylinder',
    'rocker cover leak/loose/damage'                   : 'cylinder',
    # OIL
    'oil cooler need maintenance'                      : 'oil',
    'oil leak/pressure issue'                          : 'oil',
    'oil return line issue'                            : 'oil',
    'oil dipstick/tube need repair'                    : 'oil',
    # OTHER
    'cowling miss/loose/damage'                        : 'other',
    'drain line/tube damage'                           : 'other',
    'pilot/in-flight noticed issue'                    : 'other',
}


# ============================================================
# PHASE 1 — HEADER CLEANING
# ============================================================

def clean_header(header_df: pd.DataFrame,
                 config: dict) -> pd.DataFrame:
    """
    Header cleaning — exact notebook steps 1-4.

    Step 1 : Duplicate rows drop
    Step 2 : Hierarchy imputation from label
    Step 3 : Flight length filter >= 1800s
    Step 4 : Dtype conversions

    Args:
        header_df : raw flight_header DataFrame
        config    : loaded config dict

    Returns:
        pd.DataFrame: cleaned header — 18,384 flights
    """
    try:
        logger.info("=" * 55)
        logger.info("PHASE 1 — HEADER CLEANING SHURU")
        logger.info("=" * 55)
        logger.info(f"Input shape: {header_df.shape}")

        df = header_df.copy()

        # ── Step 1: Duplicate rows drop ──────────────────────
        # Notebook Cell 4 — exact same logic
        # 6,392 fully duplicate rows hain dataset mein
        # keep='first' — pehli occurrence rakho
        logger.info("Step 1 — Duplicate rows drop")
        before = len(df)
        df = df.drop_duplicates(keep='first').copy()
        dropped = before - len(df)
        logger.info(f"  Rows before : {before:,}")
        logger.info(f"  Rows after  : {len(df):,}")
        logger.info(f"  Dropped     : {dropped:,}")

        # Validate — notebook output: 22,543 rows
        if len(df) != 22543:
            logger.warning(
                f"  ⚠️  Expected 22,543 rows after dedup, "
                f"got {len(df):,}"
            )
        else:
            logger.info("  ✅ 22,543 rows — matches notebook")

        # Duplicate distribution log (notebook Cell 4 output)
        dup_by_ba = (
            header_df[header_df.duplicated(keep=False)]
            .groupby('before_after')
            .size()
        )
        logger.debug(f"  Dupes by before_after:\n{dup_by_ba}")

        # ── Step 2: Hierarchy imputation ─────────────────────
        # Notebook Cell 5 — exact same LABEL_HIERARCHY_MAP
        # 63.43% hierarchy values NaN the — label se fill karo
        logger.info("Step 2 — Hierarchy imputation from label")

        nan_before = df['hierarchy'].isnull().sum()
        logger.info(f"  NaN before imputation: {nan_before:,}")

        df['hierarchy'] = df.apply(
            lambda row: LABEL_HIERARCHY_MAP.get(
                row['label'], 'other'
            ) if pd.isnull(row['hierarchy'])
            else row['hierarchy'],
            axis=1
        )

        nan_after = df['hierarchy'].isnull().sum()
        logger.info(f"  NaN after imputation : {nan_after:,}")

        # Log distribution (notebook Cell 5 output)
        hier_dist = df['hierarchy'].value_counts()
        for hier, cnt in hier_dist.items():
            pct = cnt / len(df) * 100
            logger.info(f"  {hier:<12}: {cnt:>6,} ({pct:.1f}%)")

        if nan_after != 0:
            logger.warning(
                f"  ⚠️  {nan_after} NaN remaining in hierarchy"
            )
        else:
            logger.info("  ✅ Zero NaN remaining — matches notebook")

        # ── Step 3: Flight length filter ─────────────────────
        # Notebook Cell 6 — >= 1800s (30 minutes)
        # Upper bound: NO cap — truncation handles long flights
        # Researcher threshold same rakha
        logger.info("Step 3 — Flight length filter (>= 1800s)")

        before_filter = len(df)
        df = df[df['flight_length'] >= 1800].copy()
        after_filter = len(df)
        removed = before_filter - after_filter

        logger.info(f"  Rows before : {before_filter:,}")
        logger.info(f"  Rows after  : {after_filter:,}")
        logger.info(f"  Removed     : {removed:,} (< 30 min)")
        logger.info(
            f"  Retention   : "
            f"{after_filter/before_filter*100:.1f}%"
        )

        # 7 flights > 5.5 hrs — log karo but keep
        # (4096s truncation handle kar legi)
        beyond_endurance = (df['flight_length'] > 19800).sum()
        if beyond_endurance > 0:
            logger.warning(
                f"  ⚠️  {beyond_endurance} flights exceed "
                f"Cessna 172 max endurance (5.5 hrs)"
            )
            logger.warning(
                "  Action: Keeping — 4096s truncation "
                "will handle these"
            )

        # Validate — notebook output: 18,384 rows
        if len(df) != 18384:
            logger.warning(
                f"  ⚠️  Expected 18,384 rows, got {len(df):,}"
            )
        else:
            logger.info("  ✅ 18,384 rows — matches notebook")

        # ── Step 4: Dtype conversions ─────────────────────────
        # Notebook Cell 7 — memory optimization
        # object → category, int64 → int32, float64 → float32
        logger.info("Step 4 — Dtype conversions")

        mem_before = df.memory_usage(deep=True).sum() / 1024
        logger.info(f"  Memory before : {mem_before:.2f} KB")

        # Categorical
        for col in ['before_after', 'label', 'hierarchy']:
            df[col] = df[col].astype('category')

        # Integer
        for col in ['date_diff', 'number_flights_before']:
            df[col] = df[col].astype('int32')

        # Float
        df['flight_length'] = df['flight_length'].astype(
            'float32'
        )

        mem_after = df.memory_usage(deep=True).sum() / 1024
        logger.info(f"  Memory after  : {mem_after:.2f} KB")
        logger.info(
            f"  Reduction     : "
            f"{(1 - mem_after/mem_before)*100:.1f}%"
        )

        # Sanity checks — notebook Cell 7 output
        logger.info(
            f"  before_after cats: "
            f"{df['before_after'].cat.categories.tolist()}"
        )
        logger.info(
            f"  hierarchy cats   : "
            f"{df['hierarchy'].cat.categories.tolist()}"
        )
        logger.info(f"  Null values      : "
                    f"{df.isnull().sum().sum()}")

        # before_after distribution post-clean
        ba_dist = df['before_after'].value_counts()
        logger.info("  before_after distribution:")
        for ba, cnt in ba_dist.items():
            pct = cnt / len(df) * 100
            logger.info(f"    {ba:<8}: {cnt:>6,} ({pct:.1f}%)")

        logger.info("=" * 55)
        logger.info("PHASE 1 COMPLETE")
        logger.info(f"Output shape: {df.shape}")
        logger.info("=" * 55)

        return df

    except Exception as e:
        raise DataTransformationException(
            e, context="Phase 1 — Header cleaning"
        )


# ============================================================
# PHASE 2 — LABEL CONSTRUCTION
# ============================================================

def construct_labels(header_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Label construction — AeroGuard ke decisions ke saath.

    Step 5 : Same day flights drop (before_after == 'same')
    Step 6 : Extreme date_diff drop (outside ±30 days)
    Step 7 : RUL binary label (date_diff <= -5 → 1)
             Regression target (days_to_maintenance)

    Args:
        header_clean: Phase 1 output (18,384 flights)

    Returns:
        pd.DataFrame: labeled header (~16,359 flights)
    """
    try:
        logger.info("=" * 55)
        logger.info("PHASE 2 — LABEL CONSTRUCTION SHURU")
        logger.info("=" * 55)
        logger.info(f"Input shape: {header_clean.shape}")

        df = header_clean.copy()

        # ── Step 5: Same day flights drop ────────────────────
        # EDA finding: same day = number_flights_before == -1
        # 100% same day flights ka nfb = -1
        # Label ambiguous — ek din maintenance wale din ki flight
        # Drop karna cleaner hai label=0 rakhne se
        logger.info("Step 5 — Same day flights drop")

        same_mask = df['before_after'].astype(str) == 'same'
        n_same = same_mask.sum()
        df = df[~same_mask].copy()

        logger.info(f"  Same day flights dropped: {n_same:,}")
        logger.info(f"  Rows remaining          : {len(df):,}")

        # ── Step 6: Extreme date_diff drop ───────────────────
        # EDA finding: beyond ±30 days = sirf 23 flights
        # Pure noise — negligible data loss
        # before < -30: 16 flights
        # after  > +30:  7 flights
        logger.info("Step 6 — Extreme date_diff drop (±30 days)")

        extreme_mask = (
            (df['date_diff'] < -30) |
            (df['date_diff'] > 30)
        )
        n_extreme = extreme_mask.sum()
        df = df[~extreme_mask].copy()

        logger.info(f"  Extreme flights dropped : {n_extreme:,}")
        logger.info(f"  Rows remaining          : {len(df):,}")

        # ── Step 7: RUL binary label ──────────────────────────
        # AeroGuard threshold: date_diff <= -2 → at-risk (1)
        # Reason: 95% before-flights capture vs paper's 70% at -2
        # date_diff >  -2 → safe (0)
        # after flights (positive date_diff) → safe (0)
        logger.info("Step 7 — RUL binary label construction")
        logger.info("  Threshold: date_diff <= -2 → label=1")

        df['label_binary'] = (
            df['date_diff'] <= -2
        ).astype('int8')

        # Regression target — sirf before flights ke liye
        # After flights = NaN (already maintained)
        df['days_to_maintenance'] = np.where(
            df['before_after'].astype(str) == 'before',
            df['date_diff'].abs(),
            np.nan
        )

        # Label distribution log
        vc = df['label_binary'].value_counts()
        total = len(df)
        logger.info("  Label distribution:")
        logger.info(
            f"    Safe (0)    : {vc.get(0,0):>7,} "
            f"({vc.get(0,0)/total*100:.1f}%)"
        )
        logger.info(
            f"    At-risk (1) : {vc.get(1,0):>7,} "
            f"({vc.get(1,0)/total*100:.1f}%)"
        )

        # before_after breakdown
        ba_dist = df['before_after'].astype(str).value_counts()
        logger.info("  before_after breakdown:")
        for ba, cnt in ba_dist.items():
            lbl1 = df[
                (df['before_after'].astype(str) == ba) &
                (df['label_binary'] == 1)
            ].shape[0]
            lbl0 = df[
                (df['before_after'].astype(str) == ba) &
                (df['label_binary'] == 0)
            ].shape[0]
            logger.info(
                f"    {ba:<8}: {cnt:>6,} flights "
                f"(label=1: {lbl1:,}, label=0: {lbl0:,})"
            )

        logger.info("=" * 55)
        logger.info("PHASE 2 COMPLETE")
        logger.info(f"Output shape: {df.shape}")
        logger.info("=" * 55)

        return df

    except Exception as e:
        raise DataTransformationException(
            e, context="Phase 2 — Label construction"
        )


# ============================================================
# PHASE 3 — SENSOR CLEANING
# ============================================================

def clean_sensor_data(
    sensor_dask: dd.DataFrame,
    header_clean: pd.DataFrame,
    config: dict
) -> dd.DataFrame:
    """
    Sensor data cleaning — exact notebook steps 8-11.

    Step 8 : Filter to header_clean Master Index IDs
    Step 9 : Drop cluster column
    Step 10: (Sort happens per-flight during extraction)
    Step 11: (NaN fill happens per-flight during extraction)

    Note: Sort + NaN fill per-flight ka kaam
    extraction step mein hoga (Phase 4) — kyunki
    Dask mein per-flight groupby sort RAM efficient nahi.

    Args:
        sensor_dask  : raw Dask DataFrame (104M+ rows)
        header_clean : cleaned header (after Phase 1)
        config       : loaded config dict

    Returns:
        dd.DataFrame: filtered + cleaned Dask DataFrame
    """
    try:
        logger.info("=" * 55)
        logger.info("PHASE 3 — SENSOR CLEANING SHURU")
        logger.info("=" * 55)

        clean_ids = set(header_clean.index.tolist())
        logger.info(
            f"Filtering to {len(clean_ids):,} clean flight IDs"
        )

        # ── Step 8: Filter sensor IDs ─────────────────────────
        # Notebook Cell — filter_partition approach
        # Sirf header_clean mein jo IDs hain unhe rakho
        # Original: 112,239,320 rows
        # After filter: 104,285,125 rows (92.9% retention)
        logger.info("Step 8 — Filtering sensor data to clean IDs")

        def filter_partition(df, ids):
            return df[df.index.isin(ids)]

        sensor_filtered = sensor_dask.map_partitions(
            filter_partition,
            ids=clean_ids
        )

        logger.info(
            "  Filter applied (lazy) — "
            "will execute on compute"
        )
        logger.info(
            f"  Columns available: "
            f"{sensor_filtered.columns.tolist()}"
        )

        # ── Step 9: Drop cluster column ───────────────────────
        # Notebook finding: cluster = maintenance label duplicate
        # Not useful as sensor feature — drop karo
        logger.info("Step 9 — Dropping cluster column")

        sensor_filtered = sensor_filtered.drop(
            columns=['cluster']
        )

        remaining_cols = sensor_filtered.columns.tolist()
        logger.info(
            f"  Columns after drop: {len(remaining_cols)}"
        )
        logger.info(f"  Remaining: {remaining_cols}")

        # ── Step 10 note ──────────────────────────────────────
        # Sort by timestep per flight — Dask mein per-flight
        # sort efficiently karna mushkil hai
        # Yeh Phase 4 (extraction) mein hoga — har flight
        # batch load karke sort + NaN fill karenge
        logger.info(
            "Step 10/11 — Sort + NaN fill: "
            "will happen during Phase 4 extraction "
            "(per-flight batch processing)"
        )

        logger.info("=" * 55)
        logger.info("PHASE 3 COMPLETE — sensor_filtered ready")
        logger.info(
            f"  Partitions : {sensor_filtered.npartitions}"
        )
        logger.info("=" * 55)

        return sensor_filtered

    except Exception as e:
        raise DataTransformationException(
            e, context="Phase 3 — Sensor cleaning"
        )


# ============================================================
# MAIN RUNNER
# ============================================================

def run_transformation(
    header_df: pd.DataFrame,
    sensor_dask: dd.DataFrame
) -> tuple[pd.DataFrame, dd.DataFrame]:
    """
    Poora transformation pipeline run karta hai.

    Args:
        header_df   : raw flight_header DataFrame
        sensor_dask : raw Dask sensor DataFrame

    Returns:
        tuple: (header_labeled, sensor_filtered)
    """
    try:
        logger.info("=" * 55)
        logger.info("AEROGUARD TRANSFORMATION PIPELINE START")
        logger.info("=" * 55)

        config = load_config()

        # Phase 1 — Header cleaning
        header_clean = clean_header(header_df, config)

        # Phase 2 — Label construction
        header_labeled = construct_labels(header_clean)

        # Phase 3 — Sensor cleaning
        sensor_filtered = clean_sensor_data(
            sensor_dask, header_clean, config
        )

        logger.info("=" * 55)
        logger.info("TRANSFORMATION PIPELINE COMPLETE ✓")
        logger.info(
            f"  Header rows    : {len(header_labeled):,}"
        )
        logger.info(
            f"  Sensor cols    : "
            f"{len(sensor_filtered.columns)}"
        )
        logger.info("=" * 55)

        return header_labeled, sensor_filtered

    except Exception as e:
        raise DataTransformationException(
            e, context="Main transformation pipeline"
        )