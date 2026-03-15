# ============================================================
# AeroGuard — Data Ingestion
#
# Kaam: NGAFID dataset ko load karna
#
# Yeh module project ka pehla real data layer hai.
# Iska kaam hai raw dataset ko disk se read karna aur
# memory-safe format mein ML pipeline ko provide karna.
#
# Teen primary datasets load honge:
#
#   1. all_flights/flight_header.csv
#      → poora dataset metadata (EDA aur fleet-level analysis ke liye)
#
#   2. 2days/flight_header.csv
#      → benchmark subset metadata (paper ka official training subset)
#
#   3. all_flights/one_parq
#      → actual sensor time-series data (100M+ rows)
#
# Extra training files:
#
#   4. 2days/flight_data.pkl
#      → preprocessed numpy arrays (model training ready)
#
#   5. stats.csv (future normalization ke liye)
#
# Dask kyun use kar rahe hain?
#
# Sensor dataset ~4.3GB hai.
# Agar pandas se load karenge toh poori file RAM mein aayegi.
# Dask lazy evaluation karta hai:
#
#   - sirf metadata read hota hai
#   - data tab load hota hai jab compute() call hota hai
#
# Isse large dataset safely process kar sakte hain.
#
# Usage Example:
#
#   from src.data.ingestion import load_data
#
#   data = load_data()
#
#   data["header_full"]        → poora flight metadata
#   data["header_2days"]       → benchmark subset
#   data["sensor_data"]        → Dask DataFrame
#   data["flight_data_2days"]  → numpy arrays
#
# ============================================================

import os
import pickle
import pandas as pd
import dask.dataframe as dd
import yaml

from src.logger import logger
from src.exception import DataIngestionException


# ------------------------------------------------------------
# CONFIG LOADER
# ------------------------------------------------------------
# Yeh function config.yaml ko load karta hai.
# Config file project ke saare paths aur parameters define karti hai.
# Hardcoding avoid karne ke liye har module config se values padhta hai.
# ------------------------------------------------------------
def load_config() -> dict:

    try:
        config_path = "configs/config.yaml"

        # Check karo config file exist karti hai ya nahi
        # Agar nahi mili toh pipeline immediately fail karegi
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.yaml nahi mili: {config_path}"
            )

        # YAML ko Python dictionary mein convert karo
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug("config.yaml successfully load hua")

        return config

    except Exception as e:
        raise DataIngestionException(e, context="Loading config.yaml")


# ------------------------------------------------------------
# FULL FLIGHT HEADER LOADER
# ------------------------------------------------------------
# Yeh function poora metadata dataset load karta hai.
#
# Use cases:
#   - Exploratory Data Analysis (EDA)
#   - Cross-flight trend analysis
#   - Fleet level statistics
#
# Dataset size ~28k flights
# ------------------------------------------------------------
def load_flight_header_full(header_path: str) -> pd.DataFrame:

    try:
        logger.info(f"Full flight header load ho raha hai: {header_path}")

        # File existence check
        if not os.path.exists(header_path):
            raise FileNotFoundError(
                f"Full flight header nahi mila: {header_path}"
            )

        # Master Index ko DataFrame index bana rahe hain
        # Yeh unique flight identifier hai
        # Sensor dataset ke cluster column se match hoga
        df = pd.read_csv(header_path, index_col="Master Index")

        logger.info(f"Full flight header load hua — Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total flights: {len(df):,}")

        # Maintenance distribution check
        if "before_after" in df.columns:

            # Maintenance se pehle wali flights
            logger.info(
                f"Before maintenance: {(df['before_after']==1).sum():,} flights"
            )

            # Maintenance ke baad wali flights
            logger.info(
                f"After maintenance: {(df['before_after']==0).sum():,} flights"
            )

        return df

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading full flight_header.csv"
        )


# ------------------------------------------------------------
# BENCHMARK HEADER LOADER
# ------------------------------------------------------------
# Yeh subset dataset hai jo research paper mein use hua tha.
#
# Sirf woh flights include hoti hain jo maintenance event
# ke 2 din ke window mein aati hain.
#
# Yeh dataset model training ke liye ideal hai kyunki
# labels already curated hain.
# ------------------------------------------------------------
def load_flight_header_2days(header_path: str) -> pd.DataFrame:

    try:
        logger.info(f"2days flight header load ho raha hai: {header_path}")

        if not os.path.exists(header_path):
            raise FileNotFoundError(
                f"2days flight header nahi mila: {header_path}"
            )

        df = pd.read_csv(header_path, index_col="Master Index")

        logger.info(f"2days flight header load hua — Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total flights: {len(df):,}")

        return df

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading 2days flight_header.csv"
        )


# ------------------------------------------------------------
# SENSOR DATA LOADER (DASK)
# ------------------------------------------------------------
# Yeh sabse bada dataset hai — 100M+ rows.
#
# Isme har flight ka per-second sensor data hota hai.
#
# Typical columns:
#   - Engine sensors
#   - Fuel system sensors
#   - Electrical system sensors
#   - Flight dynamics sensors
#
# Dask lazy load karta hai:
#
#   df = dd.read_parquet(...)
#
# Is step mein sirf schema read hota hai.
# Actual data tab load hota hai jab compute() call hota hai.
# ------------------------------------------------------------
def load_flight_sensor_data(parquet_path: str) -> dd.DataFrame:

    try:
        logger.info(
            f"Flight sensor data lazy load ho raha hai: {parquet_path}"
        )

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Sensor data folder nahi mila: {parquet_path}"
            )

        # Lazy load — memory safe
        df = dd.read_parquet(parquet_path)

        logger.info(f"Sensor data lazy load hua")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Total partitions: {df.npartitions}")

        return df

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading flight sensor parquet data"
        )


# ------------------------------------------------------------
# PREPROCESSED TRAINING DATA LOADER
# ------------------------------------------------------------
# Yeh paper ka ready-made processed dataset hai.
#
# flight_data.pkl mein numpy arrays store hote hain:
#
# Example:
#   X_train
#   y_train
#   X_val
#
# Pickle format use kiya gaya hai kyunki numpy arrays
# efficiently store ho jaate hain.
# ------------------------------------------------------------
def load_2days_flight_data(pkl_path: str) -> dict:

    try:
        logger.info(f"2days flight data pkl load ho raha hai: {pkl_path}")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"flight_data.pkl nahi mila: {pkl_path}"
            )

        # Pickle binary format read kar rahe hain
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        logger.info(f"flight_data.pkl load hua — Type: {type(data)}")

        if isinstance(data, dict):
            logger.info(f"Total flights in pkl: {len(data):,}")
            logger.debug(f"Sample flight IDs: {list(data.keys())[:10]}")
        return data

    except Exception as e:
        raise DataIngestionException(
            e, context="Loading 2days flight_data.pkl"
        )


# ------------------------------------------------------------
# MAIN DATA INGESTION PIPELINE
# ------------------------------------------------------------
# Yeh AeroGuard ka main entrypoint hai data loading ke liye.
#
# Pipeline flow:
#
#   config load
#       ↓
#   full metadata load
#       ↓
#   benchmark metadata load
#       ↓
#   sensor parquet lazy load
#       ↓
#   preprocessed training data load
#
# Final output dictionary return karta hai.
# ------------------------------------------------------------
def load_data() -> dict:

    try:
        logger.info("="*55)
        logger.info("AEROGUARD DATA INGESTION SHURU")
        logger.info("="*55)

        # Step 1 — config load
        config = load_config()

        # Step 2 — full metadata
        header_full = load_flight_header_full(
            config["data"]["flight_header_full"]
        )

        # Step 3 — benchmark subset metadata
        header_2days = load_flight_header_2days(
            config["data"]["flight_header_2days"]
        )

        # Step 4 — sensor dataset (lazy)
        sensor_data = load_flight_sensor_data(
            config["data"]["raw_flight_data"]
        )

        # Step 5 — preprocessed training arrays
        flight_data_2days = load_2days_flight_data(
            config["data"]["flight_data_2days"]
        )

        logger.info("="*55)
        logger.info("DATA INGESTION COMPLETE ✓")
        logger.info("="*55)

        return {
            "header_full": header_full,
            "header_2days": header_2days,
            "sensor_data": sensor_data,
            "flight_data_2days": flight_data_2days,
            "config": config
        }

    except Exception as e:
        raise DataIngestionException(
            e, context="Main data loading pipeline"
        )