"""Aqar Hub v22.1 — Configuration"""
import os
import numpy as np

SEED = 42
np.random.seed(SEED)

CSV_PATH = "egypt_real_estate_listings.csv"
MODEL_PATH = "aqar_hub_v22_model.joblib"
TARGET = "log_price"
TEST_SIZE = 0.20
N_SPLITS = 5

# ── STRICT PROPERTY TYPE WHITELIST ──
ALLOWED_TYPES = {
    "Apartment",
    "Villa",
    "Penthouse",
    "Duplex",
    "Chalet",
}

TYPE_NORMALIZATION_MAP = {
    "Studio": "Apartment",
}

UNSUPPORTED_RAW_TYPES = {
    "iVilla",
    "Twin House",
    "Townhouse",
    "Hotel Apartment",
    "Bulk Sale Unit",
    "Cabin",
    "Land",
}

# ── PSI drift monitoring thresholds (for external use) ──
#   PSI < 0.10 : stable
#   PSI 0.10-0.25 : monitor
#   PSI > 0.25 : retrain recommended
# NOTE: PSI monitoring is NOT implemented in runtime.
# These values are for external monitoring scripts only.
PSI_MONITOR = 0.10
PSI_RETRAIN = 0.25

# ── DEBUG MODE (prints extra logs, no external dependencies) ──
DEBUG_MODE = False   # Set to True for development diagnostics