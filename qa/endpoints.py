from __future__ import annotations


PRIMARY_ENDPOINT_COLUMNS = [
    "DIL Global Mean BE",
]

KEY_SECONDARY_ENDPOINT_COLUMNS = [
    "DIL Global Max BE",
]

SECONDARY_QA_ENDPOINT_COLUMNS = [
    "DIL Global Q50 BE",
    "BX to DIL centroid (X)",
    "BX to DIL centroid (Y)",
    "BX to DIL centroid (Z)",
    "BX to DIL centroid distance",
    "NN surface-surface distance",
    "DIL NN dist mean",
    "DIL centroid dist mean",
    "Prostatic Global Mean BE",
    "Periprostatic Global Mean BE",
    "Urethral Global Mean BE",
    "Rectal Global Mean BE",
    "Urethra NN dist mean",
    "Rectum NN dist mean",
    "BX_to_prostate_centroid_distance_norm_mean_dim",
]

PAIR_METRIC_COLUMNS = (
    PRIMARY_ENDPOINT_COLUMNS
    + KEY_SECONDARY_ENDPOINT_COLUMNS
    + SECONDARY_QA_ENDPOINT_COLUMNS
)

RADIOMICS_PREDICTOR_COLUMNS = [
    "DIL Volume",
    "DIL Maximum 3D diameter",
    "DIL Elongation",
    "DIL Flatness",
    "Prostate Volume",
    "Prostate Elongation",
    "Prostate Flatness",
]
