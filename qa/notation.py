from __future__ import annotations


TISSUE_CLASS_SYMBOLS = {
    "DIL": "D",
    "Prostatic": "P",
    "Periprostatic": "E",
    "Urethral": "U",
    "Rectal": "R",
}

FAMILY_SYMBOLS = {
    "Real": "R",
    "Centroid": "C",
    "Optimal": "O",
}

CONTRAST_FAMILY_SYMBOLS = {
    "centroid_minus_real": ("C", "R"),
    "optimal_minus_real": ("O", "R"),
    "optimal_minus_centroid": ("O", "C"),
}

METRIC_SYMBOLS = {
    "DIL Global Mean BE": r"\langle \mathcal{P}_{D} \rangle",
    "DIL Global Max BE": r"\max(\mathcal{P}_{D})",
    "DIL Global Q50 BE": r"Q_{0.50}(\mathcal{P}_{D})",
    "BX to DIL centroid distance": r"d_1",
    "DIL centroid dist mean": r"d_1",
    "DIL NN dist mean": r"d_2",
    "Prostatic Global Mean BE": r"\langle \mathcal{P}_{P} \rangle",
    "Periprostatic Global Mean BE": r"\langle \mathcal{P}_{E} \rangle",
    "Urethral Global Mean BE": r"\langle \mathcal{P}_{U} \rangle",
    "Rectal Global Mean BE": r"\langle \mathcal{P}_{R} \rangle",
    "Urethra NN dist mean": r"d_{2,U}",
    "Rectum NN dist mean": r"d_{2,R}",
    "BX to DIL centroid (X)": r"\delta_{\mathrm{LR}}",
    "BX to DIL centroid (Y)": r"\delta_{\mathrm{AP}}",
    "BX to DIL centroid (Z)": r"\delta_{\mathrm{SI}}",
    "NN surface-surface distance": r"d_0",
    "BX_to_prostate_centroid_distance_norm_mean_dim": r"d_{1,P,\mathrm{norm}}",
}

METRIC_DESCRIPTIONS = {
    "DIL Global Mean BE": "mean DIL sampling probability",
    "DIL Global Max BE": "peak DIL sampling probability",
    "DIL Global Q50 BE": "median DIL sampling probability",
    "BX to DIL centroid distance": "mean distance from sampled biopsy voxels to the targeted DIL centroid",
    "DIL NN dist mean": "mean nearest-neighbour distance from sampled biopsy voxels to the targeted DIL boundary",
    "Prostatic Global Mean BE": "mean prostatic sampling probability",
    "Periprostatic Global Mean BE": "mean periprostatic sampling probability",
    "Urethral Global Mean BE": "mean urethral sampling probability",
    "Rectal Global Mean BE": "mean rectal sampling probability",
}

AXIAL_PROBABILITY_SYMBOL = r"\mathcal{P}_{i}(z)"


def metric_symbol(metric_name: str) -> str:
    return METRIC_SYMBOLS.get(metric_name, metric_name)


def metric_math(metric_name: str) -> str:
    return f"${metric_symbol(metric_name)}$"


def family_symbol(family_name: str) -> str:
    return FAMILY_SYMBOLS[family_name]


def family_display_math(family_name: str) -> str:
    return rf"{family_name} $({family_symbol(family_name)})$"


def metric_with_family_symbol(metric_name: str, family_name: str) -> str:
    return rf"{metric_symbol(metric_name)}^{{({family_symbol(family_name)})}}"


def metric_with_family_math(metric_name: str, family_name: str) -> str:
    return f"${metric_with_family_symbol(metric_name, family_name)}$"


def contrast_symbol(contrast_key: str) -> str:
    lhs, rhs = CONTRAST_FAMILY_SYMBOLS[contrast_key]
    return rf"\Delta^{{({lhs}-{rhs})}}"


def contrast_math(contrast_key: str) -> str:
    return f"${contrast_symbol(contrast_key)}$"


def contrast_metric_symbol(metric_name: str, contrast_key: str) -> str:
    return rf"{contrast_symbol(contrast_key)}\,{metric_symbol(metric_name)}"


def contrast_metric_math(metric_name: str, contrast_key: str) -> str:
    return f"${contrast_metric_symbol(metric_name, contrast_key)}$"
