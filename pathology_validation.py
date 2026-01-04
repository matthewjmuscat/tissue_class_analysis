import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf



def build_pathology_with_spatial_radiomics_and_distances(
    pathology_df: pd.DataFrame,
    global_sum_to_one_df: pd.DataFrame,
    biopsy_basic_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
    distances_df: pd.DataFrame,
    radiomics_feature_cols: list[str] | None = None,
    pivot_all_tissues: bool = True,
) -> pd.DataFrame:
    """
    Build a single design-matrix-style dataframe for pathology validation.

    One row = one (Patient ID, Bx ID, Bx refnum, Bx index).

    Attaches to the pathology dataframe:
      - global sum-to-one features (by tissue class, optionally pivoted wide)
      - biopsy spatial features
      - radiomics for the *targeted DIL*
      - radiomics for the prostate
      - distance summaries (mean NN and mean centroid distance)
        for: targeted DIL, prostate, rectum, urethra

    Parameters
    ----------
    pathology_df
        Dataframe with pathology labels per biopsy core.
        Expected columns at minimum:
            ["Patient ID", "Bx ID", "Bx refnum", "Bx index",
             "benign", "malignant", "maybe malignant"]
    global_sum_to_one_df
        Cohort: global sum-to-one tissue results.
        Expected columns include:
            ["Patient ID", "Bx ID", "Bx refnum", "Bx index",
             "Tissue class",
             "Global Mean BE", "Global Min BE", "Global Max BE",
             "Global STD BE", "Global SEM BE",
             "Global Q05 BE", "Global Q25 BE", "Global Q50 BE",
             "Global Q75 BE", "Global Q95 BE",
             "Global CI 95 BE (lower)", "Global CI 95 BE (upper)"]
    biopsy_basic_df
        Cohort: Biopsy basic spatial features dataframe.
    radiomics_df
        Cohort: 3D radiomic features all OAR and DIL structures.
    distances_df
        Cohort: Tissue class - distances global results (multiindex columns OK).
    radiomics_feature_cols
        Optional subset of radiomic feature columns to keep.
        If None, use all columns in `radiomics_df` except the ID columns:
        ["Patient ID", "Structure ID", "Structure type", "Structure refnum"].
    pivot_all_tissues
        If True, pivot all tissue-class global BE features to wide:
        columns like "<Tissue class> Global Mean BE".
        If False, keep only the DIL row in long form and merge it.

    Returns
    -------
    pd.DataFrame
        Pathology dataframe with extra columns for:
          - global BE metrics
          - biopsy spatial features
          - DIL + prostate radiomics
          - DIL + prostate/rectum/urethra distances
          - normalized distance features (if available)
    """
    key_cols = ["Patient ID", "Bx ID", "Bx index"]

    # --- 0) Basic sanity checks ------------------------------------------------
    missing_keys_path = [c for c in key_cols if c not in pathology_df.columns]
    if missing_keys_path:
        raise KeyError(
            f"pathology_df is missing key columns: {missing_keys_path}"
        )

    missing_keys_global = [c for c in key_cols if c not in global_sum_to_one_df.columns]
    if missing_keys_global:
        raise KeyError(
            f"global_sum_to_one_df is missing key columns: {missing_keys_global}"
        )

    # --- 1) Attach global sum-to-one metrics ----------------------------------
    # Candidate BE columns (only keep those actually present)
    candidate_be_cols = [
        "Global Mean BE", "Global Min BE", "Global Max BE",
        "Global STD BE", "Global SEM BE",
        "Global Q05 BE", "Global Q25 BE", "Global Q50 BE",
        "Global Q75 BE", "Global Q95 BE",
        "Global CI 95 BE (lower)", "Global CI 95 BE (upper)",
    ]
    be_cols = [c for c in candidate_be_cols if c in global_sum_to_one_df.columns]
    if not be_cols:
        raise KeyError(
            "global_sum_to_one_df appears to be missing all expected BE columns."
        )

    global_subset = global_sum_to_one_df[key_cols + ["Tissue class"] + be_cols].copy()

    # In case of any duplicates (shouldn't happen, but safe):
    global_subset = (
        global_subset
        .groupby(key_cols + ["Tissue class"], as_index=False)[be_cols]
        .mean()
    )

    if pivot_all_tissues:
        # Wide format: one row per biopsy, columns per tissue × BE metric
        global_wide = (
            global_subset
            .set_index(key_cols + ["Tissue class"])[be_cols]
            .unstack("Tissue class")
        )
        # Columns are MultiIndex (metric, Tissue class); flatten them
        global_wide.columns = [
            f"{tissue} {metric}"
            for (metric, tissue) in global_wide.columns
        ]
        global_wide = global_wide.reset_index()

        base = pathology_df.merge(
            global_wide,
            on=key_cols,
            how="left",
            validate="1:1",
        )
    else:
        # Keep only DIL (or whatever you use as lesion tissue label)
        if "Tissue class" not in global_subset.columns:
            raise KeyError("global_sum_to_one_df must contain 'Tissue class'.")

        dil_global = (
            global_subset[global_subset["Tissue class"] == "DIL"]
            .drop(columns=["Tissue class"])
            .copy()
        )
        # Prefix BE columns to avoid collisions
        dil_global = dil_global.rename(
            columns={c: f"DIL {c}" for c in be_cols}
        )

        base = pathology_df.merge(
            dil_global,
            on=key_cols,
            how="left",
            validate="1:1",
        )

    # --- 2) Merge biopsy spatial features -------------------------------------
    for c in ["Relative DIL ID", "Relative DIL index",
              "Relative prostate ID", "Relative prostate index"]:
        if c not in biopsy_basic_df.columns:
            raise KeyError(
                f"biopsy_basic_df is missing required column: {c}"
            )

    biopsy_feature_cols = [
        "Simulated bool",
        "Simulated type",
        "Length (mm)",
        "Volume (mm3)",
        "Voxel side length (mm)",
        "Relative DIL ID",
        "Relative DIL index",
        # "BX to DIL centroid (X)",
        # "BX to DIL centroid (Y)",
        # "BX to DIL centroid (Z)",
        # "BX to DIL centroid distance",
        # "NN surface-surface distance",
        "Relative prostate ID",
        "Relative prostate index",
        "Bx position in prostate LR",
        "Bx position in prostate AP",
        "Bx position in prostate SI",
    ]

    missing_biopsy_cols = [
        c for c in biopsy_feature_cols if c not in biopsy_basic_df.columns
    ]
    if missing_biopsy_cols:
        raise KeyError(
            f"biopsy_basic_df is missing expected columns: {missing_biopsy_cols}"
        )

    biopsy_for_merge = (
        biopsy_basic_df[key_cols + biopsy_feature_cols]
        .drop_duplicates(subset=key_cols)
    )

    merged = base.merge(
        biopsy_for_merge,
        on=key_cols,
        how="left",
        validate="1:1",
    )

    # --- 3) Radiomics: targeted DIL + prostate --------------------------------
    id_cols = ["Patient ID", "Structure ID", "Structure type", "Structure refnum"]
    for c in id_cols:
        if c not in radiomics_df.columns:
            raise KeyError(
                f"radiomics_df is missing expected ID column: {c}"
            )

    if radiomics_feature_cols is None:
        base_rad_cols = [c for c in radiomics_df.columns if c not in id_cols]
    else:
        base_rad_cols = list(radiomics_feature_cols)
        missing_rad = [c for c in base_rad_cols if c not in radiomics_df.columns]
        if missing_rad:
            raise KeyError(
                f"radiomics_feature_cols contains columns not in radiomics_df: {missing_rad}"
            )

    # DIL radiomics (only the structure the biopsy was targeting)
    dil_rad = (
        radiomics_df[radiomics_df["Structure type"] == "DIL ref"]
        [["Patient ID", "Structure ID"] + base_rad_cols]
        .drop_duplicates(subset=["Patient ID", "Structure ID"])
        .copy()
    )
    dil_rad = dil_rad.rename(
        columns={c: f"DIL {c}" for c in base_rad_cols}
    )

    merged = merged.merge(
        dil_rad,
        left_on=["Patient ID", "Relative DIL ID"],
        right_on=["Patient ID", "Structure ID"],
        how="left",
        validate="m:1",
    )
    merged = merged.drop(columns=["Structure ID"], errors="ignore")

    # Prostate radiomics (Structure type == 'OAR ref' is prostate)
    prostate_rad = (
        radiomics_df[radiomics_df["Structure type"] == "OAR ref"]
        [["Patient ID", "Structure ID"] + base_rad_cols]
        .drop_duplicates(subset=["Patient ID", "Structure ID"])
        .copy()
    )
    prostate_rad = prostate_rad.rename(
        columns={c: f"Prostate {c}" for c in base_rad_cols}
    )

    merged = merged.merge(
        prostate_rad,
        left_on=["Patient ID", "Relative prostate ID"],
        right_on=["Patient ID", "Structure ID"],
        how="left",
        validate="m:1",
    )
    merged = merged.drop(columns=["Structure ID"], errors="ignore")

    # --- 4) Distances: targeted DIL + prostate + rectum + urethra -------------
    # Flatten MultiIndex columns from distances_df if needed.
    if isinstance(distances_df.columns, pd.MultiIndex):
        flat_cols = []
        for lvl0, lvl1 in distances_df.columns:
            if not lvl1 or str(lvl1).lower() == "nan":
                flat_cols.append(str(lvl0))
            else:
                flat_cols.append(f"{lvl0} {lvl1}")
        distances_flat = distances_df.copy()
        distances_flat.columns = flat_cols
    else:
        distances_flat = distances_df.copy()

    required_dist_cols = [
        "Patient ID",
        "Bx ID",
        "Bx index",
        "Relative structure ROI",
        "Relative structure type",
        "Relative structure index",
        "Struct. boundary NN dist. mean",
        "Dist. from struct. centroid mean",
    ]
    missing_dist = [c for c in required_dist_cols if c not in distances_flat.columns]
    if missing_dist:
        raise KeyError(
            f"distances_df is missing expected columns: {missing_dist}"
        )

    # Small helper to add per-structure distance summaries
    def merge_single_structure_distance(
        base_df: pd.DataFrame,
        dist_df: pd.DataFrame,
        struct_type: str,
        prefix: str,
    ) -> pd.DataFrame:
        sub = (
            dist_df[dist_df["Relative structure type"] == struct_type]
            [
                [
                    "Patient ID",
                    "Bx ID",
                    "Bx index",
                    "Struct. boundary NN dist. mean",
                    "Dist. from struct. centroid mean",
                ]
            ]
            .drop_duplicates(subset=["Patient ID", "Bx ID", "Bx index"])
            .copy()
        )

        sub = sub.rename(
            columns={
                "Struct. boundary NN dist. mean": f"{prefix} NN dist mean",
                "Dist. from struct. centroid mean": f"{prefix} centroid dist mean",
            }
        )

        return base_df.merge(
            sub,
            on=["Patient ID", "Bx ID", "Bx index"],
            how="left",
            validate="m:1",
        )

    # 4a) Targeted DIL distances (need Relative DIL index)
    dil_dist = (
        distances_flat[distances_flat["Relative structure type"] == "DIL ref"]
        [
            [
                "Patient ID",
                "Bx ID",
                "Bx index",
                "Relative structure index",
                "Struct. boundary NN dist. mean",
                "Dist. from struct. centroid mean",
            ]
        ]
        .drop_duplicates(
            subset=["Patient ID", "Bx ID", "Bx index", "Relative structure index"]
        )
        .copy()
    )

    dil_dist = dil_dist.rename(
        columns={
            "Struct. boundary NN dist. mean": "DIL NN dist mean",
            "Dist. from struct. centroid mean": "DIL centroid dist mean",
        }
    )

    merged = merged.merge(
        dil_dist,
        left_on=["Patient ID", "Bx ID", "Bx index", "Relative DIL index"],
        right_on=["Patient ID", "Bx ID", "Bx index", "Relative structure index"],
        how="left",
        validate="m:1",
    )
    merged = merged.drop(columns=["Relative structure index"], errors="ignore")

    # 4b) Prostate, rectum, urethra distances (take first row per biopsy)
    merged = merge_single_structure_distance(
        merged, distances_flat, struct_type="OAR ref",    prefix="Prostate"
    )
    merged = merge_single_structure_distance(
        merged, distances_flat, struct_type="Rectum ref", prefix="Rectum"
    )
    merged = merge_single_structure_distance(
        merged, distances_flat, struct_type="Urethra ref", prefix="Urethra"
    )

    # --- 5) Normalized distance features (use mean prostate dimension) ---------
    dim_cols = [
        "Prostate L/R dimension at centroid",
        "Prostate A/P dimension at centroid",
        "Prostate S/I dimension at centroid",
    ]

    if all(c in merged.columns for c in dim_cols):
        # Mean linear extent of the prostate at the centroid (one per biopsy)
        merged["Prostate mean dimension at centroid"] = merged[dim_cols].mean(axis=1)

        denom = merged["Prostate mean dimension at centroid"].replace(0, pd.NA)

        if "Prostate centroid dist mean" in merged.columns:
            merged["BX_to_prostate_centroid_distance_norm_mean_dim"] = (
                merged["Prostate centroid dist mean"] / denom
            )

    return merged




import numpy as np

# Optional imports for stats; handled gracefully if missing
try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover
    smf = None


def add_pathology_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean endpoint columns derived from 'benign', 'malignant',
    'maybe malignant' flags.

    Assumes one row per core with these three columns in {0, 1}.
    """
    df = df.copy()

    for col in ["benign", "malignant", "maybe malignant"]:
        if col not in df.columns:
            raise KeyError(f"Expected pathology column '{col}' in dataframe.")
        df[col] = df[col].astype(int)

    # Encode pattern as a string: "bmm" = benign/malignant/maybe
    df["pathology_pattern"] = (
        df["benign"].astype(str)
        + df["malignant"].astype(str)
        + df["maybe malignant"].astype(str)
    )

    # Pure benign: benign present, no malignant, no suspicious
    df["pure_benign"] = (
        (df["benign"] == 1)
        & (df["malignant"] == 0)
        & (df["maybe malignant"] == 0)
    )

    # Any malignant anywhere along the core
    df["malignant_present"] = df["malignant"] == 1

    # Suspicious only: suspicious present, no malignant
    df["suspicious_only"] = (
        (df["malignant"] == 0)
        & (df["maybe malignant"] == 1)
    )

    # Any concerning signal: suspicious OR malignant
    df["any_concerning"] = df["malignant_present"] | df["suspicious_only"]

    return df


def summarize_pathology_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience summary: counts of pathology patterns and endpoint tallies.
    Returns a small dataframe but also prints a short summary.
    """
    df = add_pathology_endpoints(df)

    pattern_counts = df["pathology_pattern"].value_counts().sort_index()

    pure_benign = int(df["pure_benign"].sum())
    malignant_present = int(df["malignant_present"].sum())
    suspicious_only = int(df["suspicious_only"].sum())
    any_concerning = int(df["any_concerning"].sum())

    print("Pathology pattern counts (benign/malignant/maybe as 'bmm'):")
    print(pattern_counts)
    print()
    print(f"pure_benign: {pure_benign}")
    print(f"malignant_present: {malignant_present}")
    print(f"suspicious_only: {suspicious_only}")
    print(f"any_concerning: {any_concerning}")
    print()

    out = pattern_counts.rename("count").to_frame()
    out.attrs["pure_benign"] = pure_benign
    out.attrs["malignant_present"] = malignant_present
    out.attrs["suspicious_only"] = suspicious_only
    out.attrs["any_concerning"] = any_concerning

    return out



def prepare_pathology_analysis_df(
    design_df: pd.DataFrame,
    endpoint: str,
    predictor_col: str,
    filter_simulated: bool = True,
) -> pd.DataFrame:
    """
    From the full design matrix, build a minimal dataframe for a single
    binary endpoint and a single continuous predictor.

    Output columns:
      - Patient ID, Bx ID, Bx refnum, Bx index
      - y (0/1 endpoint)
      - predictor (renamed from predictor_col)
      - Length_mm (optional; from 'Length (mm)' if present)
      - Patient_ID_str (for clustering)

    Attributes:
      - df.attrs['endpoint'], df.attrs['endpoint_label'], df.attrs['predictor_col']
    """
    df = add_pathology_endpoints(design_df)

    # Filter to real biopsies if requested
    if filter_simulated:
        if "Simulated type" in df.columns:
            df = df[df["Simulated type"] == "Real"]
        elif "Simulated bool" in df.columns:
            df = df[df["Simulated bool"] == 0]

    # Define endpoint
    if endpoint == "malignant_vs_not_malignant":
        # All cores; y=1 if any malignant present, else 0
        df["y"] = df["malignant_present"].astype(int)
        endpoint_label = "malignant present vs not malignant"

    elif endpoint == "malignant_vs_pure_benign":
        # Restrict to pure benign or malignant-present cores
        mask = df["pure_benign"] | df["malignant_present"]
        df = df[mask].copy()
        df["y"] = df["malignant_present"].astype(int)
        endpoint_label = "malignant present vs pure benign"

    elif endpoint == "any_concerning_vs_pure_benign":
        # Restrict to pure benign or any_concerning cores
        mask = df["pure_benign"] | df["any_concerning"]
        df = df[mask].copy()
        df["y"] = df["any_concerning"].astype(int)
        endpoint_label = "any concerning vs pure benign"

    else:
        raise ValueError(f"Unknown endpoint: {endpoint!r}")

    if predictor_col not in df.columns:
        raise KeyError(f"Predictor column {predictor_col!r} not found in dataframe.")

    keep_cols = [
        "Patient ID", "Bx ID", "Bx refnum", "Bx index",
        "y", predictor_col,
    ]
    if "Length (mm)" in df.columns:
        keep_cols.append("Length (mm)")

    df = df[keep_cols].copy()
    df = df.rename(columns={predictor_col: "predictor", "Length (mm)": "Length_mm"})

    # Drop rows missing y or predictor
    df = df.dropna(subset=["y", "predictor"])

    if df.empty:
        raise ValueError(
            f"Analysis dataframe is empty after filtering for endpoint={endpoint!r} "
            f"and predictor={predictor_col!r}."
        )

    df["Patient_ID_str"] = df["Patient ID"].astype(str)

    df.attrs["endpoint"] = endpoint
    df.attrs["endpoint_label"] = endpoint_label
    df.attrs["predictor_col"] = predictor_col

    return df


def run_pathology_association(
    analysis_df: pd.DataFrame,
    standardize_predictor: bool = True,
    adjust_for_length: bool = True,
    verbose: bool = True,
):
    """
    Run a minimal association analysis between the chosen predictor and
    pathology endpoint.

    Computes:
      - n_total, n_pos, n_neg
      - group medians (y=0 vs y=1)
      - Mann–Whitney U p-value
      - AUC (if sklearn is available)
      - Pearson and Spearman correlations (if scipy is available)
      - logistic regression OR per 1 SD (if statsmodels is available)

    Returns
    -------
    dict
        Dictionary with all summary statistics.
    """
    df = analysis_df.copy()

    if df.empty:
        raise ValueError("analysis_df is empty after filtering.")

    n_total = len(df)
    n_pos = int(df["y"].sum())
    n_neg = n_total - n_pos

    pred_col = "predictor"

    # Standardise predictor if requested (for logistic only)
    if standardize_predictor:
        sd = df[pred_col].std(ddof=0)
        if sd > 0:
            df["predictor_std"] = (df[pred_col] - df[pred_col].mean()) / sd
            pred_for_model = "predictor_std"
        else:
            # All values identical; fall back to raw
            pred_for_model = pred_col
        predictor_desc = f"{analysis_df.attrs.get('predictor_col', pred_col)} (1 SD)"
    else:
        pred_for_model = pred_col
        predictor_desc = analysis_df.attrs.get("predictor_col", pred_col)

    # --- Group medians & Mann–Whitney -----------------------------------------
    if stats is not None and n_pos > 0 and n_neg > 0:
        g0 = df.loc[df["y"] == 0, pred_col]
        g1 = df.loc[df["y"] == 1, pred_col]
        median0 = float(g0.median())
        median1 = float(g1.median())
        try:
            u_stat, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        except Exception:  # pragma: no cover
            u_stat = np.nan
            p_mw = np.nan
    else:
        median0 = median1 = np.nan
        u_stat = p_mw = np.nan

    # --- AUC -------------------------------------------------------------------
    if roc_auc_score is not None and n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(df["y"], df[pred_col]))
        except Exception:  # pragma: no cover
            auc = np.nan
    else:
        auc = np.nan

    # --- Correlations (Pearson = point-biserial, Spearman = rank) -------------
    pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    if stats is not None and n_pos > 0 and n_neg > 0 and df[pred_col].nunique() > 1:
        try:
            pr = stats.pearsonr(df["y"].astype(float), df[pred_col].astype(float))
            # SciPy >=1.11 returns an object with .statistic/.pvalue
            if hasattr(pr, "statistic"):
                pearson_r = float(pr.statistic)
                pearson_p = float(pr.pvalue)
            else:
                pearson_r = float(pr[0])
                pearson_p = float(pr[1])
        except Exception:  # pragma: no cover
            pass

        try:
            sr = stats.spearmanr(df["y"].astype(float), df[pred_col].astype(float))
            if hasattr(sr, "statistic"):
                spearman_r = float(sr.statistic)
                spearman_p = float(sr.pvalue)
            else:
                spearman_r = float(sr[0])
                spearman_p = float(sr[1])
        except Exception:  # pragma: no cover
            pass

    # --- Logistic with patient-clustered SEs (if available) -------------------
    or_pred = ci_low = ci_high = p_logit = np.nan
    if smf is not None and n_pos > 0 and n_neg > 0:
        # Build formula
        if adjust_for_length and "Length_mm" in df.columns:
            formula = f"y ~ {pred_for_model} + Length_mm"
        else:
            formula = f"y ~ {pred_for_model}"

        try:
            # Try cluster-robust SEs directly in fit()
            try:
                model = smf.logit(formula, data=df)
                res = model.fit(
                    disp=False,
                    maxiter=100,
                    cov_type="cluster",
                    cov_kwds={"groups": df["Patient_ID_str"]},
                )
            except TypeError:
                # Older statsmodels: no cov_type argument in fit() -> plain logit
                model = smf.logit(formula, data=df)
                res = model.fit(disp=False, maxiter=100)

            coef = res.params[pred_for_model]
            se = res.bse[pred_for_model]

            or_pred = float(np.exp(coef))
            ci_low = float(np.exp(coef - 1.96 * se))
            ci_high = float(np.exp(coef + 1.96 * se))
            p_logit = float(res.pvalues[pred_for_model])

        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"Logistic model failed: {e}")

    results = {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "endpoint": analysis_df.attrs.get("endpoint"),
        "endpoint_label": analysis_df.attrs.get("endpoint_label"),
        "predictor_desc": predictor_desc,
        "median_predictor_y0": median0,
        "median_predictor_y1": median1,
        "u_stat": u_stat,
        "p_mannwhitney": p_mw,
        "auc": auc,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "or_pred": or_pred,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_logit": p_logit,
    }

    if verbose:
        print("--------------------------------------------------")
        print("Pathology validation: minimal association analysis")
        print("--------------------------------------------------")
        print(f"Endpoint: {results['endpoint_label']}")
        print(f"Predictor: {results['predictor_desc']}")
        print(f"Total cores: {n_total}  (y=1: {n_pos}, y=0: {n_neg})")
        print()
        print("Group medians (predictor):")
        print(f"  y=0: {median0:.3f}   y=1: {median1:.3f}")
        if not np.isnan(p_mw):
            print(f"  Mann–Whitney U p-value: {p_mw:.4g}")
        else:
            print("  Mann–Whitney U not computed (stats not available or degenerate groups).")
        print()
        if not np.isnan(auc):
            print(f"AUC for predictor: {auc:.3f}")
        else:
            print("AUC not computed (sklearn not available or degenerate groups).")
        print()
        if not np.isnan(or_pred):
            print(
                f"Logistic (cluster-robust by patient): "
                f"OR per unit of '{pred_for_model}' = {or_pred:.2f} "
                f"(95% CI {ci_low:.2f}–{ci_high:.2f}, p={p_logit:.4g})"
            )
        else:
            print("Logistic model not available (statsmodels missing or model failed).")
        print()

    return results




import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm  # in addition to smf already imported


def plot_pathology_predictor(
    analysis_df: pd.DataFrame,
    use_standardized: bool = False,
    save_prefix: str | None = None,
):
    """
    Make simple plots for the pathology validation:
      1) predictor vs outcome with logistic fit
      2) ROC curve

    Parameters
    ----------
    analysis_df
        Output of prepare_pathology_analysis_df.
    use_standardized
        If True, plot x as z-score of predictor; otherwise raw units.
    save_prefix
        If not None, save figures as f"{save_prefix}_logistic.png"
        and f"{save_prefix}_roc.png".
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    df = analysis_df.copy()
    x = df["predictor"].values.astype(float)
    y = df["y"].values.astype(int)

    # Choose x for plotting
    if use_standardized:
        sd = x.std(ddof=0)
        if sd > 0:
            x_plot = (x - x.mean()) / sd
            x_label = f"{df.attrs.get('predictor_col', 'predictor')} (z-score)"
        else:
            x_plot = x
            x_label = df.attrs.get("predictor_col", "predictor")
    else:
        x_plot = x
        x_label = df.attrs.get("predictor_col", "predictor")

    # --- Logistic fit for plotting (no clustering, just to show curve) ---
    X_design = sm.add_constant(x_plot)
    logit_model = sm.Logit(y, X_design)
    logit_res = logit_model.fit(disp=False)

    x_grid = np.linspace(x_plot.min(), x_plot.max(), 200)
    X_grid = sm.add_constant(x_grid)
    y_hat = logit_res.predict(X_grid)

    # --- 1) Scatter + logistic curve ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # jitter y a bit so points don't overlap exactly on 0/1
    rng = np.random.default_rng(123)
    y_jitter = y + rng.normal(scale=0.02, size=len(y))

    ax.scatter(
        x_plot[y == 0],
        y_jitter[y == 0],
        label="Not malignant (y=0)",
        alpha=0.8,
        marker="o",
    )
    ax.scatter(
        x_plot[y == 1],
        y_jitter[y == 1],
        label="Malignant present (y=1)",
        alpha=0.8,
        marker="^",
    )
    ax.plot(x_grid, y_hat, label="Logistic fit", linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("P(malignant)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(df.attrs.get("endpoint_label", "Pathology endpoint"))
    ax.legend()

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_logistic.png", dpi=300)

    # --- 2) ROC curve ---
    fpr, tpr, _ = roc_curve(y, x)
    auc = roc_auc_score(y, x)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})", linewidth=2)
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC curve")
    ax2.legend(loc="lower right")

    fig2.tight_layout()
    if save_prefix is not None:
        fig2.savefig(f"{save_prefix}_roc.png", dpi=300)

    plt.show()



from pathlib import Path
from typing import Sequence


def scan_pathology_predictors(
    design_df: pd.DataFrame,
    predictor_cols: Sequence[str],
    endpoint: str = "malignant_vs_not_malignant",
    filter_simulated: bool = True,
    standardize_predictor: bool = True,
    adjust_for_length: bool = True,
    output_csv_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Loop over a list of predictor columns, run the minimal pathology association
    for each, and collect the results in a summary dataframe.

    Parameters
    ----------
    design_df
        Full pathology design-matrix dataframe (output of
        build_pathology_with_spatial_radiomics_and_distances).
    predictor_cols
        Iterable of column names to test as predictors.
    endpoint
        Endpoint key understood by prepare_pathology_analysis_df.
    filter_simulated
        If True, restrict to real cores.
    standardize_predictor
        If True, logistic OR is per 1 SD change.
    adjust_for_length
        If True and 'Length (mm)' is present, include it as a covariate in logistic.
    output_csv_path
        If provided, write the summary dataframe to this CSV path.
    verbose
        If True, print progress and any warnings. When scanning many predictors
        you may want to set verbose=False.

    Returns
    -------
    pd.DataFrame
        One row per predictor, with all summary statistics.
    """
    rows = []

    for col in predictor_cols:
        if col not in design_df.columns:
            if verbose:
                print(f"[scan] Predictor {col!r} not found in design matrix; skipping.")
            continue

        if verbose:
            print(f"[scan] Analyzing predictor: {col}")

        try:
            analysis_df = prepare_pathology_analysis_df(
                design_df=design_df,
                endpoint=endpoint,
                predictor_col=col,
                filter_simulated=filter_simulated,
            )
        except (KeyError, ValueError) as e:
            if verbose:
                print(f"  -> Skipping {col!r}: {e}")
            rows.append(
                {
                    "predictor_name": col,
                    "error": str(e),
                }
            )
            continue

        # If only a single class is present, association stats will mostly be NaN,
        # which is OK and informative.
        res = run_pathology_association(
            analysis_df,
            standardize_predictor=standardize_predictor,
            adjust_for_length=adjust_for_length,
            verbose=False,  # suppress per-predictor prints in bulk mode
        )
        res["predictor_name"] = col
        rows.append(res)

    summary_df = pd.DataFrame(rows)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv_path, index=False)
        if verbose:
            print(f"[scan] Wrote summary to {output_csv_path}")

    return summary_df

