import pandas as pd
import scipy.stats as stats
import pingouin as pg
import os
from scipy.stats import kruskal
from scipy.stats import wilcoxon
import numpy as np


def analyze_data(df, tissue_types, output_dir, output_filename):
    results = []

    for tissue in tissue_types:
        # Filter DataFrame for each tissue type
        filtered_df = df[(df['Simulated type'] == 'Real') & (df['Tissue class'] == tissue)]

        if len(filtered_df) > 1:
            # Perform Wilcoxon Signed-Rank Test on differences
            stat, p_value = stats.wilcoxon(filtered_df['Global Mean BE'], filtered_df['Global Mean BE optimal'])
            
            # Calculate Cohen's d for paired samples using pingouin
            cohen_d = pg.compute_effsize(filtered_df['Global Mean BE'], filtered_df['Global Mean BE optimal'], eftype='cohen', paired=True)
            
            # Calculate Common Language Effect Size using pingouin
            cles = pg.compute_effsize(filtered_df['Global Mean BE'], filtered_df['Global Mean BE optimal'], eftype='cles', paired=True)
            
            # Calculate mean difference
            mean_diff = filtered_df['Global Mean BE'].mean() - filtered_df['Global Mean BE optimal'].mean()

            results.append({
                'Tissue Type': tissue,
                'Wilcoxon Test Statistic': stat,
                'Wilcoxon P-Value': p_value,
                'Mean Difference': mean_diff,
                'Cohen\'s d': cohen_d,
                'Common Language Effect Size': cles
            })
        else:
            results.append({
                'Tissue Type': tissue,
                'Wilcoxon Test Statistic': 'N/A',
                'Wilcoxon P-Value': 'N/A',
                'Mean Difference': 'N/A',
                'Cohen\'s d': 'N/A',
                'Common Language Effect Size': 'N/A'
            })

    results_df = pd.DataFrame(results)
    full_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(full_path, index=False)

    return results_df


def kruskal_wallis_tissue_class_test(df):

    # Example data (Global Mean BE scores from your dataframe):
    dil_scores = df[df['Tissue class'] == 'DIL']['Global Mean BE']
    prostatic_scores = df[df['Tissue class'] == 'Prostatic']['Global Mean BE']
    periprostatic_scores = df[df['Tissue class'] == 'Periprostatic']['Global Mean BE']
    rectal_scores = df[df['Tissue class'] == 'Rectal']['Global Mean BE']
    urethral_scores = df[df['Tissue class'] == 'Urethral']['Global Mean BE']

    # Kruskal-Wallis Test:
    stat, pval = kruskal(dil_scores, prostatic_scores, periprostatic_scores, rectal_scores, urethral_scores)
    # Print the results
    print(f"Kruskal–Wallis H-statistic = {stat:.3f}, p-value = {pval:.3f}")


def paired_wilcoxon_signed_rank(df, tissue_types, patient_id_col='Patient ID', bx_index_col='Bx index'):
    # Create unique pairing key
    df['unique_bx_id'] = df[patient_id_col].astype(str) + '_' + df[bx_index_col].astype(str)
    
    # Pivot table to structure data for paired comparisons
    df_pivot = df.pivot_table(index='unique_bx_id',
                              columns='Tissue class',
                              values='Global Mean BE')
    
    # Initialize results storage
    results = []

    # Loop through pairs of tissue types
    for i in range(len(tissue_types)):
        for j in range(i + 1, len(tissue_types)):
            tissue_1 = tissue_types[i]
            tissue_2 = tissue_types[j]

            # Drop incomplete pairs
            paired_data = df_pivot[[tissue_1, tissue_2]].dropna()

            if len(paired_data) < 2:
                stat, pval, n_pairs = None, None, len(paired_data)
            else:
                # Wilcoxon signed-rank test
                stat, pval = wilcoxon(paired_data[tissue_1], paired_data[tissue_2])
                n_pairs = len(paired_data)

            # Append results
            results.append({
                'Tissue 1': tissue_1,
                'Tissue 2': tissue_2,
                'Wilcoxon Statistic': stat,
                'p-value': pval,
                'Number of pairs': n_pairs
            })

    # Create DataFrame of results
    results_df = pd.DataFrame(results)
    
    return results_df

def paired_effect_size_analysis(df, tissue_types, effect_sizes, patient_id_col='Patient ID', bx_index_col='Bx index', value_col='Global Mean BE', ci=0.95, n_boot=1000):
    """
    Computes pairwise effect sizes between tissue types for specified metrics using pingouin.

    Args:
        df (DataFrame): Data containing paired observations for tissue types.
        tissue_types (list): List of tissue classes to compare.
        effect_sizes (list): List of effect size metrics to compute.
                             Options: 'cohen', 'hedges', 'cles', 'mean_diff'.
        patient_id_col (str): Column name for patient identifiers.
        bx_index_col (str): Column name for biopsy indices.
        value_col (str): Column name containing values to compare.
        ci (float): Confidence interval level (default is 0.95).
        n_boot (int): Number of bootstrap samples for confidence intervals.

    Returns:
        DataFrame: Pairwise effect sizes and confidence intervals for all specified tissue comparisons and metrics.
    """
    # Create unique pairing key
    df['unique_bx_id'] = df[patient_id_col].astype(str) + '_' + df[bx_index_col].astype(str)

    # Pivot table to structure data for paired comparisons
    df_pivot = df.pivot_table(index='unique_bx_id',
                              columns='Tissue class',
                              values=value_col)

    # Initialize results storage
    results = []

    # Loop through pairs of tissue types
    for i in range(len(tissue_types)):
        for j in range(i + 1, len(tissue_types)):
            tissue_1, tissue_2 = tissue_types[i], tissue_types[j]
            paired_data = df_pivot[[tissue_1, tissue_2]].dropna()
            effect_size_results = {'Tissue 1': tissue_1, 'Tissue 2': tissue_2, 'Number of pairs': len(paired_data)}

            if len(paired_data) < 2:
                for es in effect_sizes:
                    effect_size_results[es] = None
                    effect_size_results[f'{es}_CI_lower'] = None
                    effect_size_results[f'{es}_CI_upper'] = None
            else:
                x = paired_data[tissue_1].values
                y = paired_data[tissue_2].values

                for es in effect_sizes:
                    if es == 'mean_diff':
                        diffs = x - y
                        mean_val = float(np.mean(diffs))
                        boot_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
                        ci_lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
                        ci_upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
                        effect_size_results[es] = mean_val
                        effect_size_results[f'{es}_CI_lower'] = ci_lower
                        effect_size_results[f'{es}_CI_upper'] = ci_upper
                    else:
                        es_vals = []
                        for _ in range(n_boot):
                            idx = np.random.choice(len(x), size=len(x), replace=True)
                            es_val = pg.compute_effsize(x[idx], y[idx], paired=True, eftype=es)
                            es_vals.append(es_val)
                        stat_val = pg.compute_effsize(x, y, paired=True, eftype=es)
                        ci_lower = np.percentile(es_vals, (1 - ci) / 2 * 100)
                        ci_upper = np.percentile(es_vals, (1 + ci) / 2 * 100)
                        effect_size_results[es] = stat_val
                        effect_size_results[f'{es}_CI_lower'] = ci_lower
                        effect_size_results[f'{es}_CI_upper'] = ci_upper

            results.append(effect_size_results)

    return pd.DataFrame(results)




def compute_global_tissue_scores_stats_across_all_biopsies(
    df,
    save_csv=False,
    output_dir=None,
    all_output_filename='global_tissue_scores_statistics_all_patients.csv',
    stratify_by_simulated_type=False,
    simulated_type_col='Simulated type',
    biopsy_key_cols=('Patient ID', 'Bx ID', 'Bx index'),
    stratified_filename_prefix='global_tissue_scores_statistics_simulated_type',
):
    """
    Compute boxplot summary statistics for global BE score columns for each tissue type.
    
    For each unique tissue type (in the 'Tissue class' column) and for each of the following
    global score columns:
        - Global Mean BE
        - Global Min BE
        - Global Max BE
        - Global STD BE
    
    the function computes:
        - Mean
        - Standard deviation
        - 5th, 25th, 50th, 75th, and 95th percentiles (Q05, Q25, Q50, Q75, Q95)
        - Minimum and Maximum
        - Interquartile Range (IQR = Q75 - Q25)
        - Lower fence (Q25 - 1.5 * IQR)
        - Upper fence (Q75 + 1.5 * IQR)
    
    The function returns a DataFrame with one row per tissue type for each global score.
    The DataFrame contains the following columns:
        'Tissue class', 'Feature', 'Mean', 'Std', 'Q05', 'Q25', 'Q50',
        'Q75', 'Q95', 'Min', 'Max', 'IQR', 'Lower Fence', 'Upper Fence'
    
    Parameters:
        df (pd.DataFrame): DataFrame with global BE scores and tissue type information.
        save_csv (bool): If True, save the "all patients" stats CSV.
        output_dir (str | Path | None): Directory where CSV files are written when save_csv=True.
        all_output_filename (str): Filename for the full-cohort stats CSV.
        stratify_by_simulated_type (bool): If True and save_csv=True, also save one CSV per simulated type.
        simulated_type_col (str): Column name containing simulated type labels.
        biopsy_key_cols (tuple[str, ...]): Columns used to count unique biopsies for stratified filename metadata.
        stratified_filename_prefix (str): Prefix used for stratified output filenames.

    Returns:
        pd.DataFrame: A DataFrame containing the computed boxplot statistics for all rows in df.
    """
    def _compute_stats_table(df_in):
        # Define the global score columns of interest
        global_cols = ['Global Mean BE', 'Global Min BE', 'Global Max BE', 'Global STD BE']

        stats_list = []

        # Group by tissue type
        for tissue_class, group in df_in.groupby('Tissue class'):
            for col in global_cols:
                # Exclude missing values for computation
                s = group[col].dropna()
                if s.empty:
                    continue
                mean_val = s.mean()
                std_val = s.std()
                q05 = s.quantile(0.05)
                q25 = s.quantile(0.25)
                q50 = s.quantile(0.50)
                q75 = s.quantile(0.75)
                q95 = s.quantile(0.95)
                minimum = s.min()
                maximum = s.max()
                iqr = q75 - q25
                lower_fence = q25 - 1.5 * iqr
                upper_fence = q75 + 1.5 * iqr

                stats_list.append({
                    'Tissue class': tissue_class,
                    'Feature': col,
                    'Mean': mean_val,
                    'Std': std_val,
                    'Q05': q05,
                    'Q25': q25,
                    'Q50': q50,
                    'Q75': q75,
                    'Q95': q95,
                    'Min': minimum,
                    'Max': maximum,
                    'IQR': iqr,
                    'Lower Fence': lower_fence,
                    'Upper Fence': upper_fence
                })

        return pd.DataFrame(stats_list)

    def _sanitize_for_filename(value):
        safe = ''.join(ch if str(ch).isalnum() else '_' for ch in str(value))
        safe = safe.strip('_')
        return safe if safe else 'unknown'

    stats_df = _compute_stats_table(df)

    if save_csv:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_csv=True.")

        os.makedirs(output_dir, exist_ok=True)
        stats_df.to_csv(os.path.join(output_dir, all_output_filename), index=True)

        if stratify_by_simulated_type:
            if simulated_type_col not in df.columns:
                raise KeyError(
                    f"Column '{simulated_type_col}' not found in dataframe; "
                    "cannot stratify by simulated type."
                )

            available_biopsy_key_cols = [c for c in biopsy_key_cols if c in df.columns]
            simulated_types = sorted(
                df[simulated_type_col].dropna().unique().tolist(),
                key=lambda x: str(x)
            )

            for sim_type in simulated_types:
                strat_df = df[df[simulated_type_col] == sim_type]
                strat_stats_df = _compute_stats_table(strat_df)

                if available_biopsy_key_cols:
                    n_biopsies = strat_df[available_biopsy_key_cols].drop_duplicates().shape[0]
                else:
                    n_biopsies = strat_df.shape[0]

                sim_type_for_file = _sanitize_for_filename(sim_type)
                strat_output_filename = (
                    f"{stratified_filename_prefix}_{sim_type_for_file}"
                    f"_nBiopsies_{n_biopsies}.csv"
                )
                strat_stats_df.to_csv(
                    os.path.join(output_dir, strat_output_filename),
                    index=True
                )

    return stats_df
