import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import copy
import seaborn as sns
import numpy as np
import pandas as pd 
from pathlib import Path
from statsmodels.nonparametric.kernel_regression import KernelReg
import misc_tools
import plotly.express as px
import plotting_funcs
import kaleido # imported for exporting image files, although not referenced it is required
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os


def production_plot_cohort_sum_to_one_all_biopsy_voxels_binom_est_histogram_by_tissue_class(dataframe,
                                       svg_image_width,
                                       svg_image_height,
                                       dpi,
                                       histogram_plot_name_string,
                                       output_dir,
                                       bx_sample_pts_vol_element,
                                       bin_width=0.05,
                                       bandwidth=0.1,
                                       split_by_simulated_type=False):
    
    plt.ioff()  # Turn off interactive plotting for batch figure generation
    
    # Deep copy the dataframe to prevent modifications to the original data
    df = copy.deepcopy(dataframe)

    # Create color mappings for vertical lines
    line_colors = {
        'mean': 'orange',
        'min': 'blue',
        'max': 'purple',
        'q05': 'cyan',
        'q25': 'green',
        'q50': 'red',
        'q75': 'green',
        'q95': 'cyan',
        'max density': 'magenta'
    }

    def _sanitize_for_filename(value):
        safe = ''.join(ch if str(ch).isalnum() else '_' for ch in str(value))
        safe = safe.strip('_')
        return safe if safe else 'unknown'

    def _plot_single_df(df_subset, output_path, figure_title_suffix=None):
        # Get the list of unique tissue classes
        tissue_classes = df_subset['Tissue class'].dropna().unique()
        if len(tissue_classes) == 0:
            print("Cohort sum-to-one histogram plot | No tissue classes found; skipping plot.")
            return

        # Set up the figure and subplots for each tissue class
        fig, axes = plt.subplots(
            len(tissue_classes), 1,
            figsize=(svg_image_width / dpi, svg_image_height / dpi),
            dpi=dpi,
            sharex=True
        )
        if len(tissue_classes) == 1:
            axes = [axes]

        # Increase padding between subplots
        fig.subplots_adjust(hspace=0.8)  # Adjust hspace to increase vertical padding

        for ax, tissue_class in zip(axes, tissue_classes):
            tissue_data = df_subset[df_subset['Tissue class'] == tissue_class]['Binomial estimator'].dropna()

            count = len(tissue_data)
            ax.text(-0.3, 0.85, f'Num voxels: {count}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
            ax.text(-0.3, 0.7, f'Kernel BW: {bandwidth}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
            ax.text(-0.3, 0.55, f'Bin width: {bin_width}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
            ax.text(-0.3, 0.4, f'Bx voxel volume (cmm): {bx_sample_pts_vol_element}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')

            bins = np.arange(0, 1.05, bin_width)  # Create bins from 0 to 1 with steps of 0.05

            # Plot normalized histogram with KDE
            sns.histplot(tissue_data, bins=bins, kde=False, color='skyblue', stat='density', ax=ax)

            # Calculate statistics
            mean_val = tissue_data.mean()
            min_val = tissue_data.min()
            max_val = tissue_data.max()
            quantiles = np.percentile(tissue_data, [5, 25, 50, 75, 95])

            try:
                # KDE fit for the binomial estimator values with specified bandwidth
                kde = gaussian_kde(tissue_data, bw_method=bandwidth)
                x_grid = np.linspace(0, 1, 1000)
                y_density = kde(x_grid)
                # Normalize the KDE so the area under the curve equals 1
                y_density /= np.trapz(y_density, x_grid)  # Normalize over the x_grid range

                max_density_value = x_grid[np.argmax(y_density)]

                # Overlay KDE plot
                ax.plot(x_grid, y_density, color='black', linewidth=1.5, label='KDE')

            except np.linalg.LinAlgError as e:
                # If there's a LinAlgError, it likely means all values are identical
                print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | LinAlgError: {e}")
                constant_value = tissue_data.iloc[0] if len(tissue_data) > 0 else 0
                ax.axvline(constant_value, color='black', linestyle='-', linewidth=1.5, label='All values are identical')
                max_density_value = constant_value  # Set max density to the constant value for further annotations

            except Exception as e:
                # Handle any other unexpected errors and print/log the error message
                print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | An unexpected error occurred: {e}")
                # Set a fallback for max density value or other defaults
                constant_value = tissue_data.mean() if len(tissue_data) > 0 else 0
                ax.axvline(constant_value, color='red', linestyle='-', linewidth=1.5, label='Fallback line due to error')
                max_density_value = constant_value

            # Add vertical lines for mean, min, max, quantiles, and max density
            line_positions = {
                'Mean': mean_val,
                'Min': min_val,
                'Max': max_val,
                'Q05': quantiles[0],
                'Q25': quantiles[1],
                'Q50': quantiles[2],
                'Q75': quantiles[3],
                'Q95': quantiles[4],
                'Max Density': max_density_value
            }

            # Sort line_positions by the x-values (positions of the vertical lines)
            sorted_line_positions = sorted(line_positions.items(), key=lambda item: item[1])

            # Initialize tracking variables to handle overlapping labels
            last_x_val = None
            last_label_y = 1.02  # Initial y position for text labels
            stack_count = 0  # Track count of stacked labels
            offset_x = 0  # Horizontal offset for secondary stacks

            # Iterate over the sorted line positions to add vertical lines and labels
            for label, x_val in sorted_line_positions:
                color = line_colors.get(label.lower(), 'black')
                ax.axvline(x_val, color=color, linestyle='--' if 'Q' in label else '-', label=label)

                # Check for potential overlap and adjust y-position if needed
                if last_x_val is not None and abs(x_val - last_x_val) < 0.1:
                    last_label_y += 0.15
                    stack_count += 1
                else:
                    # Reset position and stack count if no overlap
                    last_label_y = 1.02
                    stack_count = 0
                    offset_x = 0

                # Shift label to the right if stack count exceeds 3
                if stack_count > 2:
                    offset_x += 0.03  # Increment horizontal offset
                    last_label_y = 1.02  # Reset y-position for the new stack
                    stack_count = 0  # Reset stack count for the new column

                # Add text above the plot area with adjusted x and y positions
                ax.text(x_val + offset_x, last_label_y, f'{x_val:.2f}', color=color, ha='center', va='bottom',
                        fontsize=14, transform=ax.get_xaxis_transform())

                # Update last_x_val to current x_val
                last_x_val = x_val

            # Set x-axis limits to [0, 1] and enable grid lines
            ax.set_xlim(0, 1)
            ax.grid(True)
            ax.set_xticks(np.arange(0, 1.1, 0.1))  # Sets vertical grid lines every 0.1
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelsize=16)  # Adjust the number to your desired font size

            # Add title and labels with adjusted title position
            ax.set_title(f'{tissue_class}', fontsize=16, y=1, x=-0.15, ha='left')
            ax.set_ylabel('Density', fontsize=16)
            ax.tick_params(axis='y', labelsize=16)  # Adjust the number to your desired font size

        # X-axis label and figure title
        fig.text(0.5, 0.04, 'Multinomial Estimator', ha='center', fontsize=16)
        base_title = 'Cohort - Normalized Multinomial Estimator Distribution by Tissue Class For All Biopsy Voxels'
        if figure_title_suffix:
            fig.suptitle(f'{base_title} ({figure_title_suffix})', fontsize=16)
        else:
            fig.suptitle(base_title, fontsize=16)

        # Legend positioned outside the plot area with white background
        handles, labels = axes[-1].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95, 0.5), frameon=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')

        # Save the figure
        fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')

        # Close the figure to free memory
        plt.close(fig)

    if split_by_simulated_type and 'Simulated type' in df.columns:
        simulated_types = df['Simulated type'].dropna().unique()
        for sim_type in simulated_types:
            df_sim = df[df['Simulated type'] == sim_type]
            suffix = _sanitize_for_filename(sim_type)
            output_path = output_dir.joinpath(f"{histogram_plot_name_string} - simulated_type_{suffix}.svg")
            _plot_single_df(df_sim, output_path, figure_title_suffix=f"Simulated type: {sim_type}")
    else:
        output_path = output_dir.joinpath(f"{histogram_plot_name_string}.svg")
        _plot_single_df(df, output_path)





def cohort_global_scores_boxplot_by_bx_type(
    cohort_mc_sum_to_one_global_scores_dataframe,
    general_plot_name_string,
    cohort_output_figures_dir,
    statistic_label_map=None,
    plot_title="Core-level Tissue Score Distributions by Tissue Class",
    publication_style=True,
):
    """
    Plot cohort-level boxplots for core-level summary scores by tissue class.

    Parameters
    ----------
    cohort_mc_sum_to_one_global_scores_dataframe : pd.DataFrame
        Dataframe containing the global score columns and grouping columns.
    general_plot_name_string : str
        Output figure filename stem.
    cohort_output_figures_dir : Path | str
        Output directory for figure export.
    statistic_label_map : dict | None
        Optional mapping from source score columns to legend labels.
    plot_title : str
        Figure-level title.
    publication_style : bool
        If True, apply cleaner paper-oriented styling defaults.
    """
    if not isinstance(cohort_output_figures_dir, Path):
        cohort_output_figures_dir = Path(cohort_output_figures_dir)
    cohort_output_figures_dir.mkdir(parents=True, exist_ok=True)

    df = cohort_mc_sum_to_one_global_scores_dataframe.copy()

    value_vars = ["Global Min BE", "Global Mean BE", "Global Max BE", "Global STD BE"]
    default_statistic_label_map = {
        "Global Min BE": r"Core-level Min   $\min(\mathcal{P}_i)$",
        "Global Mean BE": r"Core-level Mean   $\langle \mathcal{P}_i \rangle$",
        "Global Max BE": r"Core-level Max   $\max(\mathcal{P}_i)$",
        "Global STD BE": r"Core-level SD   $\sigma(\mathcal{P}_i)$",
    }

    merged_label_map = default_statistic_label_map.copy()
    if statistic_label_map is not None:
        merged_label_map.update(statistic_label_map)

    # Melt into long format and map source column names to publication labels.
    df_melted = pd.melt(
        df,
        id_vars=["Tissue class", "Simulated type"],
        value_vars=value_vars,
        var_name="Statistic",
        value_name="Multinomial Estimator",
    )
    df_melted["Statistic Label"] = (
        df_melted["Statistic"].map(merged_label_map).fillna(df_melted["Statistic"])
    )

    hue_order = [merged_label_map.get(stat_name, stat_name) for stat_name in value_vars]
    palette = {
        hue_order[0]: "#009E73",  # green
        hue_order[1]: "#E69F00",  # orange
        hue_order[2]: "#0072B2",  # blue
        hue_order[3]: "#CC79A7",  # magenta
    }

    if publication_style:
        with sns.axes_style("whitegrid"), sns.plotting_context("paper"):
            g = sns.catplot(
                x="Tissue class",
                y="Multinomial Estimator",
                hue="Statistic Label",
                hue_order=hue_order,
                col="Simulated type",
                data=df_melted,
                kind="box",
                palette=palette,
                linewidth=1.0,
                showfliers=True,
                flierprops={
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": "white",
                    "markeredgecolor": "0.35",
                    "alpha": 0.8,
                },
                height=5.0,
                aspect=1.25,
            )
    else:
        g = sns.catplot(
            x="Tissue class",
            y="Multinomial Estimator",
            hue="Statistic Label",
            hue_order=hue_order,
            col="Simulated type",
            data=df_melted,
            kind="box",
            palette=palette,
            linewidth=1.0,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "white",
                "markeredgecolor": "0.35",
                "alpha": 0.8,
            },
            height=6.0,
            aspect=1.5,
        )

    g.set(ylim=(0, 1))
    g.set_axis_labels("Tissue Class", "Multinomial Estimator")
    g.set_titles("Simulation Type: {col_name}")

    for ax in g.axes.flat:
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(np.linspace(0, 1, 6))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    g.fig.suptitle(plot_title, y=0.99, fontsize=14 if publication_style else 12)
    g.fig.subplots_adjust(top=0.86 if publication_style else 0.9, bottom=0.20)

    if g._legend is not None:
        g._legend.set_title("Statistic")
        if publication_style:
            g._legend.set_bbox_to_anchor((1.02, 0.98))
            if hasattr(g._legend, "set_loc"):
                g._legend.set_loc("upper left")
            g._legend.get_frame().set_alpha(1.0)
            g._legend.get_frame().set_edgecolor("0.2")

    plt.tight_layout()

    svg_dose_fig_name = general_plot_name_string + ".svg"
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    g.savefig(svg_dose_fig_file_path, format="svg", bbox_inches="tight")

    plt.close(g.fig)



def plot_wilcoxon_heatmap(results_df, tissue_classes, output_dir, title='Wilcoxon Signed-Rank Test p-values', fig_name='wilcoxon_heatmap.svg'):
    """
    Plots a heatmap of p-values from pairwise Wilcoxon signed-rank tests and saves the figure.

    Args:
        results_df (DataFrame): Output from paired_wilcoxon_signed_rank() containing columns:
                                ['Tissue 1', 'Tissue 2', 'p-value'].
        tissue_classes (list): List of tissue class names tested.
        output_dir (str or Path): Directory to save the heatmap figure.
        title (str): Title for the heatmap plot.
        fig_name (str): Filename for the saved heatmap figure.

    Returns:
        None: Saves and displays a matplotlib heatmap.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an empty p-value matrix initialized to ones
    heatmap_matrix = pd.DataFrame(np.ones((len(tissue_classes), len(tissue_classes))),
                                  index=tissue_classes, columns=tissue_classes)

    # Populate matrix with p-values from results_df
    for _, row in results_df.iterrows():
        t1, t2, pval = row['Tissue 1'], row['Tissue 2'], row['p-value']
        heatmap_matrix.loc[t1, t2] = pval
        heatmap_matrix.loc[t2, t1] = pval  # Ensure symmetry

    # Mask the upper triangle and diagonal
    mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".4f", cmap='coolwarm_r',
                mask=mask, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=0, vmax=0.05)

    plt.title(title, fontsize=14)
    plt.ylabel('')
    plt.xlabel('')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / fig_name, format='svg')
    print(f"Heatmap saved to {output_dir / fig_name}")


def plot_effect_size_heatmap(results_df, tissue_classes, effect_size_key, output_dir,
                              title=None, fig_name=None, vmin=None, vmax=None,
                              axis_label_size=16, tick_fontsize=14,
                              cbar_label_fontsize=16, cbar_tick_fontsize=14):
    """
    Plots a heatmap of a specified effect size metric from the results dataframe and saves the figure.

    For directional effect sizes (e.g., 'mean_diff', 'cohen_d', 'hedges_g'),
    the value stored is computed as: tissue_1 - tissue_2. Therefore, in a fully
    populated matrix, we would expect an antisymmetric (negative symmetric) matrix.
    For symmetric metrics (e.g., 'cles'), the same value is stored in both cells.

    In order to remove rows/columns that are not useful (i.e. the top row and the last column,
    which often contain masked values or all-zero self comparisons), this function trims the matrix
    using slicing. Then, to hide the remaining values that are not part of the desired lower-triangular
    (directional) display, a strict upper triangle mask (with k=1) is applied.

    The final displayed heatmap shows only the lower portion of the trimmed matrix, free of
    the diagonal and upper-triangular cells.

    Args:
        results_df (DataFrame): Output from paired_effect_size_analysis(), including effect sizes.
        tissue_classes (list): List of tissue class names tested.
        effect_size_key (str): The effect size column to visualize (e.g., 'cohen_d', 'mean_diff').
        output_dir (str or Path): Directory to save the heatmap figure.
        title (str): Title for the heatmap plot.
        fig_name (str): Filename for the saved heatmap figure (optional, defaults to effect_size_key.svg).
        vmin (float): Minimum value for color scaling (defaults to -1).
        vmax (float): Maximum value for color scaling (defaults to 1).
        axis_label_size (int): Font size for axis labels.
        tick_fontsize (int): Font size for tick labels.
        cbar_label_fontsize (int): Font size for the colorbar label.
        cbar_tick_fontsize (int): Font size for the colorbar tick labels.

    Returns:
        None
    """
    # Define colorbar labels.
    cbar_label_dict = {
        'cohen': "Cohen's d",
        'hedges': "Hedges' g",
        'mean_diff': "Mean Difference",
        'cles': "Common Language Effect Size"
    }
    cbar_label = cbar_label_dict.get(effect_size_key, effect_size_key)

    # Prepare the output directory.
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fig_name is None:
        fig_name = f"{effect_size_key}_heatmap.svg"
    if title is None:
        title = f"Effect Size Heatmap ({cbar_label})"
    else:
        title = f"{title} ({cbar_label})"
    if vmin is None:
        vmin = -1
    if vmax is None:
        vmax = 1

    # Initialize the full square matrix.
    matrix = pd.DataFrame(np.nan, index=tissue_classes, columns=tissue_classes)

    # Fill the matrix:
    # - For directional effect sizes, use antisymmetry:
    #       matrix[tissue1, tissue2] = value and matrix[tissue2, tissue1] = -value.
    # - For symmetric metrics (like 'cles'), copy the value to both cells.
    for _, row in results_df.iterrows():
        t1, t2, value = row['Tissue 1'], row['Tissue 2'], row.get(effect_size_key, np.nan)
        if pd.isna(value):
            continue
        if t1 == t2:
            continue  # skip self comparisons
        if effect_size_key in ['mean_diff', 'cohen_d', 'hedges_g']:
            matrix.loc[t1, t2] = value
            matrix.loc[t2, t1] = -value
        else:
            matrix.loc[t1, t2] = value
            matrix.loc[t2, t1] = value

    # (Optional) You could set the diagonal to zero in the full matrix:
    for t in tissue_classes:
        matrix.loc[t, t] = 0

    # Trim the matrix: remove the top row and the last column
    # This removes the first tissue from the rows and the last tissue from the columns.
    trimmed_matrix = matrix.iloc[1:, :-1]

    # Build a mask for the strict upper triangle (k=1 means the diagonal is not masked)
    mask = np.triu(np.ones_like(trimmed_matrix, dtype=bool), k=1)

    # Plot the heatmap using the trimmed matrix and the strict upper triangle mask.
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(trimmed_matrix, annot=True, fmt=".3f", cmap='vlag',
                     mask=mask, linewidths=0.5, vmin=vmin, vmax=vmax,
                     cbar_kws={"shrink": 0.8, "label": cbar_label})
    plt.title(title, fontsize=14)
    ax.set_xlabel("Tissue Class", fontsize=axis_label_size)
    ax.set_ylabel("Tissue Class", fontsize=axis_label_size)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize, rotation=0)

    # Adjust colorbar label and tick label sizes.
    cb = ax.collections[0].colorbar
    cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)

    plt.tight_layout()
    save_path = output_dir / fig_name
    plt.savefig(save_path, format='svg')
    plt.close()
    print(f"Heatmap saved to {save_path}")


def plot_bx_histograms_by_tissue(df, patient_id, bx_index, output_dir, structs_referenced_dict, default_exterior_tissue, fig_name="histograms.svg", bin_width=0.05, spatial_df=None):
    """
    Creates overlapping outline histograms of 'Binomial estimator' values per tissue class for a given biopsy.

    Args:
        df (DataFrame): The input dataframe.
        patient_id (str): The patient ID to filter by.
        bx_index (int): The biopsy index to filter by.
        output_dir (str or Path): Directory to save the plot.
        fig_name (str): Filename for the saved plot.
        bin_width (float): Width of histogram bins.
        spatial_df (DataFrame or None): Optional spatial features dataframe to annotate with.

    Returns:
        None
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter dataframe
    filtered_df = df[(df['Patient ID'] == patient_id) & (df['Bx index'] == bx_index)]

    if filtered_df.empty:
        print(f"No data found for Patient ID: {patient_id}, Bx index: {bx_index}")
        return

    # Get unique tissue classes
    #tissue_classes = filtered_df['Tissue class'].unique()

    tissue_classes = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )

    # Create figure
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette(n_colors=len(tissue_classes))

    

    # Plot Binomial estimator outline histograms
    for tissue_class, color in zip(tissue_classes, colors):
        vals = filtered_df[filtered_df['Tissue class'] == tissue_class]['Binomial estimator'].dropna()
        total_voxels = len(vals)
        plt.hist(vals, bins=np.arange(0, 1 + bin_width, bin_width), histtype='step', linewidth=2,
                 label=tissue_class, color=color)


    # Optionally annotate with spatial info
    if spatial_df is not None:
        match = spatial_df[(spatial_df['Patient ID'] == patient_id) & (spatial_df['Bx index'] == bx_index)]
        if not match.empty:
            match = match.iloc[0]
            centroid_dist = match['BX to DIL centroid distance']
            #surface_dist = match['NN surface-surface distance']
            length = match['Length (mm)']
            position = f"{match['Bx position in prostate LR']} / {match['Bx position in prostate AP']} / {match['Bx position in prostate SI']}"

            # Add annotation box in plot
            annotation = (f"Bx Length: {length:.1f} mm\n"
                          f"Bx→DIL centroid dist: {centroid_dist:.1f} mm\n"
                          #f"Bx→DIL NN surface dist: {surface_dist:.1f} mm\n"
                          f"Total Voxels: {total_voxels}\n"
                          f"Sector: {position}")
            plt.gca().text(1.02, 0.95, annotation, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    # Final plot adjustments
    plt.title("Multinomial Estimator Distribution by Tissue Class", fontsize=16)
    plt.xlabel("Multinomial Estimator", fontsize=16)
    plt.ylabel("Number of Voxels", fontsize=16)

    
    plt.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes


    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / fig_name, format='svg')
    print(f"Histogram saved to {output_dir / fig_name}")




def production_plot_sum_to_one_tissue_class_binom_regression_matplotlib(multi_structure_mc_sum_to_one_pt_wise_results_dataframe,
                                                                        specific_bx_structure_index,
                                                                        patientUID,
                                                                        structs_referenced_dict,
                                                                        default_exterior_tissue,
                                                                        patient_sp_output_figures_dir,
                                                                        general_plot_name_string):


    def stacked_area_plot_with_confidence_intervals(patientUID,
                                                    bx_struct_roi,
                                                    df, 
                                                    stacking_order):
        """
        Create a stacked area plot for binomial estimator values with confidence intervals,
        stacking the areas to sum to 1 at each Z (Bx frame) point. Confidence intervals are 
        shown as black dotted lines, properly shifted to align with stacked lines.

        :param df: pandas DataFrame containing the data
        :param stacking_order: list of tissue class names, ordered by stacking hierarchy
        """
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Initialize cumulative variables for stacking
        y_cumulative = np.zeros_like(x_range)

        # Set color palette for tissue classes
        #colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))
        colors = sns.color_palette(n_colors=len(stacking_order))

        # Loop through the stacking order
        for i, tissue_class in enumerate(stacking_order):
            tissue_df = df[df['Tissue class'] == tissue_class]

            # Perform kernel regression for binomial estimator
            kr = KernelReg(endog=tissue_df['Binomial estimator'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            y_kr, _ = kr.fit(x_range)

            # Perform kernel regression for CI lower and upper bounds
            kr_lower = KernelReg(endog=tissue_df['CI lower vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            kr_upper = KernelReg(endog=tissue_df['CI upper vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            ci_lower_kr, _ = kr_lower.fit(x_range)
            ci_upper_kr, _ = kr_upper.fit(x_range)

            # Stack the binomial estimator values (fill between previous and new values)
            ax.fill_between(x_range, y_cumulative, y_cumulative + y_kr, color=colors[i], alpha=0.7, label=tissue_class)

            # Plot the black dotted lines for confidence intervals, shifted by the cumulative values
            ax.plot(x_range, y_cumulative + ci_upper_kr, color='black', linestyle=':', linewidth=1)  # Upper confidence interval
            ax.plot(x_range, y_cumulative + ci_lower_kr, color='black', linestyle=':', linewidth=1)  # Lower confidence interval

            # Update cumulative binomial estimator for stacking
            y_cumulative += y_kr

        # Final plot adjustments
        ax.set_title(f'{patientUID} - {bx_struct_roi} - Stacked Binomial Estimator with Confidence Intervals by Tissue Class',
             fontsize=16,      # Increase the title font size
             #fontname='Arial' # Set the title font family
            )
        ax.set_xlabel("Biopsy Axial Dimension (mm)",
              fontsize=16,    # sets the font size
              #fontname='Arial'
               )   # sets the font family

        ax.set_ylabel("Multinomial Estimator (stacked)",
                    fontsize=16,
                    #fontname='Arial'
                    )


        ax.legend(loc='best', facecolor='white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes

        plt.tight_layout()

        return fig



    tissue_heirarchy_list = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )
        
    sp_structure_mc_sum_to_one_pt_wise_results_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]
    #extract bx_struct_roi
    bx_struct_roi = sp_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx ID"].values[0]

    fig = stacked_area_plot_with_confidence_intervals(patientUID,
                                                bx_struct_roi,
                                                sp_structure_mc_sum_to_one_pt_wise_results_dataframe, 
                                                tissue_heirarchy_list)

    svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

    fig.savefig(svg_dose_fig_file_path, format='svg')

    # clean up for memory
    plt.close(fig)





def production_plot_sum_to_one_tissue_class_nominal_plotly(multi_structure_mc_sum_to_one_pt_wise_results_dataframe,
                                                patientUID,
                                                specific_bx_structure_index,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string,
                                                patient_sp_output_figures_dir,
                                                structs_referenced_dict,
                                                default_exterior_tissue
                                                ):

    def tissue_class_sum_to_one_nominal_plot(df, y_axis_order, patientID, bx_struct_roi):
        df = misc_tools.convert_categorical_columns(df, ['Tissue class', 'Nominal'], [str, int])

        # Generate a list of colors using viridis colormap in Matplotlib
        stacking_order = y_axis_order
        #colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))  # Same method you used in Matplotlib
        colors = sns.color_palette(n_colors=len(stacking_order))

        # Convert the colors to a format Plotly understands (hex strings)
        hex_colors = ['#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        hex_colors.reverse()
        # Create a color mapping for tissue classes
        color_mapping = dict(zip(stacking_order, hex_colors))
        
        # Hack to adjust the size of the markers
        scale_size = 15
        df["Nominal scaled size"] = df["Nominal"] * scale_size  # Scale size as needed

        # Create the scatter plot and pass the custom color map
        fig = px.scatter(
            df, 
            x='Z (Bx frame)', 
            y='Tissue class', 
            size="Nominal scaled size",  # Size based on Nominal scaled size (0 or 15)
            size_max=scale_size,  # Set size for the points that appear
            color='Tissue class',  # Use tissue class for color assignment
            color_discrete_map=color_mapping,  # Apply the custom color mapping
            title=f'Sum-to-one Nominal tissue class along biopsy major axis (Pt: {patientID}, Bx: {bx_struct_roi})'
        )

        

        # Clear all existing legend entries
        fig.for_each_trace(lambda trace: trace.update(showlegend=False))

        # Add dummy scatter points for the legend with fixed size
        for tissue_class in list(reversed(stacking_order)):
            fig.add_scatter(
                x=[None],  # Dummy invisible point
                y=[None],
                mode='markers',
                marker=dict(size=scale_size, color=color_mapping[tissue_class], symbol='x'),
                name=tissue_class,  # Ensure tissue class appears in legend
                showlegend=True
            )

        # Customize point style
        fig.update_traces(
            marker=dict(
                symbol='x',  # Change to other shapes like 'diamond', 'square', etc.
                #line=dict(width=2, color='DarkSlateGrey'),  # Add border to points
                #size=15,  # Set a base size (adjustable)
                #opacity=1,  # Set point transparency
                #color = 'black'
            )
        )

        # Customize labels and make the plot flatter by tweaking y-axis category settings
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Biopsy Axial Dimension (mm)",
                    font=dict(
                        size=30,
                    )
                ),
                tickfont=dict(
                    size=40,
                )
                ),
            yaxis=dict(
                title=dict(
                    text="Nominal Tissue Class",
                    font=dict(
                        size=30,
                        color="black"
                    )
                ),
                tickfont=dict(
                    size=40,
                    color="black"
                ),
                categoryorder='array',  # Set custom order
                categoryarray=y_axis_order,  # Use the provided order for categories
                tickvals=y_axis_order,  # Ensure the ticks follow this order
                tickmode='array',
                ticktext=y_axis_order,
                scaleanchor="x",  # Lock the aspect ratio of x and y
                dtick=1  # Control category spacing
            ),
            height=400,  # Adjust the overall height of the plot to flatten it
            legend_title_text='Tissue class'  # Set legend title
        )

        fig.update_xaxes(range=[-0.5, df["Z (Bx frame)"].max() +0.5])


        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)
        fig.update_layout(
        paper_bgcolor='white',  # Background of the entire figure
        plot_bgcolor='white'    # Background of the plot area
        )

        return fig 


    # Define the specific order for the y-axis categories
    y_axis_order = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )
    y_axis_order.reverse()

    mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]

    
    # Plotting loop
    bx_struct_roi = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe["Bx ID"].values[0]
    
    fig = tissue_class_sum_to_one_nominal_plot(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, y_axis_order, patientUID, bx_struct_roi)

    bx_sp_plot_name_string = f"{bx_struct_roi} - " + general_plot_name_string

    svg_dose_fig_name = bx_sp_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height/3) # added /3 here to make the y axis categories closer together, ie to make the plot flatter so that it can fit beneath the sum-to-one spatial regression plots.

    html_dose_fig_name = bx_sp_plot_name_string+'.html'
    html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 



def plot_distance_ridges_for_single_biopsy(
    distance_df,
    stats_df,
    binom_df,
    save_dir,
    fig_title_suffix,
    fig_name_suffix,
    cancer_tissue_label,
    fig_scale=1.0,
    dpi=300,
    add_text_annotations=True,
    x_label="Distance (mm)",
    y_label="Biopsy Axial Dimension (mm)"
):

    plt.ioff()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    for (struct_idx, struct_type) in distance_df[['Relative structure index', 'Relative structure type']].drop_duplicates().itertuples(index=False):
        sub_df = distance_df[(distance_df['Relative structure index'] == struct_idx) & (distance_df['Relative structure type'] == struct_type)]
        sub_stats_df = stats_df[(stats_df[('Relative structure index', '')] == struct_idx) & (stats_df[('Relative structure type', '')] == struct_type)]
        struct_name = sub_df['Relative structure ROI'].iloc[0]

        for dist_col in ['Struct. boundary NN dist.', 'Dist. from struct. centroid']:
            df = sub_df.copy()
            df = df.astype({"Voxel index": int, dist_col: float})

            coloring_enabled = binom_df is not None
            if coloring_enabled:
                binom_df_filtered = binom_df[binom_df["Tissue class"] == cancer_tissue_label]
                cmap = sns.color_palette("viridis", as_cmap=True)
                norm = Normalize(vmin=0, vmax=1)
                sm = ScalarMappable(norm=norm, cmap=cmap)

            patient_id = df['Patient ID'].iloc[0]
            bx_index = df['Bx index'].iloc[0]
            bx_id = df['Bx ID'].iloc[0]
            voxel_ids = df['Voxel index'].unique()

            def annotate_and_fill(x, color, label, **kwargs):
                label_val = float(label)
                voxel_stats = sub_stats_df[
                    (sub_stats_df[('Voxel index', '')] == label_val) &
                    (sub_stats_df[('Patient ID', '')] == patient_id) &
                    (sub_stats_df[('Bx index', '')] == bx_index)
                ].iloc[0]

                # get nominal from Trial num == 0 from the distance_df
                nominal = df[(df['Voxel index'] == label_val) & (df['Trial num'] == 0)][dist_col].mean()
                mean = voxel_stats[(dist_col, 'mean')]
                std = voxel_stats[(dist_col, 'std')]
                z_start = voxel_stats[('Voxel begin (Z)', '')]
                z_end = voxel_stats[('Voxel end (Z)', '')]
                q_vals = [voxel_stats[(dist_col, f'{q}%')] for q in [5, 25, 50, 75, 95]]

                ax = plt.gca()
                kde = gaussian_kde(x)
                x_grid = np.linspace(x.min(), x.max(), 1000)
                y_vals = kde(x_grid)
                y_scaled = y_vals / np.max(y_vals) if np.max(y_vals) > 0 else y_vals

                if coloring_enabled:
                    binom_mean = binom_df_filtered[
                        (binom_df_filtered['Voxel index'] == label_val) &
                        (binom_df_filtered['Patient ID'] == patient_id) &
                        (binom_df_filtered['Bx index'] == bx_index)
                    ]["Binomial estimator"].mean()
                    fill_color = cmap(norm(binom_mean))
                else:
                    fill_color = "gray"

                ax.fill_between(x_grid, y_scaled, alpha=0.5, color=fill_color)
                ax.axvline(x=mean, color='orange', linestyle='-', linewidth=1)
                ax.axvline(x=nominal, color='red', linestyle='-', linewidth=1)
                for qv in q_vals:
                    ax.axvline(x=qv, color='gray', linestyle='--', linewidth=1)

                if add_text_annotations:
                    annotation = (
                        f"Segment: ({z_start:.1f}, {z_end:.1f}) mm"
                        + (f" | Tumor score: {binom_mean:.2f}" if coloring_enabled else "")
                        + f"\nMean: {mean:.2f} mm | SD: {std:.2f} | Nominal: {nominal:.2f}"
                    )
                    ax.text(1.02, 0.5, annotation, transform=ax.transAxes, ha='left', va='center', fontsize=8, color=color)

                ax.set_yticks([0.5])
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='x', which='both', labelbottom=True, length=3, width=0.8)

            max_q95 = sub_stats_df[(dist_col, '95%')].max()
            min_q05 = sub_stats_df[(dist_col, '5%')].min()
            palette = {v: 'black' for v in voxel_ids}

            g = sns.FacetGrid(df, row='Voxel index', hue='Voxel index', aspect=15, height=1, palette=palette)
            g.map(annotate_and_fill, dist_col)

            sorted_voxels = sorted(voxel_ids)
            for i, ax in enumerate(g.axes.flat):
                z_start = sub_stats_df[(sub_stats_df[('Voxel index', '')] == sorted_voxels[i]) &
                                       (sub_stats_df[('Patient ID', '')] == patient_id) &
                                       (sub_stats_df[('Bx index', '')] == bx_index)]['Voxel begin (Z)'].values[0]
                z_end = sub_stats_df[(sub_stats_df[('Voxel index', '')] == sorted_voxels[i]) &
                                     (sub_stats_df[('Patient ID', '')] == patient_id) &
                                     (sub_stats_df[('Bx index', '')] == bx_index)]['Voxel end (Z)'].values[0]
                tick_label = f"V{sorted_voxels[i]} ({z_start:.1f}–{z_end:.1f})"
                ax.set_yticks([0.5])
                ax.set_yticklabels([tick_label], fontsize=9)
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='y', labelsize=9)
                ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
                ax.set_axisbelow(True)

            g.set(xlim=(min_q05, max_q95))
            g.set_titles("")
            g.set_axis_labels(x_label, "")
            g.fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=11)

            if coloring_enabled:
                g.fig.text(0.87, 0.5, 'Tumor tissue score', va='center', rotation='vertical', fontsize=10)
                cbar_ax = g.fig.add_axes([0.88, 0.2, 0.015, 0.6])
                g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

            g.fig.subplots_adjust(left=0.23, right=0.85 if coloring_enabled else 0.93, top=0.9, bottom=0.05)

            legend_lines = [
                Line2D([0], [0], color='orange', lw=1, label='Mean Distance'),
                Line2D([0], [0], color='red', lw=1, label='Nominal Distance'),
                Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Quantiles (5%, 25%, 50%, 75%, 95%)')
            ]
            g.fig.legend(
                handles=legend_lines,
                loc='upper right',
                bbox_to_anchor=(1.25, 0.985),
                frameon=True,
                facecolor='white',
                fontsize=9
            )

            g.fig.text(0.07, 0.93, f"Patient ID: {patient_id} | Bx ID: {bx_id} | Structure: {struct_name}",
                       ha='left', fontsize=9, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            plt.suptitle(f"{fig_title_suffix} - {dist_col} ({struct_name})", fontsize=12, fontweight='bold', y=0.98)

            fig_h = 1.0 * len(voxel_ids) * fig_scale
            fig_w = 10.0 * fig_scale
            g.fig.set_size_inches(fig_w, fig_h)

            filename_suffix = f"{fig_name_suffix}_{struct_name.replace(' ', '_')}_{dist_col.replace(' ', '_')}"
            save_path = os.path.join(save_dir, f"{patient_id}_{bx_id}_ridge_plot_{filename_suffix}.svg")
            g.fig.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')

            png_path = save_path.replace(".svg", ".png")
            g.fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')

            plt.close(g.fig)
