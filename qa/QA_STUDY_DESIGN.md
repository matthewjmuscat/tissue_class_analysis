# QA Study Design

## Status

This document locks the design for the first dedicated QA manuscript and the first additive `qa/` lane in this repo.

Notation for the manuscript and figures is locked separately in:

- [QA_NOTATION.md](/home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/qa/QA_NOTATION.md)

Scope of this document:

- lock the paper 1 scientific question
- lock the family definition and analysis tables
- lock the endpoint hierarchy
- lock the primary statistical comparison plan
- lock the target figure set
- define the implementation deliverables that will drive `main_pipe_QA.py` and `production_plots_QA.py`

Out of scope for paper 1:

- a full cross-fraction lesion-registry project
- a new primary threshold-based DIL-support endpoint
- broad radiomics model fishing
- guidance-map optimization as the main paper

## Paper 1 Positioning

This manuscript should build directly on the published tissue-classification PMB paper:

- [Muscat_2026_Phys._Med._Biol._71_075012-published-article.pdf](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_1-classification_paper-PMB/Muscat_2026_Phys._Med._Biol._71_075012-published-article.pdf>)

The PMB paper already established:

- the probabilistic along-core tissue framework
- the DIL sampling descriptors `⟨PD⟩` and `max(PD)`
- the role of these descriptors in QA
- the idea of simulated comparison cores and quantifiable headroom
- the future role of radiomics and cohort-level QA

This QA paper should therefore use the same descriptor language and extend it into a matched-family cohort analysis.

## Working Manuscript Concept

Working concept:

`How well did the real cores sample the intended DIL, how much improvement headroom existed relative to centroid and optimized references, and what lesion/OAR geometries determined that difficulty?`

The paper should answer three questions:

1. How well did the real cores sample the intended DIL, using the already published DIL descriptors?
2. How much improvement headroom existed relative to centroid and optimized references?
3. Which lesion geometries were hardest to target, and what prostate/OAR tradeoffs accompanied that difficulty?

## Family Definition

### Locked Family Key

The family key for paper 1 is:

- `Base patient ID + Relative DIL index`

Notes:

- `Patient ID` may be fraction-labeled, such as `184 (F1)` and `184 (F2)`.
- Raw lesion names are not reliable enough to use as the main family key.
- The index columns were built specifically to preserve the intended lesion mapping.
- In the current March 3, 2026 QA-ready cohort export, collapsing to `Base patient ID + Relative DIL index` reconstructs all `26/26` lesion contexts into complete families.

### Locked Family Members

Each family contains:

- `1` centroid reference core
- `1` optimal reference core
- `1` to `3` real observed cores

This is not a triad-only design. It is a matched-family design with one-to-many real attempts.

### Matching Principle

The primary analysis will stay within the native matching logic already encoded in the dataset.

For paper 1 we do not need a heavy lesion-registry project, provided that:

- the family audit confirms complete family reconstruction under `Base patient ID + Relative DIL index`
- the audit table is saved as part of the pipeline outputs

## Data Required

### Core Cohort Tables

Required cohort-level inputs:

- `Cohort: Biopsy basic spatial features dataframe.csv`
- `Cohort: global sum-to-one mc results.csv`
- `Cohort: Tissue class - distances global results.csv`
- `Cohort: 3D radiomic features all OAR and DIL structures.csv`

Required along-core or voxel-level inputs for exemplar figures and optional supplementary metrics:

- voxelwise or along-core tissue probability outputs from the generator
- voxelwise distance tables if needed for profile or ridge plots

### Derived Metadata Needed In QA Lane

The QA lane must derive or preserve:

- `Base patient ID`
- `Relative DIL index`
- `Simulated bool`
- `Simulated type`
- `Bx index`
- `Bx ID`
- real-core count per family
- lesion family size and completeness

## Canonical QA Tables

The QA implementation should build these tables in order.

### 1. `qa_family_members_long`

One row per biopsy-family member.

Key columns:

- `Base patient ID`
- `Patient ID`
- `Relative DIL index`
- `Bx index`
- `Bx ID`
- `Simulated bool`
- `Simulated type`

Joined data blocks:

- published DIL descriptors
- non-DIL tissue descriptors
- biopsy geometry features
- DIL distance summaries
- prostate, urethra, and rectum distance summaries
- DIL radiomics
- prostate radiomics
- contextual prostate-position variables

This is the canonical long table for most downstream analysis.

### 2. `qa_real_core_pairs`

One row per real core, with the paired centroid and optimal reference outcomes from the same family.

Purpose:

- primary headroom analysis
- real-core-level plots
- bootstrap and mixed-model analyses of real versus reference performance

Columns should include:

- family keys
- real-core identifiers
- `real_*` outcome columns
- `centroid_*` outcome columns
- `optimal_*` outcome columns
- delta columns for `centroid - real`
- delta columns for `optimal - real`

### 3. `qa_family_reference_pairs`

One row per lesion family.

Purpose:

- family-level comparison of `optimal - centroid`
- reference-only headroom analysis

Columns should include:

- family keys
- `centroid_*` outcome columns
- `optimal_*` outcome columns
- delta columns for `optimal - centroid`

### 4. `qa_family_real_aggregated`

One row per lesion family, aggregating the real attempts within that family.

Purpose:

- sensitivity analysis
- lesion-level summaries

Recommended real-core summaries:

- mean real performance within family
- best real performance within family
- worst real performance within family
- real-core count within family

## Endpoint Hierarchy

### Primary Endpoint

Primary endpoint for paper 1:

- `DIL Global Mean BE`

Interpretation:

- this is the cohort implementation of the published `⟨PD⟩` language
- it should remain the headline DIL-sampling descriptor

### Key Secondary Endpoint

Key secondary endpoint:

- `DIL Global Max BE`

Interpretation:

- this is the cohort implementation of the published `max(PD)` language
- it captures peak lesion-supporting sampling along the core

### Headroom Endpoints

Headroom must be reported explicitly, not loosely.

For both `DIL Global Mean BE` and `DIL Global Max BE`, define:

- `centroid - real`
- `optimal - real`
- `optimal - centroid`

These are the main headroom quantities.

Interpretation:

- `centroid - real` measures the execution gap relative to a simple lesion-centered reference
- `optimal - real` measures total modeled headroom
- `optimal - centroid` measures the incremental value of optimization beyond a centroid rule

### Secondary QA Endpoints

Secondary endpoints should support interpretation rather than compete with the primary DIL story.

Recommended secondary endpoints:

- `DIL Global Q50 BE`
- `BX to DIL centroid distance`
- `NN surface-surface distance`
- signed `BX to DIL centroid (X)`
- signed `BX to DIL centroid (Y)`
- signed `BX to DIL centroid (Z)`
- `Urethra NN dist mean`
- `Rectum NN dist mean`
- `Periprostatic Global Mean BE`
- `Prostatic Global Mean BE`
- `Urethral Global Mean BE`
- `Rectal Global Mean BE`

### Prostate Context Lane

Prostate metrics are contextual QA metrics, not co-primary targeting metrics.

They answer:

- did the core remain intraprostatic?
- was apparent DIL miss still intraprostatic?
- was there periprostatic drift or escape?

Recommended contextual prostate metrics:

- `Prostatic Global Mean BE`
- `Periprostatic Global Mean BE`
- `BX_to_prostate_centroid_distance_norm_mean_dim` only as a contextual geometry variable, not as a targeting goal

### Optional Supplementary Spatial Endpoint

Threshold-based support extent will not be a primary endpoint in paper 1.

If a threshold-based DIL-support metric is included, it should be supplementary and justified carefully.

Preferred threshold-free supplementary option:

- `probability-weighted DIL extent along the core`

Operationally, this can be represented as:

- the along-core integral of `PD(z)`, if voxelwise data are used
- or a length-scaled restatement of `DIL Global Mean BE` when using summary tables

This gives a spatially intuitive measure without opening a threshold argument.

## Radiomics and Difficulty Analysis

Radiomics are secondary explanatory variables, not the main novelty.

Paper 1 should ask:

- what lesion or prostate geometries are hardest to target well?
- where is headroom largest?
- when does optimization outperform a centroid rule?

Pre-specified predictor shortlist:

- `DIL Volume`
- `DIL Maximum 3D diameter`
- `DIL Elongation`
- `DIL Flatness`
- `Prostate Volume`
- `Prostate Elongation`
- `Prostate Flatness`
- lesion position context, such as prostate location or sextant variables if stable and interpretable

The predictor set should stay small and pre-declared.

## Statistical Plan

### Primary Analysis Level

Headline results will live at the real-core level.

Reason:

- multiple real attempts per lesion are clinically meaningful
- averaging them away too early would remove useful information

The matched context is the lesion family, not a simple unpaired cohort grouping.

### Primary Contrast Set

The three headline contrasts are:

1. `centroid vs real`
2. `optimal vs real`
3. `optimal vs centroid`

Contrast implementation:

- `centroid vs real` and `optimal vs real` use `qa_real_core_pairs`
- `optimal vs centroid` uses `qa_family_reference_pairs`

### Primary Effect Reporting

For each headline outcome and contrast, report:

- observed delta
- `95%` confidence interval
- p-value or significance indicator
- effect size

Priority order in the manuscript:

1. raw deltas
2. confidence intervals
3. effect size
4. p-value

### Locked Main Inference Strategy

Primary uncertainty quantification:

- clustered bootstrap with resampling at the base-patient level

Rationale:

- preserves within-patient and within-family dependence structure
- remains easy to explain
- is robust for the present cohort size

Primary significance rule:

- if the bootstrap `95%` confidence interval excludes zero, treat the contrast as statistically supported

Figure annotations may display:

- exact p-values where available
- or adjusted significance bars derived from the same locked inferential procedure

### Model-Based Analyses

Mixed-effects models will be used for:

- confirmatory model-based comparisons if needed for figures or tables
- explanatory analyses linking difficulty or headroom to radiomics and geometry

Recommended model structure for explanatory analyses:

- random intercept for `Base patient ID`
- lesion family nested within patient when needed

Explanatory models should remain small and pre-specified.

### Sensitivity Analyses

Locked sensitivity analyses:

- real-core-level results versus family-aggregated real performance
- single-real-core families versus multi-real-core families
- optional supplementary threshold sensitivity if a threshold-based support metric is shown

Not a paper-1 sensitivity priority:

- cross-fraction biological comparison
- heavy cross-fraction lesion harmonization project

## Figure Plan

### Figure 1. Study Design and Matched Family Structure

Purpose:

- explain the family definition
- show one centroid, one optimal, and one-to-many real cores per lesion family
- clarify the endpoint and headroom concept

### Figure 2. Representative Case Panel

Purpose:

- give clinical intuition
- overlay real, centroid, and optimal trajectories for one strong example family
- show along-core tissue probability profiles
- optionally show urethra or rectum context

### Figure 3. Primary Endpoint Comparison

Purpose:

- show real versus centroid versus optimal for the two published DIL descriptors

Panels:

- `DIL Global Mean BE`
- `DIL Global Max BE`

Preferred visual style:

- paired or semi-paired distribution plot
- individual observations visible
- manuscript-quality statistical annotation

### Figure 4. Headroom Decomposition

Purpose:

- explicitly display the three headroom quantities

Panels:

- `centroid - real`
- `optimal - real`
- `optimal - centroid`

This figure is one of the key manuscript deliverables and must be designed carefully.

### Figure 5. Geometry and Safety Tradeoff

Purpose:

- show that improved DIL sampling is not interpreted without geometry or safety context

Recommended content:

- DIL performance on one axis
- urethra or rectum distance or burden on the other axis
- family member type encoded by color or marker

Core question:

- when headroom exists, does achieving it appear safe and geometrically reasonable?

### Figure 6. Difficulty-of-Targeting Analysis

Purpose:

- identify which lesion geometries are hard to target
- identify where optimization provides incremental value over centroid

Recommended stratifiers:

- lesion size
- lesion flatness or elongation
- lesion location class if stable and clinically interpretable

### Figure 7. Cohort Localization Accuracy

Purpose:

- reuse and modernize the older QA-style localization figure concept
- present real-core offsets in lesion-centered coordinates

Recommended style:

- transverse and sagittal accuracy plots
- marginal distributions
- Gaussian-fit or covariance ellipse styling
- optional coloring by DIL sampling metric

This should be descriptive, polished, and visually strong, but not the main inferential figure.

## Plotting Philosophy

`production_plots_QA.py` should be a production-grade plotting module, heavily inspired by the more controlled plotting style already used in the dosimetry sibling repo.

Locked plotting principles:

- centralized font family, font sizes, line widths, marker sizes, and export settings
- vector-first output
- consistent annotation boxes and significance annotations
- no ad hoc plot styling inside analysis functions
- standardized color language for `Real`, `Centroid`, and `Optimal`
- enough low-level control to make manuscript panels consistent across revisions

## QA Lane Architecture

The QA lane should be additive and modular.

Planned structure:

```text
qa/
  QA_STUDY_DESIGN.md
  config.py
  load.py
  families.py
  endpoints.py
  stats.py
  plot_style.py
  plot_data.py
main_pipe_QA.py
production_plots_QA.py
```

Implementation principle:

- do not keep growing `main_pipe.py`
- treat the current repo as containing legacy tissue analysis plus reusable functions
- build paper 1 QA as a clean new lane on top of shared loaders and derived tables

## Locked Deliverables

### First Deliverables Before Full Pipeline

These are the first implementation targets:

1. `qa/QA_STUDY_DESIGN.md`
2. family audit table confirming complete family reconstruction
3. `qa_family_members_long`
4. `qa_real_core_pairs`
5. `qa_family_reference_pairs`
6. `qa_family_real_aggregated`
7. figure specification sheet for manuscript panels

### Pipeline Deliverables

The first full QA pipeline should produce:

- analysis-ready tables
- manuscript statistics tables
- polished cohort-level figures
- exemplar figure data packages
- family audit outputs

### Manuscript Deliverable Folder

The repo should also expose a focused manuscript-deliverable CSV pack, separate from the rawer QA analysis outputs.

Locked location:

- [output_data_QA/csv/deliverables](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/csv/deliverables>)

Locked purpose:

- provide table-ready CSVs for manuscript drafting
- provide parse-ready geometry CSVs for prose, supplementary tables, or rapid figure iteration
- avoid forcing the paper-writing workflow to depend directly on every intermediate QA analysis table

Current deliverable CSV set:

- `table_01_cohort_overview.csv`
- `table_02_primary_headroom_summary.csv`
- `table_03_safety_distance_summary.csv`
- `table_04_biopsy_case_catalog.csv`
- `table_05_targeting_feature_ranking.csv`
- `table_06_targeting_location_summary.csv`
- `geometry_biopsy_level_table.csv`
- `geometry_biopsy_level_summary.csv`
- `geometry_voxelwise_table.csv`
- `geometry_voxelwise_group_summary.csv`

These should be treated as the paper-writing layer, while the files under [output_data_QA/csv/qa](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/csv/qa>) remain the full analysis layer.

## Journal Recommendation

Recommended primary target:

- `Medical Physics`

Rationale:

- AAPM describes `Medical Physics` as its flagship journal and explicitly as the international journal of medical physics research and practice.
- This QA paper is more clinically facing than the PMB tissue paper, but it is still a quantitative medical-physics manuscript rather than a pure clinical implementation note.
- The current paper package includes probabilistic modeling, matched-reference comparisons, bootstrap inference, safety geometry, and localization-accuracy analysis, which together fit better as a research-and-practice paper than as a narrowly applied clinical workflow note.

Recommended fallback target:

- `JACMP`

Rationale:

- AAPM describes `JACMP` as publishing papers that help clinical medical physicists perform their responsibilities more effectively and efficiently for the increased benefit of the patient.
- If the manuscript is simplified into a more direct clinical QA implementation paper, with less emphasis on methodological nuance and special-case optimization, `JACMP` becomes a very reasonable target.

Not recommended as first choice for this manuscript:

- `PMB`

Rationale:

- PMB’s official scope explicitly states that papers predominantly clinical or biological in approach are not suitable.
- This paper builds on the PMB tissue paper, but the current manuscript emphasis is now cohort QA, targeting performance, headroom, and clinical geometry tradeoffs rather than a first-principles methods paper.

Source links used for this recommendation:

- [AAPM Publications](https://aapm.org/pubs/default.asp)
- [PMB official scope](https://publishingsupport.iopscience.iop.org/journals/physics-in-medicine-biology/about-physics-medicine-biology/)

## Abstract and Paper Shape

If the primary target is `Medical Physics`, the draft in the new paper repo should stop inheriting the PMB structured abstract and switch to the AAPM-style abstract pattern used in the dosimetry QA and GPR papers:

- `Background`
- `Purpose`
- `Methods`
- `Results`
- `Conclusions`

The current draft at [tissue_class_paper-QA.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_2-tissue_QA_paper/tissue_class_paper-QA.tex>) still carries PMB-style abstract and methods inheritance from paper 1, so this should be changed early rather than late.

## Proposed Manuscript Structure

Recommended section order for the QA paper:

1. `Introduction`
2. `Materials and Methods`
3. `Results`
4. `Discussion`
5. `Conclusion`

Recommended subsection structure:

### Introduction

- clinical motivation for targeted-biopsy QA in the HDR/TRUS/mpMRI setting
- uncertainty-aware sampling context established by the PMB tissue framework
- need for matched-family benchmarking against centroid and optimized references
- explicit statement that this paper asks how well real cores sampled the intended DIL, how much headroom existed, and what geometry determined that headroom

### Materials and Methods

- clinical cohort and biopsy workflow
- inheritance from the published tissue-classification framework
- family definition and matching logic
- published DIL descriptors and headroom quantities
- safety and geometric context metrics
- radiomics and lesion-location variables
- patient-clustered bootstrap and explanatory analyses

### Results

- cohort and family reconstruction summary
- headline DIL sampling comparison for real, centroid, and optimal families
- headroom decomposition
- selected biopsy profiles and disagreement cases
- safety/geometric context
- localization accuracy
- difficulty/headroom associations with lesion geometry

### Discussion

- what the paper adds beyond the tissue paper
- why most cohort-average headroom appears to be `real -> centroid`
- why `optimal -> centroid` still matters in special cases
- what the safety-distance results imply clinically
- what the lesion-size/location trends imply for future guidance-map work
- limitations and next extensions

## Citation Backbone

This manuscript should explicitly cite the full line of previous work, not only paper 1.

Core self-citation backbone:

- tissue framework paper:
  [tissue_class_paper.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_1-classification_paper-PMB/tissue_class_paper.tex>)
- dose exemplar paper:
  [dosimetry-exemplars-paper-1.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/dose_1-two_exemplars_paper-PMB/dosimetry-exemplars-paper-1.tex>)
- dose cohort QA paper:
  [main_dosimetry_cohort_uncertainties_v7.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/dose_2-probabilistic-QA-paper-JACMP/main_dosimetry_cohort_uncertainties_v7.tex>)
- GPR paper:
  [dosimetry-GPR-paper-3.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/dose_3-GPR_paper-AAPM-Medical-Physics/dosimetry-GPR-paper-3.tex>)

Recommended citation roles:

- cite paper 1 for the underlying probabilistic tissue notation, `\(\mathcal{P}_i(z)\)`, `\(\langle \mathcal{P}_{D}\rangle\)`, and `\(\max(\mathcal{P}_{D})\)`
- cite dose exemplar and dose QA papers for the broader scalar-field / uncertainty-propagation program and for polished cohort-style QA framing
- cite the GPR paper when positioning special-case optimization, spatial correlation, and the broader programmatic arc of the work

## Recommended Figure Package

### Main-Text Figures

Recommended main-text figure set:

1. a new study-design / matched-family schematic figure
2. [Fig_QA_01_headline_family_comparison.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_01_headline_family_comparison.pdf>)
3. [Fig_QA_02_headline_headroom.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_02_headline_headroom.pdf>)
4. [Fig_QA_05_selected_dil_profiles.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_05_selected_dil_profiles.pdf>)
5. [Fig_QA_04_safety_distance_family_comparison.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_04_safety_distance_family_comparison.pdf>)
6. [Fig_QA_13_localization_accuracy_centroids.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_13_localization_accuracy_centroids.pdf>)
7. [Fig_QA_10_targeting_difficulty_continuous.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_10_targeting_difficulty_continuous.pdf>)

Main-text note:

- the study-design schematic is still missing and should be created before drafting the paper in earnest
- if the figure count needs to be reduced, the safety-distance comparison or the localization figure can move to supplement

### Supplementary Figures

Recommended supplementary set:

- [Fig_QA_03_centroid_vs_optimal_disagreement.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_03_centroid_vs_optimal_disagreement.pdf>)
- [Fig_QA_06_selected_dil_profiles_step.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_06_selected_dil_profiles_step.pdf>)
- [Fig_QA_07_optimizer_difficulty_summary.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_07_optimizer_difficulty_summary.pdf>)
- [Fig_QA_08_targeting_difficulty_summary.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_08_targeting_difficulty_summary.pdf>)
- [Fig_QA_09_optimizer_difficulty_continuous.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_09_optimizer_difficulty_continuous.pdf>)
- [Fig_QA_11_optimizer_difficulty_categorical.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_11_optimizer_difficulty_categorical.pdf>)
- [Fig_QA_12_targeting_difficulty_categorical.pdf](</home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/output_data_QA/figures/qa/Fig_QA_12_targeting_difficulty_categorical.pdf>)

## Recommended Table Package

### Main-Text Tables

Recommended main-text tables:

1. `table_01_cohort_overview.csv`
2. `table_02_primary_headroom_summary.csv`
3. `table_03_safety_distance_summary.csv`

### Supplementary Tables

Recommended supplementary tables:

1. `table_04_biopsy_case_catalog.csv`
2. `table_05_targeting_feature_ranking.csv`
3. `table_06_targeting_location_summary.csv`
4. `geometry_biopsy_level_summary.csv`
5. `geometry_voxelwise_group_summary.csv`

## What Is Complete Versus Outstanding

### Completed and Ready to Transfer

- family matching and family audit
- manuscript-grade primary/headroom summary tables
- safety-distance summary tables
- biopsy case catalog
- targeting-feature and location summary tables
- parse-ready geometry CSVs at biopsy level and voxelwise level
- polished QA figure lane for the core manuscript analyses

### Still Outstanding Before Serious Drafting

- a clean study-design / matched-family schematic figure
- a final decision on which figures stay main text versus supplement
- actual transfer of the chosen figure files into the paper repo `Images/` structure
- rewriting of the inherited introduction and methods in the new paper repo
- optional later extension: superior-wall / bladder-margin export from `biopsylocalization-python`

## Transfer Guidance To Paper Repo

When drafting into [tissue_class_paper-QA.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_2-tissue_QA_paper/tissue_class_paper-QA.tex>), the first transfer pass should be:

1. switch the abstract skeleton to the target-journal style
2. replace the inherited introduction with the QA-specific storyline above
3. replace the inherited methods section with the matched-family cohort methods
4. copy only the main-text figure set into the paper repo image folder
5. use the deliverables CSV folder as the source for manuscript tables and numeric reporting

The key rule is: draft from the deliverable layer, not directly from every raw QA analysis CSV.

## Open Decisions

These decisions are intentionally deferred but must remain narrow:

- exact effect-size definition to report beside raw deltas
- exact visual style of the headroom figure
- whether any threshold-based spatial endpoint appears in supplement

These are not open:

- primary family key
- primary DIL endpoints
- headroom definition
- contextual role of prostate metrics
- secondary explanatory role of radiomics

## Locked Summary

Paper 1 is a matched-family QA manuscript built directly on the PMB tissue framework.

The locked structure is:

- family key: `Base patient ID + Relative DIL index`
- primary endpoint: `DIL Global Mean BE`
- key secondary endpoint: `DIL Global Max BE`
- headroom: `centroid - real`, `optimal - real`, `optimal - centroid`
- headline analysis level: real core, with dependence handled statistically
- prostate metrics: contextual QA only
- OAR metrics: explicit secondary tradeoff lane
- radiomics: small, pre-specified difficulty-analysis lane

This is the design that `main_pipe_QA.py` and `production_plots_QA.py` should implement.
