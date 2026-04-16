# QA Notation

This note locks the working notation for the biopsy-targeting QA paper.
It builds directly on the notation already established in the tissue-classification paper source:
[tissue_class_paper.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_1-classification_paper-PMB/tissue_class_paper.tex:109>),
especially the core-level descriptors introduced around
[tissue_class_paper.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_1-classification_paper-PMB/tissue_class_paper.tex:469>)
and the distance descriptors defined around
[tissue_class_paper.tex](</home/matthew-muscat/Documents/UBC/Research/Research_papers_git/tissue_1-classification_paper-PMB/tissue_class_paper.tex:625>).

**Inherited Tissue-Paper Notation**
- Along-core tissue-class probability map: `\(\mathcal{P}_i(z)\)`
- Tissue-class index: `\(i \in \{D,P,E,U,R\}\)` for DIL, prostatic, periprostatic, urethral, and rectal classes
- Primary DIL sampling descriptors:
  - `\(\langle \mathcal{P}_{D} \rangle\)` = mean DIL sampling probability
  - `\(\max(\mathcal{P}_{D})\)` = peak DIL sampling probability
- Distance descriptors inherited from the pathology-link section:
  - `\(d_1\)` = mean distance from sampled biopsy voxels to the targeted DIL centroid
  - `\(d_{1,\mathrm{norm}}\)` = `\(d_1 / d_{\mathrm{DIL,max}}\)`
  - `\(d_2\)` = mean nearest-neighbour distance from sampled biopsy voxels to the targeted DIL boundary

**QA Family Notation**
- Family index `\(g \in \{R,C,O\}\)`:
  - `\(R\)` = real biopsy
  - `\(C\)` = centroid reference biopsy
  - `\(O\)` = optimized reference biopsy
- A family-specific metric is written with a superscript:
  - `\(\langle \mathcal{P}_{D} \rangle^{(R)}\)`
  - `\(\langle \mathcal{P}_{D} \rangle^{(C)}\)`
  - `\(\langle \mathcal{P}_{D} \rangle^{(O)}\)`
  - and analogously for `\(\max(\mathcal{P}_{D})\)`, `\(d_1\)`, `\(d_2\)`, and OAR/prostate burden terms

**Headroom Notation**
- Real-to-reference and reference-to-reference deltas are written as:
  - `\(\Delta^{(C-R)} \langle \mathcal{P}_{D} \rangle = \langle \mathcal{P}_{D} \rangle^{(C)} - \langle \mathcal{P}_{D} \rangle^{(R)}\)`
  - `\(\Delta^{(O-R)} \langle \mathcal{P}_{D} \rangle = \langle \mathcal{P}_{D} \rangle^{(O)} - \langle \mathcal{P}_{D} \rangle^{(R)}\)`
  - `\(\Delta^{(O-C)} \langle \mathcal{P}_{D} \rangle = \langle \mathcal{P}_{D} \rangle^{(O)} - \langle \mathcal{P}_{D} \rangle^{(C)}\)`
- The same pattern is used for:
  - `\(\Delta^{(\cdot)} \max(\mathcal{P}_{D})\)`
  - `\(\Delta^{(\cdot)} d_1\)`
  - `\(\Delta^{(\cdot)} d_2\)`
  - `\(\Delta^{(\cdot)} \langle \mathcal{P}_{U} \rangle\)`, `\(\Delta^{(\cdot)} \langle \mathcal{P}_{R} \rangle\)`, `\(\Delta^{(\cdot)} \langle \mathcal{P}_{P} \rangle\)`, `\(\Delta^{(\cdot)} \langle \mathcal{P}_{E} \rangle\)`

**Proposed QA Extensions**
- Mean prostatic support: `\(\langle \mathcal{P}_{P} \rangle\)`
- Mean periprostatic burden: `\(\langle \mathcal{P}_{E} \rangle\)`
- Mean urethral burden: `\(\langle \mathcal{P}_{U} \rangle\)`
- Mean rectal burden: `\(\langle \mathcal{P}_{R} \rangle\)`
- Directional lesion-frame offsets:
  - `\(\delta_{\mathrm{LR}}\)`, `\(\delta_{\mathrm{AP}}\)`, `\(\delta_{\mathrm{SI}}\)`
- Surface-to-surface miss distance:
  - `\(d_0\)` = nearest biopsy-to-DIL surface distance
- OAR nearest-neighbour distances:
  - `\(d_{2,U}\)` = mean nearest-neighbour distance from biopsy voxels to the urethral boundary
  - `\(d_{2,R}\)` = mean nearest-neighbour distance from biopsy voxels to the rectal boundary

**Recommended Figure Usage**
- Panel titles for the headline figures should use the symbols directly:
  - `\(\langle \mathcal{P}_{D} \rangle\)`
  - `\(\max(\mathcal{P}_{D})\)`
- Family-comparison axes should use family superscripts where needed:
  - `\(\langle \mathcal{P}_{D} \rangle^{(R)}\)`, `\(\langle \mathcal{P}_{D} \rangle^{(C)}\)`, `\(\langle \mathcal{P}_{D} \rangle^{(O)}\)`
- Headroom figures should use the delta notation directly:
  - `\(\Delta^{(C-R)}\)`, `\(\Delta^{(O-R)}\)`, `\(\Delta^{(O-C)}\)`
- Centroid-vs-optimal disagreement figures should use:
  - x-axis: `\(\langle \mathcal{P}_{D} \rangle^{(C)}\)` or `\(\max(\mathcal{P}_{D})^{(C)}\)`
  - y-axis: `\(\langle \mathcal{P}_{D} \rangle^{(O)}\)` or `\(\max(\mathcal{P}_{D})^{(O)}\)`

**Current Code Anchor**
- The code-side symbol map used by the QA plotting lane lives in [notation.py](/home/matthew-muscat/Documents/UBC/Research/biopsy_tissue_class_stat_analysis_corrected/qa/notation.py:1).
