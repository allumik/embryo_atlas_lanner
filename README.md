# Script to combine Zhao et al. 2025 embryo datasets

## Instructions

1. Create the environment using `conda` or similar tool.
 
2. Run `download_datasets.py`

## Output

The output will be in the data directory, formatted as `loom` files for better compatibility between Seurat and Scanpy. The groups follow the Zhao et al. 2025 description of the datasets and how they are grouped. The main output dataset is the `comb_core_raw.loom` - a compilation of the 6 datasets they used for integration in their study.

# Citation

This script downloads data from https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html

Zhao, C., Plaza Reyes, A., Schell, J.P. et al. A comprehensive human embryo reference tool using single-cell RNA-sequencing data. Nat Methods 22, 193–206 (2025). https://doi.org/10.1038/s41592-024-02493-2

*Huge thanks to all the authors for this herculean effort to compile those datasets!*
*I know from experience that it is not an easy task to even get a hold of all those "open" datasets...*
