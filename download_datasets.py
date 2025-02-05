#!/usr/bin/env python3

# Script to read in the data table from
# https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html
# read the table in,
# get the links
# download them
# join and format attributes
# write to loom and/or anndata

# %% SETUP
import gzip as gz
import pandas as pd
import loompy as lm
import scanpy as sc
import numpy as np
import requests
import os
from io import BytesIO
from io import StringIO
from pathlib import Path
from os import getenv
from dotenv import load_dotenv
from functools import reduce
from multiprocessing import Pool

load_dotenv()

data_folder = Path(getenv("DATA_FOLDER")).expanduser()
raw_folder = Path(getenv("RAW_DATA_FOLDER")).expanduser()

## ensure that the working data directory is created
data_folder.mkdir(exist_ok=True)

def replace_non_ascii(s, replacement='*'):
  if isinstance(s, str):
    return ''.join(c if ord(c) < 128 else replacement for c in s)
  return s # Return as is if not a string

# quick loom file overview function
def glimpse_loom(loom_path:str) -> None:
  """Get a glimpse of the structure and dimensions of the `loom_path` Loom file.

  Args:
      loom_path (str): The path to The Great Loominess
  """
  with lm.connect(loom_path) as loom_f:
    print("shape:", loom_f.shape)
    print("attrs:", list(loom_f.attrs))
    print("layers:", list(loom_f.layers))
    print("cols:", list(loom_f.ca))
    print("rows:", list(loom_f.ra))

def filter_zero_genes(loom_file_path:str):
  """Remove all the attributes / rows that have in total zero reads from the `loom_file_path` Loom file.

  Args:
      loom_file_path (str): The path to the loom file.
  """
  with lm.connect(loom_file_path, mode='r') as ds:
    matrix = ds.layers[''].sparse().tocsr()
    zero_gene_indices = np.where(np.diff(matrix.indptr) == 0)[0]

    if len(zero_gene_indices) > 0:
      # Invert the zero gene selection
      genes_to_keep = np.setdiff1d(np.arange(ds.shape[0]), zero_gene_indices)
      # Subset the Loom file
      vw = ds.view[genes_to_keep, :]
    else:
      print(f"Fully populated matrix {loom_file_path}, no need to splice.")
  # Save the view, overwriting the original file
  lm.create(str(loom_file_path), vw.layers, vw.ra, vw.ca)

# get the unique geneids from the datasets for the row attributes
def get_unique_geneids(file_list:list, annotation:pd.DataFrame):
  """Read in all the parquet formatted files in the `file_list`, find the unique common index (aka intersection of index names) and combine it with the `annotation` table from Ensemble biomart.

  Args:
      file_list (list): List of paths to parquet files. Should have a column named "Gene".
      annotation (pd.DataFrame): The annotation table from biomart API. Should have a named "Gene" index column.

  Returns:
      pd.DataFrame: A new table with the annotated gene id's from the `file_list` collection of parquet tables.
  """
  gene_ids = np.unique(np.concatenate([
    pd.read_parquet(f_path, columns=["Gene"]).index.to_numpy()
      for f_path in file_list
  ]).ravel())

  # add gene annotation for the existing ids and reorder
  return pd.merge(
    pd.DataFrame(index=pd.Index(gene_ids, name="Gene")),
    annotation,
    how="left", left_index=True, right_index=True
  )

def format_and_loom(
  file_list:list, 
  phenotype:pd.DataFrame, 
  annotation:pd.DataFrame, 
  output_file:str, 
  data_output:Path=Path("./data/")
  ) -> str:
  """Takes in arguments and generates a loom file, outputting the path to the loom file.
  This is more memory efficient than combining parquets into a large matrix and saving it in one go.

  Args:
      file_list (list): List of parquet files to load in.
      phenotype (pd.DataFrame): Table containing the column attributes (features) of the matrix.
      annotation (pd.DataFrame): Annotation for the gene id's in the row attributes.
      output_file (str): The Loom file name to create.
      data_output (Path, optional): The path to the folder for the output file. Defaults to Path("./data/"). Pretty pointless, should be included in output_file...

  Returns:
      str: The path to the generated Loom file.
  """

  gene_ids = get_unique_geneids(file_list, annotation)
  loom_files = []
  for f_path in file_list: 
    count_df = pd.read_parquet(f_path).convert_dtypes()
    loom_f = str(data_folder / (Path(f_path.stem).stem + ".loom"))
    lm.create(
      filename=loom_f,
      layers=(
        count_df
        [~count_df.index.duplicated(keep="first")] # some Gene id's seem to be duplicated
        .reindex(gene_ids.index) # realign the data matrix to gene_ids
        .to_numpy(dtype=np.integer, na_value=0)
      ),
      row_attrs=gene_ids.reset_index().to_dict("list"),
      col_attrs=phenotype.loc[count_df.columns].reset_index().to_dict("list")
    )
    loom_files.append(loom_f)

  # combine the single loom files
  output_loom = data_output / output_file
  lm.combine(
    loom_files,
    output_loom
    # key="Gene" # dont need it as everything is aligned already. also breaks for some reason
  )

  # and then remove the 0 rows to make the dataset more compact (TODO: but what about dropouts?)
  filter_zero_genes(output_loom)

  # finally, remove the intermediary loom files
  for f_path in loom_files: os.remove(f_path)

  # and return the output file location as string
  return str(output_loom)

def dl_gz(input_url:str, out_path, overwrite=False) -> str: 
  """Create a bytes stream from the specified URL with the gzip decompression. Then saves it into the specified `out_path` as a parquet file and returns the filepath.

  Args:
      input_url (str): The URL from where you wish to download the gzipped table.
      out_path: Path or str of the output file folder. The file name is created from the input url.
      overwrite (bool, optional): Do you wish to overwrite the existing file at location? Defaults to False.

  Returns:
      str: The created Parquet file location.
  """
  outfile = Path(out_path) / (Path(input_url).stem + ".parquet")

  ## check if the file's already there
  if overwrite or (not outfile.exists()):
    ## read the file in with gunzip
    print("Downloading table from:", input_url)
    response = requests.get(input_url)
    with gz.open(filename=BytesIO(response.content), mode="rb") as gz_file:
      outtable = pd.read_csv(gz_file).infer_objects().set_index("Gene")

    ## write the file out to file
    print("Writing it to:", outfile)
    outtable.to_parquet(outfile)
  else:
    print("No operation needed, returning the filename:", outfile)
  return outfile


# %% get ensemble attributes for id conversion
ens_attributes = ["ensembl_gene_id","ensembl_gene_id_version","external_gene_name","external_gene_source","external_synonym","gene_biotype","arrayexpress","genecards","hgnc_symbol","mirbase_id","entrezgene_id","refseq_mrna","description","start_position","end_position","chromosome_name"]

# download the annotation
annot_f = data_folder / "annot.parquet"
if annot_f.exists():
  annot = pd.read_parquet(annot_f)
else:
  annot = reduce( # split the query into parts as there are too many attributes for it to download them all
    lambda l, r: pd.merge(l, r, how="outer", on="ensembl_gene_id_version"),
      [
        sc.queries.biomart_annotations(
          "hsapiens",
          np.concatenate([["ensembl_gene_id_version"], [ens_attr]]),
          host="www.ensembl.org", # useast.ensembl.org
        ).set_index("ensembl_gene_id_version")
        for ens_attr in ens_attributes # np.array_split(ens_attributes, 8)
      ]
  )
  annot.drop(columns=["Gene stable ID version.1"]).infer_objects().to_parquet(annot_f)

# format gene-id annotatoin for row-level annotation
annot = (
  annot
  [["external_gene_name", "ensembl_gene_id", "hgnc_symbol", "start_position", "end_position", "chromosome_name"]]
  .dropna(subset=["external_gene_name", "hgnc_symbol"])
  .drop_duplicates(subset=["external_gene_name", "hgnc_symbol"])
  .rename(columns={"ensembl_gene_id": "Accession", "external_gene_name": "Gene"}) # the loompy convention
  .set_index("Gene")
)



# %% get the table from the website https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html
print("Downloading the table from the website")
url_cols = ["metadata", "counts"]
html_table = pd.merge( # read it two times, one with links and other without -> easier to format with `pd.read_html`:)
  left = pd.read_html(
    "https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html",
    extract_links=None
  )[0].drop(columns=url_cols), # without the links
  right = pd.read_html(
    "https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html",
    extract_links="body"
  )[0][url_cols], # with the links, but `extract_links="body"`
  left_index=True, right_index=True
)

# format the table
for col in url_cols:
  html_table[[col, col + "_url"]] = pd.DataFrame(html_table[col].to_list(), index=html_table.index)
  html_table[col + "_url"] = "https://petropoulos-lanner-labs.clintec.ki.se/" + html_table[col + "_url"].astype(str)

# update the "This project" field with new information
html_table.loc[26, "Publication"] = "Lanner et al., 2024"
html_table.loc[26, "Title"] = "A comprehensive human embryo reference tool using single-cell RNA-sequencing data"
html_table.loc[26, "PubMed"] = 39543283
html_table.loc[26, "Accession number"] = "GSE254641"
# separate Ai et al into two
html_table.loc[32, "Publication"] = "Ai et al., 2023 - embryo"
html_table.loc[33, "Publication"] = "Ai et al., 2023 - assembloid"



# %% load the stuff
print("Trying to download the files into $RAW_DATA_FOLDER")

# load and join them all into a table with multiprocessing
# or do it linearly bc there seems to be some issue with multitasking 
with Pool(processes=4) as pool:
  file_list = pool.starmap(
    dl_gz,
    [(url, raw_folder)
      for url in html_table.counts_url.to_list()]
  )
html_table["local_file_loc"] = file_list # assume the dl order was the same


# %% Now add the modified suppl. table 9 from the article
# first read it in
suppl_datasets = pd.read_csv(data_folder / "41592_2024_2493_MOESM11_ESM_modified.csv")

# and join with the existing data table from the dataset website
html_table = pd.merge(
  left = html_table,
  right = suppl_datasets.drop(columns=["PubMed"]),
  how = "left",
  left_on = ["Accession number", "Publication"],
  right_on = ["GEO", "Data source"]
)



# %% format phenotype per cell id
print("Downloading and reformatting the phenotype data")
# get the cell-id to dataset annot
comb_cid = pd.concat([
  pd.read_parquet(row.local_file_loc)
    .columns
    .to_frame(index=False, name="cell")
    .assign(acc_code=row["Accession number"], pub_name=row["Publication"])
    .join(
      html_table.set_index(["Accession number", "Publication"]), 
      on=["acc_code", "pub_name"],
      how="left"
    )
    for _, row in html_table.iterrows()
], ignore_index=True).set_index("cell").infer_objects()

# get the pheno data for cells
comb_pheno = reduce(
  lambda l,r: pd.merge(l, r, how="outer"), [
    pd.read_table(StringIO(requests.get(meta_url).text))
      for meta_url in html_table.metadata_url
  ]
).convert_dtypes().set_index("cell")

# join the cell id with cell information
comb_pheno = (
  pd.merge(comb_cid, comb_pheno, how="left", on="cell")
  .replace("<NA>", np.nan)
  .infer_objects()
  .reset_index()
  .rename(columns={"cell":"CellID"})
  .set_index("CellID")
  .map(replace_non_ascii)
)
del comb_cid



# %% Now start writing out the dataasets into loom
## first load and combine the "core" set
comb_loom_list = [
  format_and_loom(
    html_table.query("core_group").local_file_loc,
    comb_pheno,
    annot,
    "comb_core_raw.loom",
    data_folder
  )
]

## then lets do all the different groupings
# let the errors start rolling in...
for ds_group in html_table.groupby("dataset_group").local_file_loc:
  output_file_name = "comb_g" + str(ds_group[0]) + "_raw.loom"
  comb_loom_list.append(
    format_and_loom(ds_group[1], comb_pheno, annot, output_file_name, data_folder)
  )



# %% clean up on the data isle
# and remove the parquet files to save up diskspace
for f_path in html_table.local_file_loc: os.remove(f_path)



# %% testing grounds
# glimpse_loom(data_folder / "comb_core_raw.loom")
# for i in range(1, 6):
#   glimpse_loom(data_folder / f"comb_g{i}_raw.loom")
# import anndata as an
# sc_dat = an.read_loom(data_folder / "comb_core_raw.loom")
# print(sc_dat)
# print(sc_dat.var.index)
# sc_dat.write_h5ad(data_folder / "embryo_lanner_comb_raw.h5ad")