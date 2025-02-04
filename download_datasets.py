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
def glimpse_loom(loom_path) -> None:
  with lm.connect(loom_path) as loom_f:
    print("shape:", loom_f.shape)
    print("attrs:", list(loom_f.attrs))
    print("layers:", list(loom_f.layers))
    print("cols:", list(loom_f.ca))
    print("rows:", list(loom_f.ra))



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
# create a bytes stream with the gzip decompression and save the count matrices into a list
# save it also into the specified path as a parquet file and return the filepath
def dl_gz(input_url, out_path, overwrite=False):
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

# load and join them all into a table with multiprocessing
# or do it linearly bc there seems to be some issue with multitasking 
with Pool(processes=4) as pool:
  file_list = pool.starmap(
    dl_gz,
    [(url, raw_folder)
      for url in html_table.counts_url.to_list()]
  )
html_table["local_file_loc"] = file_list # assume the dl order was the same


# %% Now add the modified suppl. table 9
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



# %% get the unique geneids from the datasets for the row attributes
gene_ids = np.unique(np.concatenate([
  pd.read_parquet(f_path, columns=["Gene"]).index.to_numpy()
    for f_path in html_table.local_file_loc
]).ravel())

# add gene annotation for the existing ids and reorder
# theres a mixture of id's, so merge for every id type separately
gene_ids = pd.merge(
  pd.DataFrame(index=pd.Index(gene_ids, name="Gene")),
  annot,
  how="left", left_index=True, right_index=True
)



# %% load parquet files and add them to the loom file iteratively 

# this is more memory efficient than combining parquets into a large matrix and saving it in one go
# ... like a lot more RAM efficient
def format_and_loom(file_list, gene_ids, phenotype, output_file, data_output=Path("./data/")) -> str:
  loom_files=[]
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

  # and now combine the single loom files
  output_loom = data_output / output_file
  lm.combine(
    loom_files,
    output_loom
    # key="Gene" # dont need it as everything is aligned already. also breaks for some reason
  )

  # finally, remove the intermediary loom files
  for f_path in loom_files: os.remove(f_path)

  # and return the output file location as string
  return str(output_loom)



# %% Now start writing out the dataasets into loom

## first load and combine the "core" set
comb_loom_list = [
  format_and_loom(
    html_table.query("core_group").local_file_loc,
    gene_ids,
    comb_pheno,
    "comb_core_raw.loom",
    data_folder
  )
]

## then lets do all the different groupings
# let the errors start rolling in...
for ds_group in html_table.groupby("dataset_group").local_file_loc:
  output_file_name = "comb_g" + str(ds_group[0]) + "_raw.loom"
  comb_loom_list.append(
    format_and_loom(ds_group[1], gene_ids, comb_pheno, output_file_name, data_folder)
  )



# %% clean up on the data isle
# and remove the parquet files to save up diskspace
for f_path in html_table.local_file_loc: os.remove(f_path)



# %% testing grounds
# glimpse_loom(loom_files[2])
glimpse_loom(data_folder / "comb_core_raw.loom")
# import anndata as an
# sc_dat = an.read_loom(data_folder / "comb_core_raw.loom")
# print(sc_dat)
# print(sc_dat.var.index)
# sc_dat.write_h5ad(data_folder / "embryo_lanner_comb_raw.h5ad")

# genes_ = []
# for f_path in loom_files:
  # # glimpse_loom(f_path)
  # with lm.connect(f_path) as loom_f:
    # print("attrs:", np.unique(loom_f.ra["Gene"], return_counts=True))
