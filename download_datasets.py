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

data_folder = getenv("DATA_FOLDER")
raw_folder = getenv("RAW_DATA_FOLDER")

## ensure that the working data directory is created
Path(data_folder).expanduser().mkdir(exist_ok=True)

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
annot_f = Path(raw_folder) / "annot.parquet"
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
html_table = pd.merge( # read it two times, one with links and other without -> easier to format :)
  left = pd.read_html(
    "https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html",
    extract_links=None
  )[0].drop(columns=url_cols),
  right = pd.read_html(
    "https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html",
    extract_links="body"
  )[0][url_cols],
  left_index=True, right_index=True
)

# format the table
for col in url_cols:
  html_table[[col, col + "_url"]] = pd.DataFrame(html_table[col].to_list(), index=html_table.index)
  html_table[col + "_url"] = "https://petropoulos-lanner-labs.clintec.ki.se/" + html_table[col + "_url"].astype(str)



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
    [(url, Path(raw_folder))
      for url in html_table.counts_url.to_list()]
  )
html_table["local_file_loc"] = file_list


# %% format phenotype per cell id
print("Downloading and reformatting the phenotype data")
# get the cell-id to dataset annot
comb_cid = pd.concat([
  pd.read_parquet(row.local_file_loc)
    .columns
    .to_frame(index=False, name="cell")
    .assign(acc_code=row["Accession number"])
    .join(html_table.set_index("Accession number"), on="acc_code", how="left")
    for _, row in html_table.iterrows()
], ignore_index=True).set_index("cell").infer_objects()

# get the pheno files for cells
comb_pheno = reduce(
  lambda l,r: pd.merge(l, r, how="outer"), [
    pd.read_table(StringIO(requests.get(meta_url).text))
      for meta_url in html_table.metadata_url
  ]
).convert_dtypes().set_index("cell")

# join the cell id with cell information
comb_pheno = (
  pd.merge(comb_cid, comb_pheno, how="left", on="cell")
  .replace("<NA>", np.NaN)
  .infer_objects()
  .reset_index()
  .rename(columns={"cell":"CellID"})
  .set_index("CellID")
  .map(replace_non_ascii)
)
del comb_cid



# %% some dataset filtering for downstream tasks

# annotate datasets that are weird or do not fit our expectations
weird_data = [
  "GSE44183", # has floating point values and "genes" named "1-Dec" etc, otherwise seems to be whole-rna
  "CNP0001454", # functional RNA / mirna profiles
  "GSE247111", # weird "genes" named "selection_*" in the end
  "GSE196365", # LINC genes in the list
  "PRJCA017779" # the Assembloid study with LINC genes in the list
]
whole_rna_data = [ # dts with whole transcriptome genesets
  "GSE109555", "GSE191286"
]
probe_id_data = [ # other datasets that have foreign id's
  "GSE36552","E-MTAB-3929","GSE136447",
  "E-MTAB-9388","PRJEB30442", "GSE66507",
  "E-MTAB-8060","GSE150578","GSE156596",
  "GSE177689","GSE158971","GSE178326"
  "GSE210962","GSE226794","GSE218314",
  "GSE208195","GSE232861","GSE239932",
  "GSE166422","E-MTAB-10018","GSE167924",
  "GSE182791","This project(GSE254641)",
  "E-MTAB-10581","GSE134571","GSE171820",
  "PRJCA017779"
]
# filter out those datasets with those weird data things
html_table = html_table[
  ~html_table["Accession number"]
  .isin(weird_data) # + whole_rna_data
  ].reset_index(drop=True)



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

# now add from another set of ids for GENCODE IDs
# gene_ids = pd.merge(
  # gene_ids,
  # annot.loc[~np.isin(annot.index, annot.hgnc_symbol)], 
  # how="left", left_index=True, right_on="hgnc_symbol"
# )


# %% load parquet files and add them to the loom file iteratively 
# more memory efficient than combining parquets into a large matrix and saving it in one go
# like a lot more efficient

print("Starting to write into the loom files")
loom_files=[]
for f_path in html_table.local_file_loc: 
  print("Writing loom from file:", f_path)
  count_df = pd.read_parquet(f_path).convert_dtypes()
  loom_f = str(Path(data_folder) / (Path(f_path.stem).stem + ".loom"))
  lm.create(
    filename=loom_f,
    layers=(
      count_df
      [~count_df.index.duplicated(keep="first")] # some Gene id's seem to be duplicated
      .reindex(gene_ids.index) # realign the data matrix to gene_ids
      .to_numpy(dtype=np.integer, na_value=0)
    ),
    row_attrs=gene_ids.reset_index().to_dict("list"),
    col_attrs=comb_pheno.loc[count_df.columns].reset_index().to_dict("list")
  )
  loom_files.append(loom_f)
  # and remove the parquet files to save up diskspace
  os.remove(f_path)

print("Combining and writing out the raw dataset:")
lm.combine(
  loom_files,
  Path(data_folder).expanduser() / "embryo_lanner_comb_raw.loom"
  # key="Gene" # dont need it as everything is aligned already. also breaks for some reason
)
# now remove the intermediate loom files to save up on space :)
for f_path in loom_files: os.remove(f_path)

# %% testing grounds
# glimpse_loom(loom_files[2])
glimpse_loom(Path(data_folder) / "embryo_lanner_comb_raw.loom")
# import anndata as an
# sc_dat = an.read_loom(Path(data_folder) / "embryo_lanner_comb_raw.loom")
# print(sc_dat)
# sc_dat.write_h5ad(Path(data_folder) / "embryo_lanner_comb_raw.h5ad")

# genes_ = []
# for f_path in loom_files:
  # # glimpse_loom(f_path)
  # with lm.connect(f_path) as loom_f:
    # print("attrs:", np.unique(loom_f.ra["Gene"], return_counts=True))


# %%
