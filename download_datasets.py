#!/usr/bin/env python3

# Script to inflate HUTER data (zipped subfolders) to TMP,
# read the matrix in,
# preproc,
# concat,
# write to h5ad format.

# %% SETUP
import gzip as gz
import pandas as pd
import loompy as lm
import numpy as np
import requests
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

## get ensemble attributes for trying out conversion
ens_attributes = ["ensembl_gene_id","ensembl_gene_id_version","external_gene_name","external_gene_source","external_synonym","gene_biotype","arrayexpress","genecards","hgnc_id","mirbase_id","entrezgene_id","refseq_mrna","description","start_position","end_position","chromosome_name"]

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
          host="www.ensembl.org",
        ).set_index("ensembl_gene_id_version")
        for ens_attr in ens_attributes # np.array_split(ens_attributes, 8)
      ]
  )
  annot.drop(columns=["Gene stable ID version.1"]).infer_objects().to_parquet(annot_f)

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


# %% get the table from the website https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html
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


# %% format the table
for col in url_cols:
  html_table[[col, col + "_url"]] = pd.DataFrame(html_table[col].to_list(), index=html_table.index)
  html_table[col + "_url"] = "https://petropoulos-lanner-labs.clintec.ki.se/" + html_table[col + "_url"].astype(str)

# %% load the stuff

# load and join them all into a table with multiprocessing, 
# split the list of datasets into pieces and, `reduce` allocates too much memory
with Pool(processes=4) as pool:
  file_list = pool.starmap(
    dl_gz,
    [(url, Path(raw_folder))
      for url in html_table.counts_url.to_list()]
  )
html_table["local_file_loc"] = file_list
# dl_gz(html_table.counts_url[11], Path(raw_folder))
# Path(Path(html_table.counts_url[11]).stem).stem


# %% format phenotype per cell id
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
)
del comb_cid


# %% load the downloaded datasets, combine them into one based on the geneset included

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
probe_id_data = [ # other datasets that have probe id's
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
# get the filenames for the datasets
html_table_rnaseq = html_table[html_table["Accession number"].isin(whole_rna_data)].reset_index()
html_table_probes = html_table[html_table["Accession number"].isin(probe_id_data)].reset_index()


# %% combine the datasets now
whole_comb = reduce(
  lambda l,r: pd.merge(l, r, how="outer", on="Gene"), [
    pd.read_parquet(f_path)
      for f_path in html_table.query("`Accession number` in @whole_rna_data").local_file_loc
    ]
  ).convert_dtypes()

# write the larger table out to parquet just to save time on reduce every run
tmp_writeout=Path(data_folder) / "tmp_combined_writeout.parquet"
if tmp_writeout.exists():
  probe_comb = pd.read_parquet(tmp_writeout)
else:
  probe_comb = reduce(
    lambda l,r: pd.merge(l, r, how="outer", on="Gene"), [
      pd.read_parquet(f_path)
        for f_path in html_table.query("`Accession number` in @probe_id_data").local_file_loc
      ]
    ).convert_dtypes()
  probe_comb.to_parquet(tmp_writeout)


# %% Create an loom objects from the tables and save it
# format gene-id annotatoin for row-level annotation
annot = (
  annot
  [["external_gene_name", "ensembl_gene_id", "start_position", "end_position", "chromosome_name"]]
  .dropna(subset=["external_gene_name"])
  .drop_duplicates()
  .set_index("external_gene_name")
  .rename(columns={"ensembl_gene_id": "Accession"}) # the loompy convention
)

lm.create(
  str(Path(data_folder) / "embryo_rnaseq_raw.loom"),
  whole_comb.to_numpy(dtype=np.integer, na_value=0),
  whole_comb.index.to_frame().join(annot, how="left").drop_duplicates("Gene").to_dict("list"),
  comb_pheno.loc[whole_comb.columns].reset_index().to_dict("list")
)

lm.create(
  str(Path(data_folder) / "embryo_probeset_raw.loom"),
  probe_comb.to_numpy(dtype=np.integer, na_value=0),
  probe_comb.index.to_frame().join(annot, how="left").drop_duplicates("Gene").to_dict("list"),
  comb_pheno.loc[probe_comb.columns].reset_index().to_dict("list")
)

# and finally create a dataset of inner join between probe and whole datasets
comb_dat = pd.merge(whole_comb, probe_comb, how="inner", on="Gene")
lm.create(
  str(Path(data_folder) / "embryo_raw.loom"),
  comb_dat.to_numpy(dtype=np.integer, na_value=0),
  comb_dat.index.to_frame().join(annot, how="left").drop_duplicates("Gene").to_dict("list"),
  comb_pheno.loc[comb_dat.columns].reset_index().to_dict("list")
)

# testing grounds
# import anndata as an
# an.read_loom(Path(data_folder) / "embryo_rnaseq_raw.loom")
