# One-time helper: builds the combosciplex "processed" cache that
# Data.process_data() (src/data_process/data.py:46-64) needs to skip pertpy.
#
# The actual, clearly-named file is saved at:
#   data/GRN/combosciplex/combosciplex_processed_cache.h5ad
# and then copied to the exact path process_data() hardcodes as its cache check
# (os.path.join(self.data_path, self.data_name, 'processed.h5ad')):
#   data/scDFM/combosciplex/combosciplex/processed.h5ad
#
# Data.process_data() for combosciplex only calls annotate_compounds()/
# get_molecular_fingerprints() when that second path doesn't exist yet. Once it
# exists, process_data() just loads it directly.
#
# The original annotate_compounds() (src/utils/_preprocessing.py) imports pertpy,
# which pulls in scvi-tools -> an OLD jax/jaxlib that conflicts with this repo's
# flow_matching code (which needs a NEW jax via flax/optax/equinox/lineax). Rather
# than juggling two incompatible jax versions in one environment, this script
# reimplements the compound-name -> PubChem CID/name lookup directly against
# PubChem's PUG-REST API (no pertpy needed) and reuses:
#   - get_smiles_via_http() from _preprocessing.py (already pertpy-free)
#   - get_molecular_fingerprints() from _preprocessing.py (rdkit only, no pertpy)
# so the resulting processed.h5ad has the exact same schema
# ({prefix}_pubchem_name, {prefix}_pubchem_ID, {prefix}_smiles, uns["fingerprints"])
# that process_data() and the rest of the training pipeline expect.
#
# Requirements (all safe to install in the main env - none of them touch jax):
#   pip install rdkit requests
#
# Usage (from the project root, normal environment, no new venv needed):
#   python src/utils/build_combosciplex_processed_cache.py
#   python src/utils/generate_grn_from_data_pipeline.py --datasets combosciplex

import os
import shutil
import sys
import time
from urllib.parse import quote

import numpy as np
import requests
import scanpy as sc

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils._preprocessing import get_smiles_via_http, get_molecular_fingerprints

PUBCHEM_REQUEST_DELAY = 0.2  # seconds between PubChem requests, to be polite to the API


def get_pubchem_cid(name, timeout=15):
    """Looks up the PubChem CID for a compound name via PUG-REST (no pertpy needed)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/cids/TXT"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    text = r.text.strip()
    return text.splitlines()[0].strip() if text else None


def get_pubchem_iupac_name(cid, timeout=15):
    """Looks up the PubChem IUPAC name for a CID via PUG-REST (no pertpy needed)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/TXT"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.text.strip()


def annotate_compounds_no_pertpy(adata, compound_keys, control_category="control"):
    """
    pertpy-free replacement for src.utils._preprocessing.annotate_compounds().
    Produces the same obs columns ({prefix}_pubchem_name, {prefix}_pubchem_ID,
    {prefix}_smiles) by querying PubChem's PUG-REST API directly.
    """
    for key in compound_keys:
        # Cast away from Categorical first: newer pandas raises TypeError when
        # fillna()/replace() would introduce a value that isn't an existing category.
        adata.obs[key] = adata.obs[key].astype(object)
        adata.obs[key] = adata.obs[key].replace(control_category, np.nan)
        unique_names = adata.obs[key].dropna().unique().tolist()

        cid_map, name_map, smiles_map = {}, {}, {}
        for name in unique_names:
            cid = get_pubchem_cid(name)
            time.sleep(PUBCHEM_REQUEST_DELAY)
            if cid is None:
                print(f"  [WARN] could not resolve PubChem CID for '{name}'")
                continue

            iupac_name = get_pubchem_iupac_name(cid)
            time.sleep(PUBCHEM_REQUEST_DELAY)
            smiles = get_smiles_via_http(cid)
            time.sleep(PUBCHEM_REQUEST_DELAY)

            cid_map[name] = cid
            name_map[name] = iupac_name or name
            smiles_map[name] = smiles
            print(f"  {name} -> CID={cid}, smiles={'OK' if smiles else 'MISSING'}")

        adata.obs[f"{key}_pubchem_ID"] = adata.obs[key].map(cid_map)
        adata.obs[f"{key}_pubchem_name"] = adata.obs[key].map(name_map)
        adata.obs[f"{key}_smiles"] = adata.obs[key].map(smiles_map)

        adata.obs[key] = adata.obs[key].fillna(control_category)
        adata.obs[f"{key}_pubchem_name"] = adata.obs[f"{key}_pubchem_name"].fillna(control_category)


def main():
    data_dir = os.path.join(project_root, "data", "scDFM", "combosciplex")
    raw_path = os.path.join(data_dir, "combosciplex.h5ad")

    # data.py::process_data() hardcodes this exact path as its cache check
    # (os.path.join(self.data_path, self.data_name, 'processed.h5ad')), so the file
    # must exist there for process_data() to skip pertpy. We still write the real,
    # clearly-named copy under data/GRN/combosciplex/ (grouped with the other GRN
    # pipeline artifacts) and then copy it into the path process_data() expects.
    required_cache_dir = os.path.join(data_dir, "combosciplex")
    required_cache_path = os.path.join(required_cache_dir, "processed.h5ad")

    grn_dir = os.path.join(project_root, "data", "GRN", "combosciplex")
    named_cache_path = os.path.join(grn_dir, "combosciplex_processed_cache.h5ad")

    if os.path.exists(required_cache_path):
        print(f"{required_cache_path} already exists - nothing to do.")
        return

    if os.path.exists(named_cache_path):
        print(f"{named_cache_path} already exists - copying it instead of re-annotating.")
        os.makedirs(required_cache_dir, exist_ok=True)
        shutil.copy2(named_cache_path, required_cache_path)
        print(f"-> Copied cache to {required_cache_path}")
        return

    print(f"Loading raw data from {raw_path} ...")
    adata = sc.read_h5ad(raw_path)

    adata.obs["condition"] = adata.obs.apply(
        lambda x: "control" if x["condition"] == "control+control" else x["condition"], axis=1
    )
    adata.obs["is_control"] = adata.obs.apply(
        lambda x: True if x["condition"] == "control" else False, axis=1
    )

    print("Annotating compounds via PubChem PUG-REST (no pertpy, needs network access)...")
    annotate_compounds_no_pertpy(adata, compound_keys=["Drug1", "Drug2"])

    print("Computing molecular fingerprints via rdkit...")
    get_molecular_fingerprints(adata, compound_keys=["Drug1", "Drug2"])
    adata.uns["fingerprints"]["control"] = np.zeros(1024)

    os.makedirs(grn_dir, exist_ok=True)
    adata.write(named_cache_path)
    print(f"-> Saved cache (clearly named) to {named_cache_path}")

    os.makedirs(required_cache_dir, exist_ok=True)
    shutil.copy2(named_cache_path, required_cache_path)
    print(f"-> Copied cache to the path process_data() actually reads: {required_cache_path}")


if __name__ == "__main__":
    main()
