"""Convert the downloaded Marmousi2 VP SEGY gzip file to a NumPy cache.

The public AGL/University of Houston file is stored in
data/marmousi2/vp_marmousi-ii.segy.gz. This script decompresses it,
reads traces with segyio, transposes to (depth, x), converts km/s to
m/s when needed, and writes vp_marmousi2.npy plus a checksum manifest.
"""
import gzip
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import segyio


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "marmousi2")
GZ = os.path.join(DATA, "vp_marmousi-ii.segy.gz")
SEGY = os.path.join(DATA, "vp_marmousi-ii.segy")
NPY = os.path.join(DATA, "vp_marmousi2.npy")
MANIFEST = os.path.join(DATA, "manifest.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.exists(GZ):
        raise FileNotFoundError(f"Missing downloaded file: {GZ}")
    os.makedirs(DATA, exist_ok=True)

    with gzip.open(GZ, "rb") as src, open(SEGY, "wb") as dst:
        shutil.copyfileobj(src, dst)

    with segyio.open(SEGY, "r", ignore_geometry=True) as f:
        tracecount = f.tracecount
        samples = len(f.samples)
        arr = segyio.tools.collect(f.trace[:])

    vp = arr.T.astype(np.float64)
    if np.nanmax(vp) < 20.0:
        vp *= 1000.0
    np.save(NPY, vp)

    manifest = {
        "source_url": "http://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz",
        "tracecount": int(tracecount),
        "samples_per_trace": int(samples),
        "shape_depth_x": [int(vp.shape[0]), int(vp.shape[1])],
        "velocity_units": "m/s",
        "min_velocity": float(vp.min()),
        "max_velocity": float(vp.max()),
        "sha256": {
            "vp_marmousi-ii.segy.gz": sha256(GZ),
            "vp_marmousi-ii.segy": sha256(SEGY),
            "vp_marmousi2.npy": sha256(NPY),
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
