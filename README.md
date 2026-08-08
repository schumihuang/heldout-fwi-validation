# Held-Out Shot Validation for FWI Under Modeling Mismatch

This repository contains the source code and reproducibility materials supporting the following research study:

```text
Held-Out Shot Validation for Full-Waveform Inversion Under Modeling
Mismatch: A Marmousi2 Cross-Grid Study
```

The code implements the Marmousi2 cross-grid full-waveform inversion (FWI)
experiments, held-out shot validation rule, saved JSON summaries, and
figure-generation scripts associated with this research.

Repository URL:

```text
https://github.com/schumihuang/heldout-fwi-validation
```

## Repository Contents

- `code/` - Python source files for data conversion, FWI optimization,
  validation-shot stopping, summary generation, and figure generation.
- `data/marmousi2/manifest.json` - Marmousi2 input-data provenance and
  checksum metadata.
- `data/marmousi2/vp_marmousi-ii.segy.gz` - compressed public Marmousi2
  P-wave velocity SEGY input used by the conversion script.
- `results/` - JSON summaries used to produce reported tables and
  validation curves.
- `figures/` - generated validation figures in PDF/PNG format.
- `manuscript/` - LaTeX manuscript source and referenced figure files.
- `requirements.txt` - Python packages for the full workflow.
- `quick_test.py` - small deterministic smoke test for local verification.
- `LICENSE` - MIT License.

Large derived files such as `vp_marmousi2.npy`, uncompressed SEGY files,
and `.npz` model checkpoints are intentionally not tracked. They can be
regenerated from the public compressed Marmousi2 input and scripts.

## Quick Test

The quick test checks that the repository can import the local numerical
utilities and run a deterministic TV-proximal smoke test. It does not run
the full FWI experiments.

From the repository root:

```bash
python -m pip install numpy
python quick_test.py
```

Expected final line:

```text
quick_test passed
```

## Full Environment

For the complete workflow, install the full requirements:

```bash
python -m pip install -r requirements.txt
```

The main packages are `numpy`, `scipy`, `matplotlib`, and `segyio`.

## Reproduce the Experiments

Convert the compressed Marmousi2 SEGY input to the cached NumPy velocity
file:

```bash
python code/convert_marmousi2_segy.py
```

Run the main validation-controlled FWI experiments:

```bash
python code/run_validation_shot_split_crossgrid.py
python code/run_validation_random_splits.py
python code/run_validation_noisy_crossgrid.py
python code/run_validation_random_splits.py --label cropB_sigma8_long --x0-m 9500 --z0-m 900 --sigma 8 --n-chunks 16 --chunk-iter 5 --out-prefix validation_random_splits_cropB_long
python code/summarize_validation_oracle_gap.py
```

Regenerate figures:

```bash
python code/make_validation_figures.py
python code/make_validation_random_split_figures.py
python code/make_validation_noisy_figures.py
python code/make_validation_random_split_figures.py --summary results/validation_random_splits_cropB_long_summary.json --out-stem figures/fig_validation_random_splits_cropB_long
python code/make_validation_model_comparison_figures.py
```

## License

This repository is released under the MIT License. The Marmousi2 input
model is a public benchmark model; its provenance and checksums are
recorded in `data/marmousi2/manifest.json`.
