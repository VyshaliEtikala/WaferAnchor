# DRIFT-SENSE — complete synthetic benchmark and hybrid localization

DRIFT-SENSE is a hybrid computer-vision system for recovering the location of a high-magnification semiconductor reference region inside a larger lower-magnification search image and estimating the resulting navigation/localization error.

This project is a runnable hackathon prototype for:

**Problem Statement 2: AI-powered navigation-error recovery for wafer inspection tools.**

Wafer inspection tools can experience navigation errors in which the inspection system is positioned at an incorrect physical location.

Given:
a high-magnification reference image representing a small physical region, and
a larger lower-magnification search image,
the objective is to automatically determine where the reference region occurs inside the search image and estimate the localization/navigation error.

Ground-truth coordinates are used only for synthetic-data generation and evaluation. They are not supplied to the inference engine.

It intentionally avoids OpenCV (`cv2`). The implementation uses NumPy, SciPy,
scikit-image and Matplotlib.

## Key Features
DRAM-style and FinFET-style synthetic semiconductor structures
Approximately 1000×1000 search scenes
Reference/search scale relationship around 10×
Randomized target placement
Independent reference and search noise
SEM-like edge brightening
Blur, illumination and contrast variation
Rotation and scaling variation
Structural imperfections
Difficult highly-periodic/ambiguous cases
Coarse-to-fine localization
Sub-pixel refinement
Quantitative self-evaluation
Visualization of ground truth vs prediction
No OpenCV / cv2 dependency

## Pipeline

1. **Synthetic semiconductor image generation**
   - DRAM: word-lines, bit-lines and contact/via crossings.
   - FinFET: dense fins and horizontal gate bars.
   - Randomized pitch, line width, contrast, rotation, scale, blur, illumination,
     edge brightening, independent sensor noise and structural defects.
   - 25% difficult periodic cases by default.
   - Search and reference captures receive independent degradation/noise.
   - Exact ground-truth target coordinates are stored in JSON.

2. **Hybrid localization**
   - Multi-scale intensity normalized cross-correlation.
   - Edge-profile correlation.
   - Ridge/structure correlation.
   - Candidate peak extraction.
   - Sub-pixel quadratic peak refinement.
   - Local phase-correlation refinement.
   - Multiple-point/geometric gradient agreement.
   - Confidence/uncertainty estimation.
   - All signals are fused into one final score.

3. **Evaluation**
   - Center displacement `(dx, dy)`.
   - Euclidean navigation error in search-image pixels.
   - Normalized error.
   - Success rate, median, mean, P90 and 2/5 px thresholds.
   - Runtime per sample.

4. **Visualization**
   - Reference.
   - Search image with ground-truth rectangle.
   - Search image with predicted rectangle and center markers.
   - Error histogram and per-sample error plot.

## Install

```bash
pip install -r requirements.txt
```

## One-command benchmark

```bash
python -m driftsense.cli all --n 100
```

This creates:

```text
data/synthetic/
  references/*.npy
  searches/*.npy
  meta/*.json
  dataset.json

outputs/
  results.csv
  summary.json
  error_histogram.png
  per_sample_error.png
  sample_00000.png
  sample_00001.png
  sample_00002.png
```

## Separate commands

Generate 150 samples:

```bash
python -m driftsense.cli generate --out data/synthetic --n 150 --seed 2026
```

Evaluate:

```bash
python -m driftsense.cli evaluate --data data/synthetic --out outputs/results.csv
```

Visualize sample 7:

```bash
python -m driftsense.cli visualize --data data/synthetic --sample 7
```

## Jupyter use

```python
from driftsense.synthetic import save_dataset
from driftsense.evaluate import evaluate_dataset, summarize

save_dataset("data/synthetic", n=100, seed=2026)
rows = evaluate_dataset("data/synthetic", "outputs/results.csv")
summary = summarize(rows)
summary
```

## Important interpretation

The synthetic generator has a latent semiconductor scene. The search image is
a degraded lower-magnification global capture. The reference is extracted from
the corresponding physical region and then independently rendered as a cleaner
high-magnification capture. Noise arrays are generated separately, so the
localizer cannot win by memorizing identical sensor noise.

The `target_scale=0.10` setting means the reference footprint is approximately
10% of its reference-image width/height when represented in the 1000×1000
search image. It can be changed to stress the system.

## For the real hidden test set

The inference entry point is:

```python
from driftsense.localize_api import predict
x = predict(reference, search, architecture="dram")
```

It returns the estimated top-left coordinate, width/height, center, fused
confidence score and uncertainty in search-image pixels.

No ground-truth coordinate is used by inference.

## Architecture

- `synthetic.py` — physical-inspired DRAM/FinFET generator.
- `preprocess.py` — normalization and structural maps.
- `localization.py` — complete hybrid coarse-to-fine localizer.
- `evaluate.py` — benchmark and metrics.
- `visualize.py` — diagnostic figures.
- `cli.py` — command line interface.
- `localize_api.py` — hidden-test-friendly inference API.

## Failure and Ambiguity Handling
Highly periodic semiconductor layouts can contain multiple visually similar locations.
DRIFT-SENSE retains these difficult cases during evaluation rather than hiding them. Confidence and uncertainty information can be used to distinguish high-confidence localization from potentially ambiguous matches.
This is important for practical deployment because a visually similar periodic match should not automatically be treated as a certain physical localization.
