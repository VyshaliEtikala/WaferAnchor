# DRIFT-SENSE — complete synthetic benchmark and hybrid localization

This project is a runnable hackathon prototype for:

**Problem Statement 2: AI-powered navigation-error recovery for wafer inspection tools.**

It intentionally avoids OpenCV (`cv2`). The implementation uses NumPy, SciPy,
scikit-image and Matplotlib.

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
