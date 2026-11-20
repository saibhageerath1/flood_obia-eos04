# Object-based flood mapping from EOS-04 NDFI and temporal backscatter variability

Python tools accompanying the manuscript:

> Bhageerath, Y. V. S., Suresh Babu, A. V., Shashivardhan Reddy, P., & Durga Rao, K. H. V. *Object-based machine learning for transferable flood mapping from NDFI and temporal backscatter variability: a leave-one-location-out assessment over five Indian flood events.* (under review)

The workflow segments the NDFI and temporal standard-deviation (SD) layers into SLIC superpixels,
computes object features, trains machine-learning classifiers (Extra Trees, Random Forest,
Gradient Boosting, SVM, Logistic Regression) against reference flood maps, and evaluates them
with leave-one-location-out (LOLO) validation. A pixel-based baseline and map-production tools are included.
All tools run on a standard CPU; no GPU is required.

## Contents

| Script | Purpose |
|---|---|
| `scripts/flood_segmentation_gui.py` | Multi-class thresholding (K-means, multi-Otsu, quantile, equal interval) of NDFI or SD with per-class statistics against a reference flood layer (GUI). |
| `scripts/flood_obia_ml_gui.py` | Single-scene OBIA: SLIC segmentation, 14 object features, classifier training with spatial block validation, flood and probability maps; applies saved models to new scenes (GUI). Also provides shared functions used by the other tools. |
| `scripts/flood_obia_multisite.py` | Multi-location OBIA with leave-one-location-out validation, model comparison, final model trained on all locations (GUI). |
| `scripts/flood_obia_experiments.py` | Refinement experiments on saved object features: cut-off tuning (nested LOLO), per-location normalisation, input ablation, balanced sampling (GUI). |
| `scripts/flood_pixel_lolo.py` | Pixel-based baseline with identical LOLO validation and object-vs-pixel comparison (GUI). |
| `scripts/error_map.py` | Error map (TP / FP / FN) with map elements for one location (command line). |
| `scripts/make_report_maps.py` | All map figures (input layers, segmentation zoom, flood maps, error maps) for all locations (command line). |
| `scripts/india_districts.geojson`, `scripts/india_states.geojson` | Boundary layers used for map figures. |
| `examples/sites_example.csv` | Template of the location list used by the multi-location and mapping tools. |

All scripts must stay in the same folder because they import functions from `flood_obia_ml_gui.py`
and `flood_obia_multisite.py`.

## Installation

```bash
conda env create -f environment.yml
conda activate flood
```

or `pip install -r requirements.txt` (rasterio and geopandas are easiest to install with conda).

## Input data

For each location: an NDFI GeoTIFF, a temporal SD GeoTIFF on the same grid, and a binary reference flood
GeoTIFF (1 = flood, 0 = no flood). The reference may have a different extent, resolution or projection;
it is reprojected and clipped to the NDFI grid automatically. In the study, the inputs were derived from
EOS-04 MRS Level-2B (HH, γ°, 18 m) data from the NRSC Bhoonidhi portal, and reference maps were NRSC flood
inundation maps. These data are not redistributed here.

## Typical workflow

1. **Multi-location training and validation**
   `python scripts/flood_obia_multisite.py` → add the locations (or load a CSV like `examples/sites_example.csv`),
   choose a model (or *Compare all models*) and run. Outputs: LOLO flood and probability maps, object maps,
   `LOLO_results.csv`, `all_sites_object_features.joblib` and the final model (`.joblib`).
2. **Refinement experiments**
   `python scripts/flood_obia_experiments.py` → load `all_sites_object_features.joblib`.
   Outputs: `experiments_summary.csv`, `experiments_per_site.csv`, optional final model with tuned cut-off.
3. **Pixel-based baseline**
   `python scripts/flood_pixel_lolo.py` → load the same location CSV and, optionally, `LOLO_results.csv`
   for the comparison. Outputs: `pixel_LOLO_results.csv`, `object_vs_pixel_comparison.csv`.
4. **Apply a saved model to a new scene**
   `python scripts/flood_obia_ml_gui.py` → tab *Apply saved model*.
5. **Map figures**
   ```bash
   python scripts/make_report_maps.py sites.csv path/to/obia_multisite_output path/to/report_maps
   python scripts/error_map.py pred_flood.tif reference_flood.tif out_prefix --title "Location"
   ```

## Main settings

SLIC: target object size ≈ 100 pixels, compactness 1.0, σ = 1, 10 iterations, 2048-pixel tiles.
Context window: 31 × 31 pixels. Object label: flood if ≥ 50 % of pixels are flood.
Extra Trees / Random Forest: 300 trees, min. 2 samples per leaf; all classifiers use balanced class weights and random seed 0.

## Boundary data

District boundaries: [datta07/INDIAN-SHAPEFILES](https://github.com/datta07/INDIAN-SHAPEFILES);
state/UT boundaries: [Datameet maps](https://github.com/datameet/maps) (Survey of India depiction). Both simplified for display.
Please check their licences before reuse.

## Citation

If you use this code, please cite the paper above (citation details will be updated on publication)
and this repository (Zenodo DOI to be added).

## License

MIT – see `LICENSE`.
