# SGEP

PyTorch implementation of **Spatially Grounded Evidential Prototypes for Open-Set Medical Image Recognition** by Qinao Hu.

SGEP combines independent global and ROI encoders with learnable class prototypes. Prototype distances produce Dirichlet evidence; ROI consistency and background suppression regularize its spatial source. The repository includes data preparation, fixed SAM masks, training, open-set inference, baselines, ablations, calibration, statistical analysis, and figures.

## 1. Install

Use Python 3.10 or newer. The CPU CI environment uses Python 3.11, PyTorch 2.7.1, and torchvision 0.22.1.

```bash
git clone https://github.com/Hqa77880011/SGEP.git
cd SGEP
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell. Install matching PyTorch and torchvision builds. For CPU:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

For CUDA, choose the matching wheels using the [PyTorch installer](https://pytorch.org/get-started/locally/), then install the package. Add SAM when preparing automatic masks:

```bash
python -m pip install -e ".[sam]"
```

Check the interfaces without downloading medical images or pretrained weights:

```bash
python -m pytest
sgep smoke
```

The smoke check generates small synthetic images, performs one epoch with a tiny encoder for SGEP and Softmax, and checks optimizer updates, safe checkpoint reload, threshold calibration, individual decisions, score identities, analysis, and paired cluster resampling. Its default temporary files are removed. Use `sgep smoke --output runs/smoke` to inspect them. These fixture outputs are for software validation.

## 2. Obtain and prepare data

Download data from their providers and keep their terms of use:

| Dataset | Source | Use |
| --- | --- | --- |
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) | Source dermoscopy recognition |
| GastroVision | [Dataset repository](https://github.com/DebeshJha/GastroVision), [OSF](https://osf.io/84e7f/) | Source endoscopy recognition |
| PH2 | [Provider page](https://www.fc.up.pt/addi/ph2%20database.html) | External dermoscopy evaluation |
| HyperKvasir | [Dataset repository](https://github.com/simula/hyper-kvasir), [OSF](https://osf.io/mh9sj/) | External endoscopy transfer evaluation |

Data, masks, annotations, weights, and run outputs stay outside Git. The repository contains the category roles in [roles.csv](sgep/resources/roles.csv) and the fixed external crosswalk in [external_mapping.csv](sgep/resources/external_mapping.csv).

For HAM10000, place the extracted image folders under `data/ham/images` and the release metadata at `data/ham/HAM10000_metadata.csv`:

```bash
sgep prepare-ham --metadata data/ham/HAM10000_metadata.csv --images data/ham/images --output data/ham/manifest.csv
```

Image folders are searched recursively. The preparation uses `lesion_id` to keep repeated views together. Complete, reliable `patient_id` values supersede lesions when included. You may supply reconciled `group_id` and `group_type` columns directly.

GastroVision's public filename/class metadata do not include patient, procedure, or sequence identifiers. Obtain real group linkage and provide a CSV with `image_id,group_id,group_type,source_cohort`; `image_id` accepts the filename with or without its extension. `group_type` is `patient`, `procedure`, or `sequence`.

```bash
sgep prepare-gastro --metadata data/gastro/GastroVision_metadata.csv --images data/gastro/images --linkage data/gastro/linkage.csv --output data/gastro/manifest.csv
```

The source split targets 70/10/5/15% of known groups for `train`, `selection`, `calibration`, and `test`, with split seed 2026. Proxy groups are divided 2:1 between selection and calibration. Final unknowns occur only in test. Whole groups stay together; final-unknown status takes precedence for a group spanning roles. Actual image counts depend on the supplied grouping.

For other directory layouts, use normalized metadata and `sgep prepare --metadata data/metadata.csv --output data/manifest.csv`. A custom role CSV can be passed through `--roles`. Normalized metadata columns are:

| Column | Meaning |
| --- | --- |
| `dataset`, `release` | Dataset and release identifiers |
| `image_id`, `relative_path` | Unique image identifier and path relative to `--root` |
| `category` | Literal source label |
| `group_id`, `group_type` | Real patient/lesion/procedure/sequence linkage |
| `source_cohort` | Supplied provenance; a release namespace may be used when finer provenance is unavailable |
| `mask_path` | Automatic binary ROI path; can be empty before SAM generation |
| `exclusion_reason` | Optional prespecified exclusion |

Preparation adds `mapped_class,role,split,exclusion_reason`. Paths are relative to one data root; group IDs are local to a dataset and must reconcile repeated acquisitions across source cohorts. Filenames are not substitutes for patient or sequence linkage.

## 3. Generate the fixed ROI prior

Download the [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) into `weights/sam_vit_b_01ec64.pth`, then run:

```bash
sgep masks --manifest data/ham/manifest.csv --root data/ham/images --checkpoint weights/sam_vit_b_01ec64.pth --device cuda --output data/ham/manifest_sam.csv
```

Use `--device cpu` when CUDA is unavailable. Repeat with the GastroVision manifest and its image root. Masks are written below the image root at `masks/sam_vit_b/<dataset>/<image_id>.png`. The output CSV binds those masks to the images; `manifest_sam_mask_log.csv` records candidate scores and full-image fallbacks. Existing masks can be supplied in a manifest using `mask_path`, with one binary mask at each image's original dimensions.

SAM is frozen and uses automatic prompts. The candidate rule, SAM settings, and preprocessing are specified in [implementation.md](docs/implementation.md). Independently annotated reference masks are reserved for quality evaluation.

## 4. Train, calibrate, and evaluate

The default configuration follows the revision: independent ImageNet ResNet18 branches, 256-dimensional fusion, AdamW, batch size 32, and 100 epochs. Only known training images update parameters. Known/proxy selection images choose the best checkpoint by OSCR; ties retain the earlier epoch.

```bash
sgep train --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --seed 11 --output runs/ham/sgep/seed_11
sgep calibrate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/sgep/seed_11/calibration
sgep evaluate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --threshold runs/ham/sgep/seed_11/calibration/threshold.json --output runs/ham/sgep/seed_11/test
sgep analyze --predictions runs/ham/sgep/seed_11/test/predictions.npz --calibration runs/ham/sgep/seed_11/calibration/calibration_predictions.npz --threshold runs/ham/sgep/seed_11/calibration/threshold.json --near-category mel --output runs/ham/sgep/seed_11/analysis
```

For GastroVision, change the manifest/root and use `--near-category "Gastric Polyps"`; category matching is case-insensitive. If a manifest contains multiple datasets, add `--dataset HAM10000` or `--dataset GastroVision`. `--device auto` chooses CUDA when available. New training runs require new output directories.

The primary threshold is the smallest calibration score accepting at least 95% of known calibration images. `calibrate --rule coverage90`, `coverage97`, and `youden` provide the other operating points. Youden maximizes calibration CCR minus proxy false acceptance, choosing the smaller threshold on ties. Final test scores never select this threshold.

| Output | Contents |
| --- | --- |
| `checkpoint.pt`, `config.json`, `history.csv` | Selected parameters, class order, post-hoc state, configuration, losses, and selection OSCR |
| `selection_predictions.npz` | Individual scores from the selected epoch |
| `calibration_predictions.npz`, `threshold.json` | Reserved calibration scores and fixed rejection rule |
| `predictions.npz`, `predictions.csv`, `metrics.json` | Individual test scores, probabilities, accepted/rejected labels, and metrics |
| `roc`, `oscr`, `reliability`, `uncertainty`, `evidence` figures | PNG/PDF plots where their required populations exist |
| Analysis CSV files | Threshold sensitivity, near/mixed unknowns, reference-mask strata, evidence diagnostics, and annotation coverage when supplied |

All unknown scores use the same direction: larger scores imply more unknown. Known classes are accepted when `score <= threshold`. AUROC treats unknowns as positives. OSCR integrates correct-known acceptance against unknown false acceptance over the full score curve; it is independent of the deployment threshold. FPR95 is a descriptive test-curve statistic at 95% known acceptance, while `unknown_fpr` uses the saved calibration threshold. Accuracy, ECE, AUROC, OSCR, coverage, and FPR are percentages; NLL and multiclass Brier are dimensionless. Metrics requiring a missing population are JSON `null`.

## 5. Baselines, search, and paired experiments

Methods are `softmax`, `maxlogit`, `openmax`, `energy`, `cac`, `arpl`, `postmax`, `edl`, `prototype`, `roi_guided`, and `sgep`. For a single baseline, use `train --method energy` with the same manifest and follow the calibration/evaluation commands above. Softmax, MaxLogit, Energy, OpenMax, and PostMax use the same cross-entropy training architecture. Their rejection scores select checkpoints separately by selection OSCR. OpenMax, PostMax, and CAC fit their post-hoc state using correctly classified known training examples only.

The paper search budget is 24 configurations with selection seeds 11 and 22. Search records every configuration and failure, writes the selected configuration, and does not evaluate calibration or test images:

```bash
sgep search --method sgep --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/search/sgep
sgep train --config runs/ham/search/sgep/selected.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --seed 11 --output runs/ham/tuned_sgep/seed_11
```

Repeat search with `--method` for each tuned baseline. SGEP keeps the supplied configuration as trial one and draws the other 23 without replacement from the paper grid. Baselines use their method-specific grids; when the grid has fewer than 24 combinations, repeated draws are recorded. `--count` and `--seeds` adjust the budget. Add `--plan` to inspect the search without training.

Suites execute paired seeds 11, 22, 33, 44, and 55, then calibrate and evaluate each selected checkpoint. `main` covers all methods; `ablation` covers the seven component transitions; `factorial` crosses ROI/background loss switches; `spatial` retrains aligned, random-area, displaced, mean-fill, and dual-CE controls. `all` includes every block. Suites use the supplied configuration for SGEP and explicitly defined changes for other variants; use per-method selected YAML files for separately tuned baseline comparisons.

```bash
sgep suite --suite all --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/paper --plan
sgep suite --suite main --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/main
sgep suite --suite ablation --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/ablation
sgep suite --suite factorial --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/factorial
sgep suite --suite spatial --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/spatial
sgep compare --runs runs/ham/main runs/gastro/main --output runs/comparison
sgep compare --runs runs/ham/factorial runs/gastro/factorial --output runs/factorial_analysis
```

Suite run directories contain `config.json`, `checkpoint.pt`, `history.csv`, `calibration_predictions.npz`, `threshold.json`, `test_predictions.npz`, and `metrics.json`. The suite root has a concrete run plan and `results.csv`. `compare` consumes completed run records, produces mean/sample-SD tables and figures, and pairs seeds for SGEP versus ROI-guided. With both datasets supplied, the four primary AUROC/FPR95 tests form one Holm family. Factorial analysis reports paired ROI/background main effects and their interaction.

Sampling intervals use genuine group IDs and individual predictions. For each reporting seed, run:

```bash
sgep bootstrap --baseline runs/ham/main/roi_guided/seed_11/test_predictions.npz --proposed runs/ham/main/sgep/seed_11/test_predictions.npz --baseline-threshold runs/ham/main/roi_guided/seed_11/threshold.json --proposed-threshold runs/ham/main/sgep/seed_11/threshold.json --output runs/ham/cluster_seed_11.json
```

The default is 2,000 paired cluster resamples. Both models receive identical draws; whole groups remain together, including mixed known/unknown groups. Deployment thresholds stay fixed, while descriptive FPR95 is recalculated in each resampled curve. These intervals quantify sampling uncertainty separately from variation across training seeds.

## 6. Spatial sensitivity and mask quality

Retrained spatial controls use their masks in both training and testing. Test-only interventions instead change the input to a fixed SGEP checkpoint:

```bash
sgep evaluate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --threshold runs/ham/sgep/seed_11/calibration/threshold.json --intervention translation --output runs/ham/sensitivity/translation
```

Other interventions are `jitter` (random erosion/dilation/translation), `erosion`, `dilation`, `dropout` (20% foreground-pixel removal), and `background`. `--mask-policy random_area`, `--mask-policy displaced`, or `--fill mean` support test-only control shifts. They use the original source threshold. Interpret these as distribution-shift sensitivity; retrained controls make the spatial/capacity comparison.

For the HAM mask-quality analysis, select test images before reading predictions:

```bash
sgep select-reference --manifest data/ham/manifest_sam.csv --output data/ham/annotations.csv
```

The selection requests 100 test images each from `nv`, `bkl`, `mel`, and `akiec`, retaining all available images if a split has fewer. Have two annotators provide root-relative `annotator_a_path` and `annotator_b_path`, an adjudicated `reference_mask_path`, and `adjudicated=true/false`. Then:

```bash
sgep mask-quality --manifest data/ham/manifest_sam.csv --root data/ham/images --annotations data/ham/annotations.csv --output data/ham/quality.csv
sgep analyze --predictions runs/ham/sgep/seed_11/test/predictions.npz --threshold runs/ham/sgep/seed_11/calibration/threshold.json --mask-quality data/ham/quality.csv --output runs/ham/mask_quality
```

The analysis uses native-resolution per-image Dice, fixed low/medium/high strata at 0.65 and 0.85, known/unknown counts, actual group counts, inter-annotator agreement, adjudication counts, and annotation coverage. Both-empty masks have undefined Dice. Human reference masks are never recognition inputs or model-selection targets.

## 7. External evaluation

Create normalized PH2/HyperKvasir metadata with the fields from section 2 and real group linkage. PH2 category names are `common nevi`, `atypical nevi`, and `melanoma`; HyperKvasir uses its literal release labels. The supplied crosswalk maps categories to source-known or final-unknown roles and excludes quality/content-only labels.

```bash
sgep prepare-external --metadata data/hyper/metadata.csv --output data/hyper/manifest.csv
sgep audit-external --source data/gastro/manifest_sam.csv --source-root data/gastro/images --external data/hyper/manifest.csv --external-root data/hyper/images --output data/hyper/manifest_clean.csv
sgep masks --manifest data/hyper/manifest_clean.csv --root data/hyper/images --checkpoint weights/sam_vit_b_01ec64.pth --device cuda --output data/hyper/manifest_sam.csv
sgep evaluate --checkpoint runs/gastro/main/sgep/seed_11/checkpoint.pt --manifest data/hyper/manifest_sam.csv --root data/hyper/images --threshold runs/gastro/main/sgep/seed_11/threshold.json --output runs/hyper/sgep/seed_11
```

The overlap check compares decoded pixels and perceptual signatures with all source fitting/validation partitions and excludes detected overlapping groups. Supply `linked_source_group_id` when provider linkage identifies a source patient/procedure/sequence. Perceptual candidates are conservatively excluded at the configured Hamming cutoff; inspect the recorded nearest image and reason. Image similarity cannot establish patient independence. GastroVision and HyperKvasir share upstream image sources, so without cross-dataset provider linkage this is related-source transfer evaluation.

PH2 follows the same workflow with the HAM checkpoint. External evaluation retains the source threshold and reports shifted-known recognition/calibration separately from known-versus-semantic-unknown rejection. AUROC/OSCR require both populations.

## Configuration and implementation conventions

`configs/sgep.yaml` holds the paper defaults. All configuration keys and the few details not fully specified by the revision are documented in [implementation.md](docs/implementation.md), including OpenMax's SciPy tail fit, PostMax probability interpretation, component ablations, and the precise mask interventions. `configs/smoke.yaml` is only for small CPU checks. Every command also provides `--help`.

The test suite verifies mathematical formulas, gradient paths, category/group isolation, tied-score metrics, spatial controls, and the complete fixture pipeline. Full benchmark training, pretrained SAM inference, and the reported five-seed experiments are intended to be run with the downloaded data and supplied linkage/annotations; this repository does not include prefilled experimental results.

## Citation and license

```bibtex
@unpublished{hu_sgep,
  author = {Qinao Hu},
  title = {Spatially Grounded Evidential Prototypes for Open-Set Medical Image Recognition},
  note = {Manuscript}
}
```

Code is distributed under the [MIT license](LICENSE). Datasets and pretrained model assets retain their providers' licenses. Cite the source datasets and baseline papers when using them.
