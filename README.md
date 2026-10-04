# SGEP

PyTorch implementation of **Spatially Grounded Evidential Prototypes for Open-Set Medical Image Recognition** by Qinao Hu.

SGEP uses separate global and region-of-interest encoders, with class prototypes that turn feature distances into Dirichlet evidence. ROI consistency and background suppression guide where that evidence comes from.

## Installation

Python 3.10 or later is required. Create and activate a virtual environment:

```bash
git clone https://github.com/Hqa77880011/SGEP.git
cd SGEP
python -m venv .venv
```

Use `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell. For a CPU installation:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

For CUDA, install matching PyTorch and torchvision builds from the [PyTorch installer](https://pytorch.org/get-started/locally/) before installing SGEP. Add SAM for automatic mask generation:

```bash
python -m pip install -e ".[sam]"
```

```bash
python -m pytest
sgep smoke
```

`smoke` runs training, checkpoint loading, calibration, evaluation, and analysis on small synthetic images with a tiny CPU encoder. Use `--output runs/smoke` to keep its outputs. All commands provide `--help`.

## Data preparation

Download and extract the datasets from their providers:

| Dataset | Source | Evaluation role |
| --- | --- | --- |
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) | Source dermoscopy |
| GastroVision | [Repository](https://github.com/DebeshJha/GastroVision), [OSF](https://osf.io/84e7f/) | Source endoscopy |
| PH2 | [Provider](https://www.fc.up.pt/addi/ph2%20database.html) | External dermoscopy |
| HyperKvasir | [Repository](https://github.com/simula/hyper-kvasir), [OSF](https://osf.io/mh9sj/) | External endoscopy |

Keep datasets, masks, weights, and run outputs outside version control. Category assignments are in [roles.csv](sgep/resources/roles.csv); external category mappings are in [external_mapping.csv](sgep/resources/external_mapping.csv).

For HAM10000, put the extracted images under `data/ham/images` and the release metadata at `data/ham/HAM10000_metadata.csv`:

```bash
sgep prepare-ham --metadata data/ham/HAM10000_metadata.csv --images data/ham/images --output data/ham/manifest.csv
```

Images are found recursively. Splitting uses `lesion_id`, or complete patient identifiers when supplied. Reconciled `group_id` and `group_type` columns can also be provided.

GastroVision preparation requires a linkage CSV with `image_id,group_id,group_type,source_cohort`. Use patient, procedure, or sequence identifiers from the data provider; the public class metadata does not contain them. `image_id` accepts a filename with or without its extension.

```bash
sgep prepare-gastro --metadata data/gastro/GastroVision_metadata.csv --images data/gastro/images --linkage data/gastro/linkage.csv --output data/gastro/manifest.csv
```

Known groups use a target 70/10/5/15% split across training, checkpoint selection, threshold calibration, and testing. Proxy-unknown groups are split 2:1 across selection and calibration; final-unknown groups are reserved for testing. The split seed is 2026. Groups stay together, with final-unknown status taking precedence when a group spans category roles.

For other layouts, use `sgep prepare --metadata data/metadata.csv --output data/manifest.csv`, with `--roles` for a custom category map. Normalized metadata has these columns:

| Columns | Values |
| --- | --- |
| `dataset`, `release`, `image_id` | Dataset, release, and unique image identifiers |
| `relative_path`, `category` | Image path relative to the data root and source category label |
| `group_id`, `group_type` | Patient, lesion, procedure, or sequence linkage |
| `source_cohort` | Acquisition provenance, or release namespace if finer provenance is unavailable |
| `mask_path` | Binary ROI path; may be empty before mask generation |
| `exclusion_reason` | Optional prespecified exclusion |

Preparation adds `mapped_class`, `role`, and `split`. Group IDs are local to each dataset and must link repeated acquisitions across source cohorts.

## ROI masks

Download the [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `weights/sam_vit_b_01ec64.pth`:

```bash
sgep masks --manifest data/ham/manifest.csv --root data/ham/images --checkpoint weights/sam_vit_b_01ec64.pth --device cuda --output data/ham/manifest_sam.csv
```

Use `--device cpu` for CPU inference. Repeat with the GastroVision manifest and image root. Masks are saved under `masks/sam_vit_b/<dataset>/<image_id>.png` within the image root. The output manifest records their paths, and `manifest_sam_mask_log.csv` records candidate scores and full-image fallbacks. To use existing masks, set `mask_path` to a binary mask at the image's original dimensions.

SAM stays frozen. Its automatic prompting, candidate selection, and preprocessing settings are documented in [implementation.md](docs/implementation.md).

## Training and evaluation

The default configuration uses two ImageNet-initialized ResNet18 encoders, 256-dimensional fused features, AdamW, a batch size of 32, and 100 epochs. Known training images update the model; known and proxy-unknown selection images choose the checkpoint with the highest OSCR.

```bash
sgep train --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --seed 11 --output runs/ham/sgep/seed_11
sgep calibrate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/sgep/seed_11/calibration
sgep evaluate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --threshold runs/ham/sgep/seed_11/calibration/threshold.json --output runs/ham/sgep/seed_11/test
sgep analyze --predictions runs/ham/sgep/seed_11/test/predictions.npz --calibration runs/ham/sgep/seed_11/calibration/calibration_predictions.npz --threshold runs/ham/sgep/seed_11/calibration/threshold.json --near-category mel --output runs/ham/sgep/seed_11/analysis
```

For GastroVision, change the manifest and root, and use `--near-category "Gastric Polyps"`. If a manifest contains multiple datasets, add `--dataset HAM10000` or `--dataset GastroVision`. `--device auto` selects CUDA when available. Each training run needs a new output directory.

The default threshold accepts at least 95% of known calibration images. Other calibration rules are `--rule coverage90`, `coverage97`, and `youden`. Youden maximizes correct-known acceptance minus proxy false acceptance. Thresholds are fixed before test evaluation.

| Output | Contents |
| --- | --- |
| `checkpoint.pt`, `config.json`, `history.csv` | Selected model, class order, post-hoc state, settings, losses, and selection OSCR |
| `selection_predictions.npz` | Scores from the selected epoch |
| `calibration_predictions.npz`, `threshold.json` | Calibration scores and rejection threshold |
| `predictions.npz`, `predictions.csv`, `metrics.json` | Test scores, probabilities, acceptance decisions, and metrics |
| Analysis directory | ROC, OSCR, reliability, uncertainty, and evidence plots; threshold and subgroup tables |

Larger scores indicate unknown images; `score <= threshold` is accepted. AUROC treats unknowns as positives. OSCR measures correct-known acceptance against unknown false acceptance across thresholds. `FPR95` is read from the test curve at 95% known acceptance; `unknown_fpr` uses the saved calibration threshold. Accuracy, ECE, AUROC, OSCR, coverage, and FPR are percentages. NLL and multiclass Brier are dimensionless. Metrics are `null` when a required population is absent.

## Baselines and experiment suites

Available methods are `softmax`, `maxlogit`, `openmax`, `energy`, `cac`, `arpl`, `postmax`, `edl`, `prototype`, `roi_guided`, and `sgep`. Add `--method energy`, for example, to the training command and use the same calibration and evaluation steps. Method objectives, rejection scores, and baseline adaptations are listed in [implementation.md](docs/implementation.md).

Hyperparameter search uses 24 configurations and selection seeds 11 and 22 by default:

```bash
sgep search --method sgep --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/search/sgep
sgep train --config runs/ham/search/sgep/selected.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --seed 11 --output runs/ham/tuned_sgep/seed_11
```

Search writes trial records and `selected.yaml`, using the selection split only. Repeat with each method to tune baselines separately. `--count` and `--seeds` change the budget; `--plan` writes the proposed runs without training. SGEP starts with the supplied configuration and samples the remaining trials from its grid. Smaller baseline grids allow repeated draws.

Suites train, calibrate, and evaluate paired seeds 11, 22, 33, 44, and 55. Use `main` for method comparisons, `ablation` for the seven component variants, `factorial` for ROI/background loss combinations, `spatial` for retrained mask and fill controls, or `all` for every suite:

```bash
sgep suite --suite all --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/paper --plan
sgep suite --suite main --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/main
sgep suite --suite ablation --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/ablation
sgep suite --suite factorial --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/factorial
sgep suite --suite spatial --config configs/sgep.yaml --manifest data/ham/manifest_sam.csv --root data/ham/images --output runs/ham/spatial
sgep compare --runs runs/ham/main runs/gastro/main --output runs/comparison
sgep compare --runs runs/ham/factorial runs/gastro/factorial --output runs/factorial_analysis
```

Suites use the supplied SGEP configuration and defined method-specific changes. For separately tuned baselines, train with each method's selected YAML. Each suite writes a run plan, `results.csv`, and per-run checkpoints, histories, calibration files, `test_predictions.npz`, and metrics. `compare` produces mean/sample-SD tables, figures, and paired-seed comparisons. Supplying both datasets applies Holm correction to the four primary AUROC/FPR95 tests. Factorial comparisons include both main effects and their interaction.

For paired group-bootstrap intervals, run this for each reporting seed:

```bash
sgep bootstrap --baseline runs/ham/main/roi_guided/seed_11/test_predictions.npz --proposed runs/ham/main/sgep/seed_11/test_predictions.npz --baseline-threshold runs/ham/main/roi_guided/seed_11/threshold.json --proposed-threshold runs/ham/main/sgep/seed_11/threshold.json --output runs/ham/cluster_seed_11.json
```

The default is 2,000 paired resamples of whole groups. Both models receive the same draws, and calibration thresholds stay fixed. These intervals measure sampling uncertainty; paired-seed statistics measure variation across training runs.

## Spatial sensitivity and mask quality

Spatial suites retrain under each mask/fill control. To change masks only at test time, evaluate a fixed checkpoint with an intervention:

```bash
sgep evaluate --checkpoint runs/ham/sgep/seed_11/checkpoint.pt --manifest data/ham/manifest_sam.csv --root data/ham/images --threshold runs/ham/sgep/seed_11/calibration/threshold.json --intervention translation --output runs/ham/sensitivity/translation
```

Other interventions are `jitter`, `erosion`, `dilation`, `dropout`, and `background`. Test-only controls also accept `--mask-policy random_area`, `--mask-policy displaced`, or `--fill mean`. They retain the source threshold and measure sensitivity to changed inputs.

Select HAM10000 images for independent mask annotation before inspecting predictions:

```bash
sgep select-reference --manifest data/ham/manifest_sam.csv --output data/ham/annotations.csv
```

The selection takes up to 100 test images from each of `nv`, `bkl`, `mel`, and `akiec`. Fill in root-relative `annotator_a_path`, `annotator_b_path`, and `reference_mask_path`, plus `adjudicated=true/false`, after annotation:

```bash
sgep mask-quality --manifest data/ham/manifest_sam.csv --root data/ham/images --annotations data/ham/annotations.csv --output data/ham/quality.csv
sgep analyze --predictions runs/ham/sgep/seed_11/test/predictions.npz --threshold runs/ham/sgep/seed_11/calibration/threshold.json --mask-quality data/ham/quality.csv --output runs/ham/mask_quality
```

Outputs include native-resolution Dice, mask-quality strata with cutoffs at 0.65 and 0.85, inter-annotator agreement, annotation coverage, and image/group counts. Both-empty masks have undefined Dice. Reference annotations are used only for mask-quality evaluation.

## External evaluation

Prepare normalized PH2 or HyperKvasir metadata with the columns above and group linkage. PH2 labels are `common nevi`, `atypical nevi`, and `melanoma`; use the release labels for HyperKvasir. The supplied crosswalk assigns known/unknown roles and excludes quality/content-only categories.

```bash
sgep prepare-external --metadata data/hyper/metadata.csv --output data/hyper/manifest.csv
sgep audit-external --source data/gastro/manifest_sam.csv --source-root data/gastro/images --external data/hyper/manifest.csv --external-root data/hyper/images --output data/hyper/manifest_clean.csv
sgep masks --manifest data/hyper/manifest_clean.csv --root data/hyper/images --checkpoint weights/sam_vit_b_01ec64.pth --device cuda --output data/hyper/manifest_sam.csv
sgep evaluate --checkpoint runs/gastro/main/sgep/seed_11/checkpoint.pt --manifest data/hyper/manifest_sam.csv --root data/hyper/images --threshold runs/gastro/main/sgep/seed_11/threshold.json --output runs/hyper/sgep/seed_11
```

The overlap audit excludes groups matched by decoded pixels, perceptual similarity, or supplied `linked_source_group_id` values. Review its recorded matches and exclusion reasons. GastroVision and HyperKvasir share upstream image sources; without provider linkage across datasets, this comparison measures related-source transfer rather than patient-independent transfer.

Use the same workflow for PH2 with the HAM10000 checkpoint. External evaluation keeps the source threshold and reports shifted-known recognition/calibration alongside semantic-unknown rejection. AUROC and OSCR require both known and unknown images.

## Configuration and citation

[configs/sgep.yaml](configs/sgep.yaml) contains the training defaults. [Implementation settings](docs/implementation.md) describes every configuration key, preprocessing, loss terms, baseline adaptations, and choices for details left unspecified in the manuscript. [configs/smoke.yaml](configs/smoke.yaml) is for small CPU checks.

```bibtex
@unpublished{hu_sgep,
  author = {Qinao Hu},
  title = {Spatially Grounded Evidential Prototypes for Open-Set Medical Image Recognition},
  note = {Manuscript}
}
```

The code uses the [MIT license](LICENSE). Datasets and pretrained weights retain their providers' licenses. Cite the source datasets and baseline papers when using them.
