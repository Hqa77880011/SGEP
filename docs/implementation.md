# Implementation settings

## Model and loss

The two ResNet18 encoders are independent copies initialized from the same `ResNet18_Weights.IMAGENET1K_V1` weights. Their 512-dimensional pooled features are concatenated, projected by a bias-free Xavier-initialized linear layer, and divided by their L2 norm plus `1e-7`. The fused dimension is 256.

One unconstrained prototype represents each known class. It is initialized from unaugmented known training features of the initial network. Squared Euclidean distances produce `e_c = softplus(beta - d_c)`, `alpha_c = e_c + 1`, `p_c = alpha_c / sum(alpha)`, and `u = K / sum(alpha)`. The shared scalar `beta` starts at 3. Acceptance uses `u <= threshold`.

The EDL loss is expected cross-entropy under the Dirichlet plus KL to a uniform Dirichlet, after removing true-class evidence from the KL target. Prototype loss combines the true-class squared distance and mean incorrect-class margin penalties. ROI consistency is the mean of both directed **Dirichlet** KL terms. Background suppression is `mean(log(1 + sum(e_background)))`. Both encoders are reused for the perturbed/background passes, with gradients through each view.

The evidence-source ratio is `E_full / (E_full + E_background + 1e-7)`. The normalized top-two evidence margin is `(e_top1 - e_top2) / (E_full + 1e-7)`. These compare nonlinear full-model outputs. Per-image uncertainty is calculated before aggregation.

## Configuration keys

Configurations are flat YAML mappings. Unspecified fields use the defaults in `sgep/config.py`; unknown fields raise an error.

| Key | Default | Meaning |
| --- | --- | --- |
| `method`, `backbone` | `sgep`, `resnet18` | Recognition head and encoder (`tiny` is for checks) |
| `pretrained`, `view` | `true`, `dual` | ImageNet initialization and `global`/`roi`/`dual` views |
| `feature_dim`, `beta_init` | 256, 3 | Projection dimension and evidence offset initialization |
| `image_size`, `batch_size`, `workers` | 224, 32, 0 | Recognition input size and loading settings |
| `epochs`, `seed`, `device`, `threads` | 100, 11, `auto`, 0 | Training length, RNG seed, device, CPU thread override (0 keeps runtime default) |
| `lr`, `weight_decay` | `1e-4`, `1e-4` | AdamW settings; betas `(0.9,0.999)`, epsilon `1e-8` |
| `warmup_epochs`, `start_lr`, `minimum_lr` | 5, `1e-5`, `1e-6` | Linear warmup endpoints, then cosine to the final epoch |
| `lambda_proto`, `margin` | 0.1, 1 | Prototype regularizer and incorrect-class squared-distance margin |
| `lambda_roi`, `lambda_bg` | 0.05, 0.05 | Spatial loss coefficients |
| `lambda_kl_max`, `kl_ramp_epochs` | 0.01, 10 | KL starts at zero in epoch 1 and reaches its maximum at epoch 11 |
| `mask_policy`, `mask_seed`, `fill` | `aligned`, 2026, `black` | Spatial control and `black`/ImageNet `mean` fill |
| `energy_temperature` | 1 | Energy score temperature; search `{0.5,1,2}` |
| `openmax_tail_size`, `openmax_rank`, `openmax_distance` | 20, 3, `eucos` | Upper tail, top-class recalibration count (clamped to K), distance (`euclidean`, `cosine`, `eucos`) |
| `cac_anchor`, `lambda_cac` | 10, 0.1 | Fixed anchor magnitude and anchor-loss coefficient |
| `arpl_weight`, `arpl_temperature` | 0.1, 1 | Reciprocal-point margin coefficient and classification temperature |

For single-stream baselines, `train --method` and the suite set `view=global`; ROI-guided uses `view=roi`; SGEP uses `view=dual`. When editing YAML directly, set the view explicitly. Baseline optimizer settings can be supplied in their own YAML files.

## Input and spatial controls

Images use bilinear resize and masks use nearest-neighbor resize. Training applies horizontal flip with probability 0.5, rotation in `[-15,15]` degrees, and brightness/contrast factors in `[0.8,1.2]`. Geometry is shared with the binary mask; both model views use the same transformed RGB image. No vertical flip, class reweighting, oversampling, mixup, or CutMix is used.

Masking occurs in RGB `[0,1]` before ImageNet normalization. Black fill therefore has nonzero normalized values. Mean-fill uses `(0.485,0.456,0.406)`. Rotation padding is black in the full image; the ROI fill remains controlled by `fill`.

ROI training perturbations choose erosion, dilation, or translation uniformly. Morphology uses a 3-by-3 square; translation draws each integer offset from `[-5,5]` and pads with zeros. Empty perturbed masks revert to the original and are counted in `history.csv`.

Random-area masks select exactly the same foreground-pixel count uniformly without replacement at recognition resolution. The generator seed combines 2026 with the stable CRC32 of the image ID. Displaced masks choose the minimum-overlap cyclic shift among the eight combinations of horizontal/vertical quarter-image offsets, resolving ties lexicographically in `(dy,dx)`. Displacement preserves area and shape on a torus and can split an object at image borders.

The revision does not specify exact test-only dropout topology. Here, dropout removes each foreground pixel with probability 0.2, reverting an empty outcome. Translation, erosion, dilation and `jitter` use the training perturbation definitions. These are explicit sensitivity interventions on a fixed checkpoint. Spatial suites separately retrain under each mask/fill policy.

The component sequence is implemented as global CE, ROI CE, dual-view CE, dual CE with prototype regularization, fused evidential prototypes without prototype regularization, fused evidential prototypes with prototype regularization, and full SGEP. This fixes the heads and loss switches for transitions not fully specified in the revision. The four factorial variants keep prototype/evidential objectives fixed while setting each spatial coefficient to 0 or 0.05.

## Fixed SAM configuration

SAM ViT-B uses the official `sam_vit_b_01ec64.pth` checkpoint and its native longest-side resize to 1024, normalization, and padding. It is frozen and is not medically fine-tuned. Native mask threshold is 0.0.

Automatic generation uses `points_per_side=32`, `points_per_batch=64`, `pred_iou_thresh=0.88`, `stability_score_thresh=0.95`, `stability_score_offset=1.0`, `box_nms_thresh=0.7`, `crop_n_layers=0`, `crop_nms_thresh=0.7`, `crop_overlap_ratio=512/1500`, `crop_n_points_downscale_factor=1`, and `min_mask_region_area=0`.

Candidates with 5–80% foreground area are ranked by predicted IoU times stability, then smaller area, then generator order. No qualifying candidate gives a counted full-image fallback. No post-selection morphology or manual correction is applied. Reference annotations are not prompts.

## Baseline adaptations

These heads use the same ResNet18 family and training partitions. Single-stream CE heads use an unnormalized projected feature; prototype and dual-view heads use normalized features.

| Method | Objective / larger-is-unknown score |
| --- | --- |
| Softmax / ROI-guided / dual CE | Cross-entropy; `1 - max(softmax(logits))` |
| MaxLogit | Cross-entropy; `-max(logits)` |
| Energy | Cross-entropy; `-T * logsumexp(logits/T)` |
| Evidential | Linear-head softplus evidence and EDL; `K / sum(alpha)` |
| Prototype | Cross-entropy on negative squared distances plus prototype regularization; minimum squared distance |
| CAC | CE on negative Euclidean distance to fixed `anchor * I`, plus true-class anchor distance; `min(distance * (1 - softmin(distance)))` |
| ARPL | CE on reciprocal distance (`mean squared L2 - dot product`), plus `relu(true_L2_distance - radius + 1)`; negative maximum reciprocal distance |
| OpenMax | CE encoder, class means and Weibull tails from correct known training activations; K+1 posterior's unknown probability |
| PostMax | CE encoder, pooled generalized-Pareto fit to `max(logits)/L2(feature)` on correct known training images; `1 - max(class CDF support)` |

CAC inference uses correctly classified training activation means, following the [authors' evaluation code](https://github.com/dimitymiller/cac-openset). ARPL follows the [reciprocal-point objective](https://github.com/iCGY96/ARPL) without a confusing-sample generator; its learnable radius starts at zero.

OpenMax uses [the original recalibration rule](https://github.com/abhijitbendale/OSDN), with SciPy `weibull_min` upper-tail maximum-likelihood fitting and fixed location zero in place of libMR. `eucos` is Euclidean distance divided by 200 plus cosine distance. Its known probabilities are conditional K-class probabilities for the common ECE/NLL/Brier definitions; the rejection score remains the K+1 unknown probability.

PostMax follows [Algorithm 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01043.pdf) with SciPy `genpareto.fit`. CDF class support need not sum to one, so the common calibration metrics use the underlying CE softmax posterior; rejection uses the CDF support. Post-hoc state is never fit from proxy or final-unknown images.

When an early epoch lacks enough correct training examples or has a degenerate EVT tail, its post-hoc fit is recorded as unavailable and that epoch cannot become the selected checkpoint. If every epoch is unavailable, training fails with the recorded fit reason. A missing state is not replaced with invented parameters.

## Metrics and statistics

OSCR processes unique tied scores together and includes reject-all/accept-all endpoints. The accept-all correct-recognition rate equals known closed-set accuracy. FPR95 accepts ties at the smallest observed threshold reaching at least 95% known acceptance.

ECE uses 15 equal-width top-label confidence bins, with one included in the final bin. NLL clips probabilities below at `1e-12`. Brier sums squared class errors without dividing by K. Images receive equal weights in point estimates; cluster resampling quantifies sampling uncertainty.

The five-seed comparison uses proposed-minus-baseline paired differences, sample standard deviations, a two-sided Student t test, and a t-based 95% interval. A constant difference has no estimable t-test variance and its p-value is undefined. The four primary comparisons use one Holm family when both datasets are aggregated.

Bootstrap resamples known-only, unknown-only, and mixed-role group strata, preserving whole clusters and pairing the two models' draws. Its percentile limits are 2.5% and 97.5%. Evidence diagnostics use per-image quantities, empirical image-weighted means, and group resampling. Annotation coverage and independent-group counts are exported separately from mask-quality performance.
