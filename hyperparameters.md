# Hyperparameter Optimisation

All three model types use **nested cross-validation** for unbiased evaluation. Hyperparameters are selected in the inner loop; the outer loop measures generalisation performance on held-out data.

## Search strategy

| | Tree models (XGBoost, RF) | MIL |
|---|---|---|
| Method | `RandomizedSearchCV` (scikit-learn) | Random sampling from grid |
| Iterations | 50 per outer fold | 20 per outer fold |
| Inner folds | 5-fold CV | 5-fold CV |
| Selection metric | R² (uniform average across targets) | R² (uniform average across targets) |
| Refit | Yes — best params refitted on full outer training set | Yes |

For XGBoost, the estimator is wrapped in `MultiOutputRegressor` (one XGB model per archetype target, fitted serially). Random Forest natively handles multi-output regression.

---

## XGBoost

Grid defined in `scripts/param_grids/xgboost_params.json`.

| Parameter | Values searched | What it controls |
|---|---|---|
| `n_estimators` | 100, 200, 300, 500, 700 | Number of boosting rounds (trees) |
| `max_depth` | 3, 4, 5, 6, 7, 8 | Maximum depth of each tree; lower = more regularisation |
| `learning_rate` | 0.005, 0.01, 0.05, 0.1, 0.2 | Shrinkage applied to each tree's contribution |
| `subsample` | 0.6, 0.7, 0.8, 0.9, 1.0 | Fraction of training samples used per tree |
| `colsample_bytree` | 0.4, 0.5, 0.6, 0.7, 0.8, 1.0 | Fraction of features sampled per tree |
| `min_child_weight` | 1, 3, 5, 7, 10 | Minimum sum of instance weights in a leaf; higher = more conservative splits |
| `gamma` | 0, 0.05, 0.1, 0.2, 0.5 | Minimum loss reduction required to make a split |
| `reg_alpha` | 0, 0.001, 0.01, 0.1, 1.0, 5.0 | L1 regularisation on leaf weights |
| `reg_lambda` | 0.01, 0.1, 1.0, 5.0, 10.0 | L2 regularisation on leaf weights |

**Grid size:** 5 × 6 × 5 × 5 × 6 × 5 × 5 × 6 × 5 = 3,375,000 combinations; 50 are randomly drawn per outer fold.

---

## Random Forest

Grid defined in `scripts/param_grids/rf_params.json`.

| Parameter | Values searched | What it controls |
|---|---|---|
| `n_estimators` | 100, 200, 300, 500, 700 | Number of trees in the forest |
| `max_depth` | None, 5, 10, 15, 20, 30 | Maximum depth per tree (`None` = grow until pure leaves) |
| `min_samples_split` | 2, 5, 10, 20 | Minimum samples required to split an internal node |
| `min_samples_leaf` | 1, 2, 4, 8 | Minimum samples required at a leaf node |
| `max_features` | "sqrt", "log2", 0.2, 0.3, 0.5, 0.7 | Number of features considered at each split |
| `bootstrap` | True, False | Whether to bootstrap samples when building trees |
| `max_samples` | None, 0.7, 0.8, 0.9 | Fraction of samples drawn per tree (only when `bootstrap=True`) |

**Grid size:** 5 × 6 × 4 × 4 × 6 × 2 × 4 = 23,040 combinations; 50 are randomly drawn per outer fold.

---

## Attention MIL

Grid defined in `scripts/param_grids/mil_params.json`.

The MIL network uses gated attention pooling (Ilse et al., 2018). The architecture is:

```
cells → [Linear → LayerNorm → ReLU] × 2  (encoder)
      → tanh(V·h) ⊙ σ(U·h) → softmax    (attention gate)
      → weighted sum                      (bag embedding)
      → Linear → Softmax                  (archetype prediction)
```

| Parameter | Values searched | What it controls |
|---|---|---|
| `hidden_dim` | 64, 128, 256, 512 | Width of the per-cell encoder MLP |
| `attention_dim` | 32, 64, 128 | Width of the attention gate matrices (V, U) |
| `dropout` | 0.1, 0.2, 0.3, 0.5 | Dropout rate applied in the encoder |
| `lr` | 0.0001, 0.0005, 0.001, 0.005 | Adam learning rate |
| `weight_decay` | 0.0, 0.0001, 0.001, 0.01 | Adam L2 weight decay |
| `batch_size` | 8, 16, 32 | Number of bags (samples) per training step |
| `epochs` | 50, 100, 150, 200 | Training epochs per candidate configuration |

**Grid size:** 4 × 3 × 4 × 4 × 4 × 3 × 4 = 9,216 combinations; 20 are randomly drawn per outer fold (fewer than tree models due to training cost).

Training uses the **Adam** optimiser with **cosine-annealing** learning rate decay. Cells are subsampled to `max_cells_per_bag` (default 2,000) per sample at each epoch.

---

## Selection and evaluation

At the end of each outer fold the best inner-loop configuration is refitted on the full outer training set and evaluated on the held-out test fold. The reported metrics (R², RMSE) are therefore **out-of-fold** estimates on data that never influenced model selection.
