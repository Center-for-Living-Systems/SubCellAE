FA Label-Efficiency Benchmark Plan
Liya Ding
2026.08.25
Questions to answer
1. For a given dataset/day, how does adhesion vs. no-adhesion classification performance change with the number of labeled patches?
2. How many labels are needed to reach ~90% balanced accuracy?
3. Does SupCon-AE reduce the labeling requirement compared with CellProfiler- and ilastik-style classical features?

Scope
We will use Batch 2 (Prototype) labels only. Batch 1 was collected with a different patch-selection/labeling procedure and will not be included. The goal is within-dataset/day label efficiency; cross-dataset/day generalization is outside this benchmark.

Data and labels
Batch 2 summary:
Dataset 1 Control: 539 labels / 4 images (342 no-adhesion, 197 adhesion).
Dataset 1 Y-compound: 685 labels / 14 images (428 no-adhesion, 257 adhesion).
Combined Dataset 1: 1,224 labels / 18 images (770 no-adhesion, 454 adhesion).
Dataset 2 Control: 211 labels / 4 images (60 no-adhesion, 151 adhesion).
Dataset 3 Control: 261 labels / 4 images, all adhesion, so it cannot support the binary benchmark.

The current benchmark therefore uses Dataset 1 and Dataset 2 independently. Adhesion sub-classes are collapsed into adhesion vs. no adhesion.

Open concern: Should Dataset 1 Control and Dataset 1 Y-compound be treated as two separate datasets for this benchmark, or combined as one Dataset 1 because they were collected on the same imaging day/setup?

Cross-validation and label sampling
Each CV fold will use 80% of labeled patches for training and 20% as the held-out test set. Within a dataset, training patches can come from all images; only patches assigned to the current test fold are excluded. Across five folds, every labeled patch serves as test data once.

Within each 80% training pool, we will simulate different annotation budgets by randomly sampling increasing numbers of labeled patches, for example 20, 50, 100, 200, 300, 500, and the maximum available. Training samples will be approximately balanced between adhesion and no adhesion where possible.

For each label count, the random training-label sampling will be repeated five times. This gives 5 CV folds × 5 sampling repeats = 25 evaluations for each dataset, feature method, and label count.

Feature extraction methods
1. CellProfiler-style handcrafted features
Classical patch-level intensity and Haralick/GLCM texture features, including multiple spatial distances/scales (e.g., 1, 2, and 4 pixels). Expected feature dimension: ~40–50. Segmentation-dependent object morphology features will not be included.

2. ilastik-style multiscale features
Multiscale Gaussian, LoG, gradient, DoG, structure-tensor, and Hessian filter responses (approximately five scales spanning fine to broader patch structure), followed by simple patch-level statistical pooling. Expected feature dimension: ~80.

3. Supervised Contrastive Autoencoder (SupCon-AE) features
SupCon-AE learns image features using only the sampled training labels for that run. Training combines two objectives: (1) normalized L1 reconstruction loss, which preserves image information while reducing the dominance of high-intensity pixels; and (2) supervised contrastive loss, which pulls augmented views and patches with the same adhesion/no-adhesion label together while separating different-label patches. We will use z_recon = 32 for the main benchmark and test 64 as a secondary setting, keeping the learned representation in a similar dimensional range to the classical feature sets.

Classification and evaluation
Downstream classifier: LightGBM (LGBM) will be used as the primary classifier for all three feature representations so the comparison focuses on the feature extractor. Logistic regression can be included as a simple sensitivity check; an MLP is not part of the primary benchmark.

Primary metric: balanced accuracy (average recall across adhesion and no-adhesion classes).
Secondary metrics: adhesion recall, no-adhesion recall, and precision.

For SupCon-AE, the held-out test fold must not be used during representation learning or classifier training.

Main output
For Dataset 1 and Dataset 2 separately, plot balanced accuracy versus number of labeled training patches, with three curves: CellProfiler-style, ilastik-style, and SupCon-AE. A 90% reference line will show the target performance level.

The main benchmark result will be the minimum number of labels needed for each feature method to reach approximately 90% balanced accuracy, together with the variation across CV folds and repeated label sampling.

Interpretation
This benchmark is intended to answer a practical annotation-effort question: for a given dataset/day, how many patches does a user need to label before a reliable adhesion vs. no-adhesion classifier can be trained?


