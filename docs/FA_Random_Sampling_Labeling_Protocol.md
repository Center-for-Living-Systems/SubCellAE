# FA Random-Sampling Labeling Protocol

## Goal

Estimate the natural distribution of focal adhesion categories and provide an unbiased dataset for model evaluation and cross-dataset comparison.

This protocol focuses on **random patch labeling**, where patches are sampled automatically and each presented patch receives a label.

## Datasets

1. Vinc-Pax-Zyx-Act-031125
2. pPax-Pax-Zyx-Act-072025
3. pFAK-Pax-Zyx-Act-072125
4. Vinc-Pax-Zyx-Act-022726

## Sampling Strategy

For each dataset:

- Randomly sample patches from cell regions.
- Sampling should be independent of FA morphology.
- Include patches from multiple images to capture image-to-image variability.
- Avoid manually selecting "good examples" during sampling.

## Labeling Mode

Each task presents:

- the cropped patch
- the corresponding whole-cell image with patch location indicated

Assign one label to every patch:

1. Nascent Adhesion
2. Focal Complex
3. Focal Adhesion
4. Fibrillar Adhesion
5. No Adhesion

## Target Numbers

Per dataset, aim for:

- ~20–30 images contributing patches
- ~150–300 randomly sampled patches initially
- Additional patches can be added if class frequencies are poorly estimated

Class balance is **not required**. The sampled distribution should reflect the underlying dataset.

## Rare Class Handling

Fibrillar adhesions may be rare.

- Label rare classes when encountered.
- Do not oversample or relabel patches to artificially balance categories.
- The observed frequency of each class is part of the result.

## Labeling Principles

For each patch:

- Use both the patch and image context when assigning labels.
- Select the category that best represents the central structure in the patch.
- Use the same criteria across all datasets.
- If a patch is difficult to classify, choose the closest category rather than skipping it.

For random sampling, the goal is to characterize the full distribution of FA morphologies, including transitional and ambiguous examples.

## Metadata to Record

For every label:

- Dataset name
- Image ID
- Patch coordinate
- FA class
- Labeler
- Labeling mode: random_sample
- Date

## Intended Use

Random-sampled labels will be used mainly for:

- estimating class frequencies
- evaluating model performance
- measuring cross-dataset generalization
- comparing latent-space organization across datasets

Because sampling is unbiased, these labels should be considered the primary benchmark for assessing real-world model performance.

---

**Prototype labeling asks "What are the clearest examples of each class?"**

**Random labeling asks "What does the dataset actually contain?"**
