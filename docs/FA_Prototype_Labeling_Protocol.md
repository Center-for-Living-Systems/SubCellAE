# FA Prototype Labeling Protocol

## Goal

Collect high-confidence examples of focal adhesion categories across four datasets for model training and cross-dataset comparison.

This protocol focuses on **prototype / click-based labeling**, where the labeler selects clear examples from whole-cell images.

## Datasets

1. Vinc-Pax-Zyx-Act-031125
2. pPax-Pax-Zyx-Act-072025
3. pFAK-Pax-Zyx-Act-072125
4. Vinc-Pax-Zyx-Act-022726

## Sampling Strategy

For each dataset:

* If the dataset has ≤20 images: label all images when possible.
* If the dataset has many images: start with ~20–30 representative images.
* Prefer images with clear cell morphology, good signal, and visible adhesion structures.
* Avoid images with severe artifacts, poor focus, or very low signal.

## Labeling Mode

Use the whole image view.

For each selected image, click representative examples of each available category:

1. Nascent Adhesion
2. Focal Complex
3. Focal Adhesion
4. Fibrillar Adhesion
5. No Adhesion

## Target Numbers

Per dataset, aim for:

* ~20–30 images initially
* ~30 examples per common class if available
* More targeted labeling for rare classes, especially fibrillar adhesions

Do not force every image to contain every class. Some classes may be absent.

## Rare Class Handling

Fibrillar adhesions are expected to be rare.

* Label them wherever clear examples are present.
* It is okay if rare classes are collected unevenly across images.
* Do not label uncertain examples as fibrillar just to balance the dataset.

## Labeling Principles

Select examples that are:

* visually clear
* representative of the category
* not heavily overlapping with other structures
* not ambiguous unless using the uncertain label

For prototype labeling, the goal is not to capture the full natural distribution. The goal is to collect reliable anchor examples for each class.

## Metadata to Record

For every label:

* Dataset name
* Image ID
* Patch coordinate
* FA class
* Labeler
* Labeling mode: prototype_click
* Date

## Intended Use

Prototype labels will be used mainly for:

* improving class coverage
* enriching rare FA types
* defining high-confidence examples
* training or fine-tuning models

They should be analyzed separately from random-sampled labels when evaluating real-world model performance.
