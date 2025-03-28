# Compositional Generalization via Factorized Representations

This repository contains the code to reproduce the experiments and results from our paper: "**Compositional Generalization Requires More Than Disentangled Representations**" accepted at ICML 2025. The experiments aim to explore how disentangled or factorized representations in generative models impact compositional generalization and extrapolation to out-of-distribution (OOD) regions. We provide implementations of the main experiments described in the paper and supporting experiments for reproducibility.

## Folder Structure

The repository contains the following folders, each corresponding to a set of experiments:

- **`gaussian_exps/`**: The main experiments presented in the paper, focusing on 2D Gaussian bump generation tasks using both CNN and MLP architectures. This folder contains the experiments with **bump-based encoding** and **scalar-based encoding** inputs.
- **`kernel_exps/`**: Supporting experiments involving kernel-based perspectives on factorization and compositional generalization. This folder explores how learned kernels can provide insight into the failure of models to generalize compositional tasks.
- **`rotation_exps/`**: Supporting experiments where we explore model performance on rotation-based tasks using MNIST digits. This folder investigates the failure of compositional generalization in a simple task of rotating images.
- **`scaling_exps/`**: Experiments analyzing data scaling and the impact of dataset size on model performance, particularly how smaller datasets can enhance generalization.
- **`encoder_exps/`**: Supporting experiments on different encoder architectures, comparing how various designs affect the learned representations' ability to generalize.

## Basic Summary of Experiments

### Main Experiment: Gaussian Bump Generation (`gaussian_exps/`)
This set of experiments explores the ability of neural networks to learn 2D Gaussian "bump" generation tasks with compositional generalization. The networks are trained on images where a Gaussian bump is located at a 2D coordinate \((x, y)\), with the goal of generating bumps in unseen (OOD) regions. We experiment with two types of network inputs:
1. **Bump-based encoding**: where the \(x\) and \(y\) coordinates are encoded as Gaussian bumps in the input space.
2. **Scalar-based encoding**: where the \(x\) and \(y\) coordinates are represented as scalar values.

The main findings show that the model's failure to generalize to OOD regions is not solely due to disentangled representations but is significantly impacted by the network's ability to maintain or induce factorization throughout the architecture, especially in the output (pixel) space.

The **gaussian_exps** folder includes:
- Experiments using **CNN** and **MLP** models.
- Model training with **disentangled** (factorized) input representations.
- Tests on **compositional generalization** and extrapolation capabilities of models when forced to maintain factorization through architectural regularization.

### Supporting Experiment: Kernel-based Analysis (`kernel_exps/`)
These experiments provide insights into how the learned kernel of a neural network reflects its capacity to generalize. By analyzing the kernel matrix, we investigate the role of memory (i.e., memorization strategies) and factorization in the network's learning process. These experiments also link kernel factorization with the ability to compose novel data combinations.

### Supporting Experiment: Rotation Task (`rotation_exps/`)
This experiment tests the network's ability to generalize on an image rotation task using the MNIST dataset. Here, models trained with different input encodings are tested on their ability to extrapolate to unseen rotations. Results show that models fail to generalize to OOD rotations due to memorization, and the phenomenon observed in the Gaussian experiments persists in more straightforward tasks.

### Supporting Experiment: Scaling Analysis (`scaling_exps/`)
Experiments in this folder analyze how the scaling of data (e.g., increasing image size or dataset size) affects compositional generalization. Results show how certain datasets can enhance model performance and improve extrapolation through specific regularization techniques or by adding independent factors of variation to the training data.

### Supporting Experiment: Encoder Architectures (`encoder_exps/`)
In this experiment, we explore different encoder architectures and their impact on the learned representations' factorization and compositional generalization. We compare the performance of different encoding schemes (e.g., bump-based vs. scalar-based encoding) and their effects on generalization.

## Environment Setup

To install the required packages for running the experiments, you can use the provided `environment.yaml` file. This file sets up the necessary environment for running the code in a conda environment. Here’s how you can set it up:

### Install Conda (if you don't have it)
If you don’t have conda installed, follow these instructions to install Anaconda or Miniconda:
- [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)
- [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

### Create the environment
After installing Conda, create the environment from the `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

### Activate the environment
Once the environment is created, activate it using:

```bash
conda activate compositional-gen
```
