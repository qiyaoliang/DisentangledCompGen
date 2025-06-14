# Compositional Generalization via Forced Rendering of Disentangled Latents

This repository contains code to reproduce the experiments and results from our ICML 2025 paper: **"Compositional Generalization via Forced Rendering of Disentangled Latents"**. The experiments investigate why disentangled (factorized) latent representations alone are insufficient for compositional generalization and explore methods to achieve robust extrapolation in generative models by forcing these representations into explicit rendering in the output (pixel) space.

## Folder Structure

The repository is structured as follows:

- **`gaussian_exps/`**: Main experiments on 2D Gaussian bump generation tasks using CNN and MLP architectures. Explores the impact of different input encodings (**bump-based**, **ramp-based**, **1-hot**) on compositional generalization.
- **`kernel_exps/`**: Supporting experiments employing a kernel-based perspective, analyzing how learned kernels reflect memorization versus compositional generalization strategies in neural networks.
- **`rotation_exps/`**: Supporting experiments on rotation-based tasks using MNIST digits. Investigates the network’s generalization failures on simpler compositional tasks, revealing memorization behaviors.
- **`scaling_exps/`**: Data efficiency and scaling experiments, demonstrating how data augmentation with independent factors ("stripes") significantly improves compositional generalization and data efficiency.
- **`encoder_exps/`**: Experiments comparing different encoder architectures and their effectiveness in supporting compositional generalization.

## Summary of Experiments

### Main Experiment: Gaussian Bump Generation (`gaussian_exps/`)
The primary experiments evaluate neural network performance on a 2D Gaussian bump image generation task, testing extrapolation to unseen (OOD) coordinate combinations. Networks are provided fully disentangled input representations:

- **Bump-based encoding**: Input coordinates encoded as Gaussian bumps.
- **Ramp-based encoding**: Coordinates encoded as continuous ramp signals.
- **1-hot encoding**: Discrete indicator encoding of coordinates.

**Findings:**
- Standard networks fail compositional generalization despite explicitly disentangled inputs.
- Models rely on memorization and superposition of ID examples for OOD generalization.
- Forced rendering of disentangled latents in the pixel domain (via low-rank architectural constraints or dataset augmentation with independent factors) achieves robust compositional generalization.

### Supporting Experiment: Kernel-based Analysis (`kernel_exps/`)
- Analyzes learned kernels to reveal memorization-based OOD generalization behaviors.
- Confirms that neural networks approximate "binary factorized kernels" and rely on superposition of memorized in-distribution data rather than genuine factor composition.

### Supporting Experiment: Rotation Task (`rotation_exps/`)
- Uses MNIST digit rotation tasks to investigate simpler compositional generalization scenarios.
- Results confirm the broader phenomenon of memorization and linear superposition of in-distribution samples as a generic neural network generalization strategy.

### Supporting Experiment: Scaling Analysis (`scaling_exps/`)
- Explores data scaling and demonstrates how augmenting datasets with isolated factors (1D stripes) significantly enhances compositional generalization and reduces data requirements from cubic (\(N^3\)) to linear (\(N\)) scaling with respect to image size.

### Supporting Experiment: Encoder Architectures (`encoder_exps/`)
- Compares the effectiveness of different encoder architectures and input encoding strategies on compositional generalization performance.

## Environment Setup

Install the required packages for running the experiments using the provided `environment.yaml` file.

### Install Conda (if needed)
- [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)
- [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

### Create the environment

```bash
conda env create -f environment.yaml
```

### Activate the environment
Once the environment is created, activate it using:

```bash
conda activate compositional-gen
```

# Citation
Please cite our paper as follows:

```bibtex
@inproceedings{Liang2025compositional,
    title={Compositional Generalization via Forced Rendering of Disentangled Latents},
    author={Qiyao Liang and Daoyuan Qian and Liu Ziyin and Ila Fiete},
    booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
    year={2025}
}
```

# Paper Link and Code
[[Link to paper (arXiv)](https://link_to_your_paper)](https://arxiv.org/abs/2501.18797)

