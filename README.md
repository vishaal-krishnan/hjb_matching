# HJB Matching

Official implementation of **HJB Matching: Generative Modeling via Hamilton-Jacobi-Bellman Optimal Control**.

## Repository Structure

```
hjb_matching/
├── toy/                    # Toy 2D experiments (JAX)
│   ├── configs/            # YAML configs for each experiment
│   │   ├── 4gaussian.yaml
│   │   ├── two_moon.yaml
│   │   ├── swissroll.yaml
│   │   └── lensing.yaml
│   ├── distributions.py    # Sampling functions for target distributions
│   ├── model.py            # Neural network architectures (Haiku)
│   ├── train.py            # Training loop and loss functions
│   ├── plot.py             # Visualization utilities
│   ├── run.py              # Entry point for training
│   ├── fig_paper.py        # Paper figure generation
│   └── make_figures.py     # Per-experiment figure generation
├── mnist/                  # MNIST experiment (PyTorch)
│   ├── configs/
│   │   └── mnist.yaml
│   ├── model.py            # U-Net score network
│   ├── train.py            # Training and sampling
│   └── run.py              # Entry point
├── notebooks/              # Colab notebooks
│   └── toy_experiments.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/sumit-sinha-seas/HJB_matching.git
cd HJB_matching
pip install -r requirements.txt
```

### Requirements

- **Toy experiments**: JAX, dm-haiku, Optax, PyYAML, Matplotlib
- **MNIST experiment**: PyTorch, torchvision, scikit-learn, tqdm

## Usage

### Toy Experiments

Each toy experiment is configured via a YAML file. To train:

```bash
python -m toy.run --config toy/configs/4gaussian.yaml
python -m toy.run --config toy/configs/two_moon.yaml
python -m toy.run --config toy/configs/swissroll.yaml
python -m toy.run --config toy/configs/lensing.yaml
```

Training outputs (model weights, loss history, config) are saved to `toy/outputs/<experiment>/`.

### Running on Google Colab

For GPU-accelerated training, use the provided notebook:

1. Open `notebooks/toy_experiments.ipynb` in Google Colab
2. Select a GPU runtime
3. Run all cells

The notebook trains all toy experiments and provides a zip download of the outputs.

### Generating Figures

After training, generate per-experiment figures:

```bash
python -m toy.make_figures --outputs toy/outputs/4gaussians
```

Generate the paper figure (requires outputs from multiple experiments):

```bash
python -m toy.fig_paper
```

### MNIST Experiment

```bash
python -m mnist.run --config mnist/configs/mnist.yaml
```

Requires a CUDA-capable GPU.

## Experiments

| Experiment | Distribution | Config | Description |
|------------|-------------|--------|-------------|
| 4 Gaussians | `four_gaussian` | `toy/configs/4gaussian.yaml` | Mixture of 4 Gaussians |
| Two Moons | `two_moon` | `toy/configs/two_moon.yaml` | Two crescent moons |
| Swiss Roll | `swissroll` | `toy/configs/swissroll.yaml` | Swiss roll manifold |
| Lensing | `lensing` | `toy/configs/lensing.yaml` | Gravitational lensing (analytical nu) |
| MNIST | — | `mnist/configs/mnist.yaml` | MNIST digit generation |

