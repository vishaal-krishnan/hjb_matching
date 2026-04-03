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
│   │   ├── lensing.yaml
│   │   ├── lensing_concave.yaml
│   │   ├── lensing_convex.yaml
│   │   ├── lensing_flat.yaml
│   │   └── lensing_uncontrolled.yaml
│   ├── distributions.py    # Sampling functions and cost profiles (nu)
│   ├── model.py            # Neural network architectures (Haiku)
│   ├── train.py            # Training loop and loss functions
│   ├── plot.py             # Visualization utilities
│   └── run.py              # Entry point for training
├── mnist/                  # MNIST experiment (PyTorch)
│   ├── configs/
│   │   └── mnist.yaml
│   ├── model.py            # U-Net score network
│   ├── train.py            # Training and sampling
│   └── run.py              # Entry point
├── notebooks/              # Colab notebooks
│   ├── toy_experiments.ipynb
│   └── lensing_experiments.ipynb
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
```

Training outputs (model weights, loss history, config) are saved to `toy/outputs/<experiment>/`.

### Lensing Experiments

The lensing experiments demonstrate geometric control of stochastic transport via spatial cost fields. Four variants are provided:

```bash
python -m toy.run --config toy/configs/lensing_uncontrolled.yaml
python -m toy.run --config toy/configs/lensing_flat.yaml
python -m toy.run --config toy/configs/lensing_concave.yaml
python -m toy.run --config toy/configs/lensing_convex.yaml
```

### Running on Google Colab

For GPU-accelerated training, use the provided notebooks:

- `notebooks/toy_experiments.ipynb` — trains all toy experiments (4 Gaussians, Two Moons, Swiss Roll)
- `notebooks/lensing_experiments.ipynb` — trains all 4 lensing variants

Open in Colab, select a GPU runtime, and run all cells. Outputs are zipped for download.

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
| Lensing (uncontrolled) | `single_gaussian` | `toy/configs/lensing_uncontrolled.yaml` | No cost field (baseline) |
| Lensing (flat) | `single_gaussian` | `toy/configs/lensing_flat.yaml` | Uniform cost field |
| Lensing (concave) | `single_gaussian` | `toy/configs/lensing_concave.yaml` | Attractive potential well |
| Lensing (convex) | `single_gaussian` | `toy/configs/lensing_convex.yaml` | Repulsive potential barrier |
| MNIST | — | `mnist/configs/mnist.yaml` | MNIST digit generation |

## Citation

```bibtex
@inproceedings{sinha2025hjb,
  title={HJB Matching: Generative Modeling via Hamilton-Jacobi-Bellman Optimal Control},
  author={Sinha, Sumit},
  booktitle={NeurIPS},
  year={2025}
}
```
