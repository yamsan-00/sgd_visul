# sgd_visul

> Interactive visualizer for **Stochastic Gradient Descent** — see how different optimizers (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam) navigate loss surfaces in real time.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GUI-PyQt-green?logo=qt&logoColor=white" alt="PyQt">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Motivation

Optimization is the engine of machine learning, but most introductions treat SGD variants as black-box formulas. `sgd_visul` is a small, self-contained desktop app that lets you **watch** optimizers descend a 2D loss surface — building intuition for step size, momentum, adaptive learning rates, and saddle-point pathology.

## Features

- Real-time animation of optimizer trajectories on a 2D contour plot
- Side-by-side comparison of multiple optimizers
- Configurable loss functions (quadratic, Rosenbrock, saddle, Beale, custom)
- Adjustable learning rate, momentum, and iteration count from the GUI
- Loss-vs-iteration and gradient-norm-vs-iteration plots

## Getting Started

### Prerequisites

- Python 3.10+
- `numpy`, `matplotlib`, `PyQt5`

### Install

```bash
git clone https://github.com/yamsan-00/sgd_visul.git
cd sgd_visul
pip install -r requirements.txt     # see requirements below
```

### Run

```bash
python main.py
```

## Project Structure

```
sgd_visul/
├── algorithms.py     # Optimizer implementations (SGD, Momentum, Adam, ...)
├── formula_panel.py  # LaTeX-style formula display panel
├── theme.py          # Qt styling / dark theme
├── main.py           # Application entry point
├── app.py            # Main window + widget wiring
├── sgd_visualizer_desktop.py   # Desktop launcher (canonical)
├── sgd_visualizer_desktop(1).py    # [legacy duplicate — slated for removal]
├── test_functions.py  # Loss function library
└── test_functions.py
```

## Dependencies

```text
numpy>=1.24
matplotlib>=3.7
PyQt5>=5.15
```

## Roadmap

- [ ] Add Nadam, AdaBelief, Lion
- [ ] 3D surface mode
- [ ] Record trajectories as GIF
- [ ] Unit tests + GitHub Actions CI
- [ ] Publish as pip-installable package

## License

MIT — see [LICENSE](LICENSE).