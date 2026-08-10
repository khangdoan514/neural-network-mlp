# **Neural Network MLP**

**MLP** (Multilayer Perceptron) implemented from scratch with **Python** and **NumPy** only. The project covers tensors, reverse mode autograd, backpropagation, linear layers, activations (ReLU, Tanh, Sigmoid, Softmax), losses (MSE, BCE with logits, Cross Entropy), optimizers (SGD with momentum, Adam), minibatch training, finite difference gradient checks, and demos on XOR, moons, and spirals. No PyTorch, TensorFlow, or JAX.

> *The repository is organized as an ordered, step by step curriculum. Each step includes math notes and maps to matching source files.*

---

## **Mathematical Overview**

### **1. Chain rule**

How a scalar loss $L$ pushes gradients backward through $y = g(x)$:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}\,
\frac{\partial y}{\partial x}
$$

### **2. Finite differences**

Numerical check against analytic `.grad`:

$$
\partial_{i} f(x)
\approx
\frac{f(x+\varepsilon e_{i})-f(x-\varepsilon e_{i})}{2\varepsilon}
$$

### **3. Affine map**

Forward pass of every `Linear` module:

$$
y = xW + b
$$

with $x \in \mathbb{R}^{N \times d_{\mathrm{in}}}$,
$W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$,
$b \in \mathbb{R}^{d_{\mathrm{out}}}$.

### **4. Activations**

Elementwise nonlinearities (and softmax over a class axis):

$$
\begin{aligned}
\mathrm{ReLU}(x) &= \max(x, 0) \\
\sigma(x) &= \frac{1}{1 + e^{-x}} \\
\tanh x &= \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \\
\mathrm{softmax}(z)_{i} &= \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}
\end{aligned}
$$

### **5. MSE**

Regression objective:

$$
L = \frac{1}{N}\sum_{i}(\hat{y}_{i} - y_{i})^{2}
$$

### **6. Cross entropy**

Classification objective from class probabilities $p$:

$$
L = -\sum_{i} y_{i}\log p_{i}
\quad \text{(one hot } y\text{)}
$$

$$
L = -\log p_{c}
\quad \text{(class index } c\text{)}
$$

### **7. SGD**

Basic parameter update:

$$
\theta \leftarrow \theta - \eta\, g
$$

### **8. SGD with momentum**

Velocity smoothed update:

$$
v \leftarrow \mu v + g,
\qquad
\theta \leftarrow \theta - \eta\, v
$$

### **9. Adam**

Adaptive moments:

$$
\begin{aligned}
m &\leftarrow \beta_{1} m + (1-\beta_{1})g \\
v &\leftarrow \beta_{2} v + (1-\beta_{2})g^{2} \\
\hat{m} &= \frac{m}{1-\beta_{1}^{t}},
\quad
\hat{v} = \frac{v}{1-\beta_{2}^{t}} \\
\theta &\leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
\end{aligned}
$$

---

## **Step-by-step Roadmap**

| Step | Topic | Code | Guide |
|------|-------|------|-------|
| 01 | Tensors and ops | `src/tensor.py` | [docs/Tensors.md](docs/Tensors.md) |
| 02 | Autograd / chain rule | `src/tensor.py` (`backward`) | [docs/Autograd.md](docs/Autograd.md) |
| 03 | Gradient checking | `src/gradcheck.py` | [docs/Gradcheck.md](docs/Gradcheck.md) |
| 04 | Modules and `Linear` | `src/modules.py` | [docs/Layers.md](docs/Layers.md) |
| 05 | Activations | `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | [docs/Activations.md](docs/Activations.md) |
| 06 | Losses | `src/losses.py` | [docs/Losses.md](docs/Losses.md) |
| 07 | Optimizers | `src/optim.py` | [docs/Optimizers.md](docs/Optimizers.md) |
| 08 | MLP (`Sequential`) | `src/modules.py` | [docs/MLP.md](docs/MLP.md) |
| 09 | Training loop | `src/training.py` | [docs/Training.md](docs/Training.md) |
| 10 | Experiments | `examples/` | [docs/Experiments.md](docs/Experiments.md) |

---

## **Getting Started**

```bash
git clone https://github.com/<username>/neural-network-mlp.git
cd neural-network-mlp

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### **Run Tests**

```bash
pytest -q
```

### **Run Demos**

```bash
python examples/train_xor.py
python examples/train_moons.py
python examples/train_spirals.py
```

### **Minimal Usage**

```python
from src import Tensor, Linear, ReLU, Sequential, Adam, BCEWithLogitsLoss, train
from src.data import make_xor
from src.training import accuracy_binary

x, y = make_xor()
model = Sequential(Linear(2, 16), ReLU(), Linear(16, 1))
train(model, BCEWithLogitsLoss(), Adam(model.parameters(), lr=0.05), x, y, epochs=100, metric_fn=accuracy_binary)
```

---

## **Project Structure**

```text
neural-network-mlp/
├── src/
│   ├── tensor.py       # Tensor and autograd ops
│   ├── gradcheck.py    # Finite difference checks
│   ├── modules.py      # Linear, activations, Sequential
│   ├── losses.py       # MSE, BCE, CrossEntropy
│   ├── optim.py        # SGD, Adam
│   ├── training.py     # Minibatch train loop
│   └── data.py         # XOR / moons / spirals
│
├── docs/               # Step-by-step guides
├── examples/           # Runnable demos
├── tests/              # Unit and learning tests
└── README.md
```