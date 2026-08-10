# **Neural Network MLP**

**MLP** (Multilayer Perceptron) implemented from scratch with **Python** and **NumPy** only. The project covers tensors, reverse mode autograd, backpropagation, linear layers, activations (ReLU, Tanh, Sigmoid, Softmax), losses (MSE, BCE with logits, Cross Entropy), optimizers (SGD with momentum, Adam), minibatch training, finite difference gradient checks, and demos on XOR, moons, and spirals. No PyTorch, TensorFlow, or JAX.

> *The repository is organized as an ordered, step by step curriculum. Each step includes math notes and maps to matching source files.*

---

## **Mathematical Overview**

<table>
<tr>
<td width="28%"><b>Chain rule</b><br/><sub>Gradient flow through composition</sub></td>
<td>

$$\displaystyle \frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\,\frac{\partial y}{\partial x}$$

</td>
</tr>
<tr>
<td><b>Affine layer</b><br/><sub>Every <code>Linear</code> map</sub></td>
<td>

$$\displaystyle
\begin{aligned}
y &= xW + b \\[4pt]
x &\in \mathbb{R}^{N \times d_{\mathrm{in}}},\ 
W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}},\ 
b \in \mathbb{R}^{d_{\mathrm{out}}}
\end{aligned}
$$

</td>
</tr>
<tr>
<td><b>Activations</b><br/><sub>Elementwise nonlinearity</sub></td>
<td>

$$\displaystyle
\begin{aligned}
\mathrm{ReLU}(x) &= \max(x,0) \\
\sigma(x) &= \frac{1}{1+e^{-x}} \\
\tanh x &= \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} \\
\mathrm{softmax}(z)_{i} &= \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}
\end{aligned}
$$

</td>
</tr>
<tr>
<td><b>MSE</b><br/><sub>Regression objective</sub></td>
<td>

$$\displaystyle L = \frac{1}{N}\sum_{i}(\hat{y}_{i} - y_{i})^{2}$$

</td>
</tr>
<tr>
<td><b>Cross entropy</b><br/><sub>Classification objective</sub></td>
<td>

$$\displaystyle
L =
\begin{cases}
-\sum_{i} y_{i}\log p_{i} & \text{one hot } y \\[4pt]
-\log p_{c} & \text{class index } c
\end{cases}
$$

</td>
</tr>
<tr>
<td><b>SGD</b><br/><sub>Basic parameter update</sub></td>
<td>

$$\displaystyle \theta \leftarrow \theta - \eta\, g$$

</td>
</tr>
<tr>
<td><b>SGD with momentum</b><br/><sub>Velocity smoothed update</sub></td>
<td>

$$\displaystyle v \leftarrow \mu v + g,\qquad \theta \leftarrow \theta - \eta\, v$$

</td>
</tr>
<tr>
<td><b>Adam</b><br/><sub>Adaptive moments</sub></td>
<td>

$$\displaystyle
\begin{aligned}
m &\leftarrow \beta_{1} m + (1-\beta_{1})g \\
v &\leftarrow \beta_{2} v + (1-\beta_{2})g^{2} \\
\hat{m} &= \frac{m}{1-\beta_{1}^{t}},\quad
\hat{v} = \frac{v}{1-\beta_{2}^{t}} \\
\theta &\leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
\end{aligned}
$$

</td>
</tr>
<tr>
<td><b>Finite differences</b><br/><sub>Gradient check</sub></td>
<td>

$$\displaystyle
\partial_{i} f(x)
\approx
\frac{f(x+\varepsilon e_{i})-f(x-\varepsilon e_{i})}{2\varepsilon}
$$

</td>
</tr>
</table>

---

## **Step-by-step Roadmap**

| Step | Topic | Code | Guide |
|------|-------|------|-------|
| 01 | Tensors and ops | `src/tensor.py` | [steps/Tensors.md](steps/Tensors.md) |
| 02 | Autograd / chain rule | `src/tensor.py` (`backward`) | [steps/Autograd.md](steps/Autograd.md) |
| 03 | Gradient checking | `src/gradcheck.py` | [steps/Gradcheck.md](steps/Gradcheck.md) |
| 04 | Modules and `Linear` | `src/modules.py` | [steps/Layers.md](steps/Layers.md) |
| 05 | Activations | `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | [steps/Activations.md](steps/Activations.md) |
| 06 | Losses | `src/losses.py` | [steps/Losses.md](steps/Losses.md) |
| 07 | Optimizers | `src/optim.py` | [steps/Optimizers.md](steps/Optimizers.md) |
| 08 | MLP (`Sequential`) | `src/modules.py` | [steps/MLP.md](steps/MLP.md) |
| 09 | Training loop | `src/training.py` | [steps/Training.md](steps/Training.md) |
| 10 | Experiments | `examples/` | [steps/Experiments.md](steps/Experiments.md) |

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
├── steps/              # Step-by-step guides
├── examples/           # Runnable demos
├── tests/              # Unit and learning tests
└── README.md
```