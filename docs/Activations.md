# **Activations**

Elementwise nonlinearities so stacked `Linear` layers do not collapse to one affine map. Math and autograd live on `Tensor`; `Module` wrappers let you drop activations into a network.

> *`src/tensor.py`, `src/modules.py`*

---

## **Math**

| Name | Forward | Backward |
|------|---------|----------|
| ReLU | $\mathrm{ReLU}(x)=\max(x,0)$ | $\dot x = \dot y \cdot 1_{x>0}$ |
| Tanh | $\tanh x$ | $\dot x = \dot y \cdot (1-\tanh^{2} x)$ |
| Sigmoid | $\sigma(x)=\dfrac{1}{1+e^{-x}}$ | $\dot x = \dot y \cdot \sigma(x)(1-\sigma(x))$ |
| Softmax | $\mathrm{softmax}(z)_{i}=\dfrac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}$ | $\dot z = p \odot (g - \mathbf{1}^{\top}(g \odot p))$ |

Without a nonlinearity, $W_2(W_1 x + b_1) + b_2$ is still one linear map. ReLU, Tanh, and Sigmoid break that collapse so the network can learn XOR-like boundaries. The ReLU subgradient at zero is taken to be zero. Sigmoid squashes to $(0, 1)$. Tanh squashes to $(-1, 1)$.

Softmax is usually applied over the class axis. The backward pass uses the Jacobian vector product above ($p=\mathrm{softmax}(z)$, $g=\dot p$).

---

## **Tensor Methods**

### **`relu()`**

```python
def relu(self) -> Tensor:
    out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, children=(self,), operation="relu")

    def local_backward() -> None:
        if self.requires_grad:
            assert self.grad is not None and out.grad is not None
            self.grad += out.grad * (self.data > 0.0)

    out.gradient_function = local_backward

    return out
```

### **`tanh()`**

```python
def tanh(self) -> Tensor:
    values = np.tanh(self.data)
    out = Tensor(values, requires_grad=self.requires_grad, children=(self,), operation="tanh")

    def local_backward() -> None:
        if self.requires_grad:
            assert self.grad is not None and out.grad is not None
            self.grad += out.grad * (1.0 - values**2)

    out.gradient_function = local_backward

    return out
```

### **`sigmoid()`**

```python
def sigmoid(self) -> Tensor:
    values = 1.0 / (1.0 + np.exp(-self.data))
    out = Tensor(values, requires_grad=self.requires_grad, children=(self,), operation="sigmoid")

    def local_backward() -> None:
        if self.requires_grad:
            assert self.grad is not None and out.grad is not None
            self.grad += out.grad * values * (1.0 - values)

    out.gradient_function = local_backward

    return out
```

### **`softmax()`**

Stable softmax (subtract max before `exp`).  Backward uses the Jacobian vector product form from the math table.

```python
def softmax(self, axis: int = -1) -> Tensor:
    shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=axis, keepdims=True)
    out = Tensor(probabilities, requires_grad=self.requires_grad, children=(self,), operation="softmax")

    def local_backward() -> None:
        if not self.requires_grad:
            return

        assert self.grad is not None and out.grad is not None
        gradient = out.grad
        sum_gradient_probability = np.sum(gradient * probabilities, axis=axis, keepdims=True)
        self.grad += probabilities * (gradient - sum_gradient_probability)

    out.gradient_function = local_backward

    return out
```

---

## **Module Wrappers**

Thin layers that call the matching `Tensor` method so activations compose with `Linear` and `Sequential`.

### **`ReLU`**

```python
class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()
```

### **`Tanh`**

```python
class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()
```

### **`Sigmoid`**

```python
class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()
```

### **`Softmax`**

```python
class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        return input.softmax(axis=self.axis)
```

Explanation:

1. Gradients are already defined on `Tensor`; wrappers add no new math.
2. Prefer `Softmax` as a module when class probabilities are part of the forward graph.
3. For classification with logits + cross entropy, you often skip an explicit softmax and fold it into the loss (step 06).

---

## **Check**

```python
from src import Tensor
from src.gradcheck import check_tensor_gradient
from src.modules import ReLU, Softmax

x = Tensor([-1.0, 0.5, 2.0], requires_grad=True)
passed, relative_error = check_tensor_gradient(lambda tensor: ReLU()(tensor).power(2).sum(), x)
print(passed, relative_error)

logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
probs = Softmax(axis=1)(logits)
print(probs.data.sum(axis=1))
```