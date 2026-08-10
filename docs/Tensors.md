# **Tensors**

Represent data as a differentiable array: values now, gradients later. A `Tensor` wraps a NumPy array and records how it was created so `.backward()` can apply the chain rule.

Code lives in `src/tensor.py`.

---

## **Math**

A tensor here is a multidimensional array $x \in \mathbb{R}^{n_1 \times \cdots \times n_d}$.

Common MLP shapes:

- Input batch: $(N, D_{\text{in}})$
- Weights: $(D_{\text{in}}, D_{\text{out}})$
- Bias: $(D_{\text{out}},)$

---

## **Utility functions**

### **Convert input to a NumPy array**

`as_data` accepts either a `Tensor` or an array like value and always returns `float64` data.

```python
def as_data(x: ArrayLike | Tensor) -> NDArray[np.float64]:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x, dtype=np.float64)
```

### **Undo broadcasting in the backward pass**

When NumPy broadcasts shapes in the forward pass, the upstream gradient may be larger than the original tensor. `unbroadcast` sums out those extra axes so the gradient matches the parameter shape again.

```python
def unbroadcast(grad: NDArray[np.float64], shape: tuple[int, ...]) -> NDArray[np.float64]:
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape, strict=True)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)
```

Explanation:

1. Drop leading axes that broadcasting invented.
2. Sum over axes that were size `1` in the original tensor but expanded during the forward op.
3. Reshape back to the original `shape`.

---

## **The `Tensor` class**

A differentiable multidimensional array.

| Field | Role |
|-------|------|
| `data` | Array like payload (`np.ndarray`) |
| `requires_grad` | If true, accumulate gradients into `.grad` during backward |
| `children` | Constructor arg: upstream tensors for this op |
| `parents` | Stored tuple of those upstream tensors |
| `operation` | Debug name of the creating operation |
| `gradient_function` | Local reverse mode rule for this node (not the public `backward` method) |

```python
class Tensor:
    def __init__(
        self,
        data: ArrayLike | Tensor,
        *,
        requires_grad: bool = False,
        children: Iterable[Tensor] = (),
        operation: str = "",
        gradient_function: Callable[[], None] | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad) or any(c.requires_grad for c in children)
        self.grad: NDArray[np.float64] | None = (
            np.zeros_like(self.data) if self.requires_grad else None
        )
        self.parents = tuple(children)
        self.operation = operation
        self.gradient_function = gradient_function or (lambda: None)
```

`requires_grad` is also true if any parent needs gradients, so the graph stays connected.

---

## **Construction helpers**

Factory methods for common initial tensors:

```python
@staticmethod
def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

@staticmethod
def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

@staticmethod
def randn(*shape: int, requires_grad: bool = False, rng=None) -> Tensor:
    rng = rng or np.random.default_rng()
    return Tensor(rng.standard_normal(shape, dtype=np.float64), requires_grad=requires_grad)
```

Also available:

- `detach()` — copy values without gradient tracking
- `numpy()` — raw `data`
- `shape`, `ndim`, `T` — convenience properties (`T` calls `transpose()`)

---

## **Autograd entry points**

### **Clear gradients**

```python
def zero_grad(self) -> None:
    if self.grad is not None:
        self.grad.fill(0.0)
```

### **Reverse mode AD**

Call `loss.backward()` on a scalar. For non scalars, pass an explicit upstream gradient.

```python
def backward(self, gradient: ArrayLike | None = None) -> None:
    if not self.requires_grad:
        raise RuntimeError("backward called on a tensor that does not require grad")

    if gradient is None:
        if self.data.size != 1:
            raise RuntimeError("backward for non-scalar requires an explicit gradient")
        gradient = np.ones_like(self.data)
    else:
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.data.shape:
            raise ValueError(
                f"gradient shape {gradient.shape} != tensor shape {self.data.shape}"
            )

    topo: list[Tensor] = []
    visited: set[int] = set()

    def build(node: Tensor) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in node.parents:
            build(child)
        topo.append(node)

    build(self)

    self.grad = gradient.copy()
    for node in reversed(topo):
        node.gradient_function()
```

Explanation:

1. Seed $\frac{\partial L}{\partial L} = 1$ for scalars (or use the provided gradient).
2. Build a topological order of the computation graph.
3. Walk the graph in reverse and run each node's local `gradient_function`.

---

## **Core ops**

Each op stores a local backward closure on the output tensor.

### **Shared binary helper**

`binary` runs a forward NumPy function, builds an output `Tensor`, and wires both parents' gradient rules with `unbroadcast`.

```python
def binary(self, other, *, forward, backward_for_a, backward_for_b, operation: str) -> Tensor:
    b = other if isinstance(other, Tensor) else Tensor(other)
    out_data = forward(self.data, b.data)
    out = Tensor(
        out_data,
        requires_grad=self.requires_grad or b.requires_grad,
        children=(self, b),
        operation=operation,
    )

    def local_backward() -> None:
        assert out.grad is not None
        if self.requires_grad:
            assert self.grad is not None
            self.grad += unbroadcast(backward_for_a(out.grad, self.data, b.data), self.data.shape)
        if b.requires_grad:
            assert b.grad is not None
            b.grad += unbroadcast(backward_for_b(out.grad, self.data, b.data), b.data.shape)

    out.gradient_function = local_backward
    return out
```

### **Arithmetic local rules**

| Op | Forward | Backward |
|----|---------|----------|
| Add | $c = a + b$ | $\dot a = \dot c$, $\dot b = \dot c$ |
| Sub | $c = a - b$ | $\dot a = \dot c$, $\dot b = -\dot c$ |
| Mul | $c = a \cdot b$ | $\dot a = \dot c \cdot b$, $\dot b = \dot c \cdot a$ |
| Div | $c = a / b$ | $\dot a = \dot c / b$, $\dot b = -\dot c \cdot a / b^{2}$ |
| Pow | $c = a^{p}$ | $\dot a = \dot c \cdot p \cdot a^{p-1}$ |
| Matmul | $C = AB$ | $\dot A = \dot C B^{\top}$, $\dot B = A^{\top}\dot C$ |

```python
def add(self, other):
    return self.binary(
        other,
        forward=lambda a, b: a + b,
        backward_for_a=lambda g, a, b: g,
        backward_for_b=lambda g, a, b: g,
        operation="add",
    )

def matmul(self, other):
    b = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ b.data, requires_grad=..., children=(self, b), operation="matmul")

    def local_backward() -> None:
        if self.requires_grad:
            self.grad += out.grad @ b.data.T
        if b.requires_grad:
            b.grad += self.data.T @ out.grad

    out.gradient_function = local_backward
    return out
```

### **Reductions and shape ops**

- `sum` / `mean` — reduce values; backward scatters the upstream gradient back to the full shape.
- `reshape` — view change; backward reshapes the gradient to the original shape.
- `transpose` — permute axes; backward applies the inverse permutation.

### Activations on the tensor

| Op | Forward | Backward |
|----|---------|----------|
| ReLU | $\max(x,0)$ | $\dot x = \dot y \cdot 1_{x>0}$ |
| Tanh | $\tanh x$ | $\dot x = \dot y \cdot (1-\tanh^{2} x)$ |
| Sigmoid | $\sigma(x)$ | $\dot x = \dot y \cdot \sigma(1-\sigma)$ |
| Exp | $e^{x}$ | $\dot x = \dot y \cdot e^{x}$ |
| Log | $\log x$ | $\dot x = \dot y / x$ |

### **Softmax**

Stable softmax (subtract max before `exp`). Backward uses the Jacobian vector product:

$$
\frac{\partial L}{\partial x} = p \odot \big(g - \mathbf{1}^{\top}(g \odot p)\big)
$$

```python
def softmax(self, axis: int = -1) -> Tensor:
    shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    probs = ex / np.sum(ex, axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=self.requires_grad, children=(self,), operation="softmax")

    def local_backward() -> None:
        g = out.grad
        p = probs
        sum_gp = np.sum(g * p, axis=axis, keepdims=True)
        self.grad += p * (g - sum_gp)

    out.gradient_function = local_backward
    return out
```

### **Indexing**

Basic `get_item` for convenience. Gradients flow back with a dense scatter via `np.add.at`.

```python
def get_item(self, item):
    out_data = self.data[item]
    out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), operation="getitem")

    def local_backward() -> None:
        if self.requires_grad:
            np.add.at(self.grad, item, out.grad)

    out.gradient_function = local_backward
    return out
```

---

## **Check**

```python
from src import Tensor

a = Tensor([[1.0, 2.0]], requires_grad=True)
b = a.multiply(3.0)
print(b.data)  # [[3.0, 6.0]]

loss = b.sum()
loss.backward()
print(a.grad)  # [[3.0, 3.0]]
```