# **Tensors**

Represent data as a differentiable array: values now, gradients later. A `Tensor` wraps a NumPy array and records how it was created so `.backward()` can apply the chain rule.

> *`src/tensor.py`*

---

## **Math**

A tensor here is a multidimensional array $x \in \mathbb{R}^{n_1 \times \cdots \times n_d}$.

Common MLP shapes:

- Input batch: $(N, D_{\text{in}})$
- Weights: $(D_{\text{in}}, D_{\text{out}})$
- Bias: $(D_{\text{out}},)$

---

## **Helper Functions**

### **Convert Input to a NumPy Array**

`as_data` accepts either a `Tensor` or an array like value and always returns `float64` data.

```python
def as_data(x: ArrayLike | Tensor) -> NDArray[np.float64]:
    if isinstance(x, Tensor):
        return x.data

    return np.asarray(x, dtype=np.float64)
```

### **Undo Broadcasting in the Backward Pass**

When NumPy broadcasts shapes in the forward pass, the upstream gradient may be larger than the original tensor. `unbroadcast` sums out those extra axes so the gradient matches the parameter shape again.

```python
def unbroadcast(grad: NDArray[np.float64], shape: tuple[int, ...]) -> NDArray[np.float64]:
    if grad.shape == shape:
        return grad

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for index, (grad_size, shape_size) in enumerate(zip(grad.shape, shape, strict=True)):
        if shape_size == 1 and grad_size != 1:
            grad = grad.sum(axis=index, keepdims=True)

    return grad.reshape(shape)
```

Explanation:

1. Drop leading axes that broadcasting invented.
2. Sum over axes that were size `1` in the original tensor but expanded during the forward op.
3. Reshape back to the original `shape`.

---

## **The `Tensor` Class**

A differentiable multidimensional array.

| Field | Role |
|-------|------|
| `data` | Array like payload (`np.ndarray`) |
| `requires_grad` | If true, accumulate gradients into `.grad` during backward |
| `children` | Constructor arg: upstream tensors for this op |
| `parents` | Stored tuple of those upstream tensors |
| `operation` | Debug name of the creating operation |
| `gradient_function` | Local reverse mode rule for this node (not the public `backward` method) |
| `grad` | Accumulator for $\partial L / \partial(\text{this tensor})$ |

```python
class Tensor:
    def __init__(self, data: ArrayLike | Tensor, *, requires_grad: bool = False, children: Iterable[Tensor] = (), operation: str = "", gradient_function: Callable[[], None] | None = None) -> None:
        if isinstance(data, Tensor):
            self.data = data.data.copy()

        else:
            self.data = np.asarray(data, dtype=np.float64)

        self.requires_grad = bool(requires_grad) or any(child.requires_grad for child in children)
        self.grad: NDArray[np.float64] | None = np.zeros_like(self.data) if self.requires_grad else None
        self.parents = tuple(children)
        self.operation = operation
        self.gradient_function = gradient_function or (lambda: None)
```

`requires_grad` is also true if any parent needs gradients, so the graph stays connected.

---

## **Construction Functions**

Factory methods for common initial tensors:

```python
@staticmethod
def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

@staticmethod
def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

@staticmethod
def randn(*shape: int, requires_grad: bool = False, random_generator: np.random.Generator | None = None) -> Tensor:
    random_generator = random_generator or np.random.default_rng()

    return Tensor(random_generator.standard_normal(shape, dtype=np.float64), requires_grad=requires_grad)
```

Also available:

- `detach()` — copy values without gradient tracking
- `numpy()` — raw `data`
- `shape`, `ndim`, `T` — convenience properties (`T` calls `transpose()`)

---

## **Core Operations**

Each operation stores a local backward closure on the output tensor. How those closures are wired is covered in [Autograd.md](Autograd.md).

### **Shared Binary Function**

Most elementwise binary operations go through `binary`. It runs a forward NumPy function, builds the output tensor, and attaches one `local_backward` that updates both parents through `unbroadcast`.

```python
def binary(self, other: ArrayLike | Tensor, *, forward: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], backward_for_a: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], backward_for_b: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], operation: str) -> Tensor:
    right = other if isinstance(other, Tensor) else Tensor(other)
    out_data = forward(self.data, right.data)
    out = Tensor(out_data, requires_grad=self.requires_grad or right.requires_grad, children=(self, right), operation=operation)

    def local_backward() -> None:
        assert out.grad is not None
        if self.requires_grad:
            assert self.grad is not None
            self.grad += unbroadcast(backward_for_a(out.grad, self.data, right.data), self.data.shape)

        if right.requires_grad:
            assert right.grad is not None
            right.grad += unbroadcast(backward_for_b(out.grad, self.data, right.data), right.data.shape)

    out.gradient_function = local_backward

    return out
```

`unbroadcast` undoes NumPy broadcasting so each parent receives a gradient with its original shape.

### **Arithmetic Local Rules**

| Operation | Forward | Backward |
|----|---------|----------|
| Add | $c = a + b$ | $\dot a = \dot c$, $\dot b = \dot c$ |
| Sub | $c = a - b$ | $\dot a = \dot c$, $\dot b = -\dot c$ |
| Mul | $c = a \cdot b$ | $\dot a = \dot c \cdot b$, $\dot b = \dot c \cdot a$ |
| Div | $c = a / b$ | $\dot a = \dot c / b$, $\dot b = -\dot c \cdot a / b^{2}$ |
| Pow | $c = a^{p}$ | $\dot a = \dot c \cdot p \cdot a^{p-1}$ |
| Matmul | $C = AB$ | $\dot A = \dot C B^{\top}$, $\dot B = A^{\top}\dot C$ |

### **Example: multiply()**

```python
def multiply(self, other: ArrayLike | Tensor) -> Tensor:
    return self.binary(other, forward=lambda left, right: left * right, backward_for_a=lambda gradient, left, right: gradient * right, backward_for_b=lambda gradient, left, right: gradient * left, operation="mul")
```

### **Example: matmul()**

Matmul does not use `binary`; the local rule is written directly.

```python
def matmul(self, other: ArrayLike | Tensor) -> Tensor:
    right = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ right.data, requires_grad=self.requires_grad or right.requires_grad, children=(self, right), operation="matmul")

    def local_backward() -> None:
        assert out.grad is not None
        if self.requires_grad:
            assert self.grad is not None
            self.grad += out.grad @ right.data.T

        if right.requires_grad:
            assert right.grad is not None
            right.grad += self.data.T @ out.grad

    out.gradient_function = local_backward

    return out
```

### **Reductions and Shape Operations**

- `sum` / `mean` — reduce values; backward scatters the upstream gradient back to the full shape.
- `reshape` — view change; backward reshapes the gradient to the original shape.
- `transpose` — permute axes; backward applies the inverse permutation.

### **Activations on the Tensor**

| Operation | Forward | Backward |
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

### **Indexing**

Basic `get_item` for convenience. Gradients flow back with a dense scatter via `np.add.at`.

```python
def get_item(self, item: Any) -> Tensor:
    out_data = self.data[item]
    out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), operation="getitem")

    def local_backward() -> None:
        if self.requires_grad:
            assert self.grad is not None and out.grad is not None
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
print(b.data)

loss = b.sum()
loss.backward()
print(a.grad)
```
