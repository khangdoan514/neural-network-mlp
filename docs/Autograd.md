# **Autograd**

Reverse mode automatic differentiation: build a graph on the forward pass, then push gradients backward with the chain rule. Each op stores a local `gradient_function` on its output; `backward()` walks the graph in reverse and runs those rules.

> *`src/tensor.py`*

---

## **Math**

For a composition $L = f(g(x))$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \frac{\partial g}{\partial x}
$$

Reverse mode:

1. Forward: evaluate ops and record parents on each output tensor.
2. Seed $\frac{\partial L}{\partial L} = 1$ at the scalar loss.
3. Walk the graph backward; each node applies its local rule and accumulates into parent `.grad` buffers.

Local rules for the ops used most in an MLP:

| Op | Forward | Backward |
|----|---------|----------|
| Add | $c = a + b$ | $\dot a = \dot c$, $\dot b = \dot c$ |
| Sub | $c = a - b$ | $\dot a = \dot c$, $\dot b = -\dot c$ |
| Mul | $c = a \cdot b$ | $\dot a = \dot c \cdot b$, $\dot b = \dot c \cdot a$ |
| Div | $c = a / b$ | $\dot a = \dot c / b$, $\dot b = -\dot c \cdot a / b^{2}$ |
| Matmul | $C = AB$ | $\dot A = \dot C B^{\top}$, $\dot B = A^{\top}\dot C$ |
| ReLU | $y = \max(x,0)$ | $\dot x = \dot y \cdot 1_{x>0}$ |

Dot notation $\dot a$ means $\frac{\partial L}{\partial a}$, the upstream gradient flowing into $a$.

---

## **Graph Fields**

Every tensor that participates in an op carries the edges needed for reverse mode:

| Field | Role |
|-------|------|
| `parents` | Upstream tensors that produced this node |
| `operation` | Debug name of the creating op |
| `gradient_function` | Local reverse mode rule (closure) |
| `grad` | Accumulator for $\partial L / \partial(\text{this tensor})$ |

`requires_grad` is true if this tensor was created with it, or if any parent needs gradients, so the graph stays connected.

---

## **`backward()`**

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
            raise ValueError(f"gradient shape {gradient.shape} != tensor shape {self.data.shape}")

    topo: list[Tensor] = []
    visited: set[int] = set()

    def build(node: Tensor) -> None:
        if id(node) in visited:
            return

        visited.add(id(node))
        for parent in node.parents:
            build(parent)

        topo.append(node)

    build(self)
    self.grad = gradient.copy()
    for node in reversed(topo):
        node.gradient_function()
```

Explanation:

1. Seed $\frac{\partial L}{\partial L} = 1$ for scalars (or use the provided gradient).
2. Depth first walk of `parents` builds a topological order (`topo`).
3. Walk `reversed(topo)` so every consumer runs before its producers.
4. Each node's `gradient_function` adds into parent `.grad` with `+=`.

Call `zero_grad()` before a new backward if you reuse parameters across steps; otherwise gradients keep accumulating.

```python
def zero_grad(self) -> None:
    if self.grad is not None:
        self.grad.fill(0.0)
```

---

## **Wiring a Local Rule**

### **Shared Binary Function**

Most elementwise binary ops go through `binary`. It runs a forward NumPy function, builds the output tensor, and attaches one `local_backward` that updates both parents through `unbroadcast`.

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

---

## **Worked Walkthrough**

$$
L = \sum_i (3 a_i), \qquad a = [1, 2]
$$

Forward graph: `a` → `multiply(3)` → `b` → `sum` → `L`.

```python
from src import Tensor

a = Tensor([[1.0, 2.0]], requires_grad=True)
b = a.multiply(3.0)
loss = b.sum()
loss.backward()
```

Backward:

1. Seed `loss.grad = 1`.
2. `sum`'s rule scatters `1` onto `b.grad` → `[[1, 1]]`.
3. `multiply`'s rule: $\dot a = \dot b \cdot 3$ → `a.grad = [[3, 3]]`.

---

## **Check**

```python
from src import Tensor

x = Tensor([[0.5, -1.0], [2.0, 0.0]], requires_grad=True)
w = Tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)

y = x.matmul(w).relu()
loss = y.mean()
loss.backward()

print(loss.data)
print(x.grad)
print(w.grad)
```