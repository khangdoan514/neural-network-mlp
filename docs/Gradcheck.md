# **Gradcheck**

Never trust a backward pass you have not checked numerically. Finite differences approximate each partial derivative by perturbing one entry of the input; compare that vector to analytic `.grad` from autograd.

> *`src/gradcheck.py`*

---

## **Math**

Central difference for a scalar $f(x)$:

$$
\partial_i f(x) \approx \frac{f(x+\varepsilon e_i) - f(x-\varepsilon e_i)}{2\varepsilon}
$$

Relative error between analytic gradient $g$ and numeric gradient $\tilde g$:

$$
\frac{\|g - \tilde g\|}{\|g\| + \|\tilde g\| + \delta}
$$

With $\varepsilon \approx 10^{-5}$, a healthy relative error is about $10^{-5}$ to $10^{-3}$.

---

## **`finite_difference_gradient()`**

Walk every index of the array, evaluate $f$ at $x_i \pm \varepsilon$, and fill one entry of the numeric gradient.

```python
def finite_difference_gradient(function: Callable[[NDArray[np.float64]], float], data: NDArray[np.float64], *, epsilon: float = 1e-5) -> NDArray[np.float64]:
    gradient = np.zeros_like(data, dtype=np.float64)
    iterator = np.nditer(data, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(data[index])
        data[index] = original + epsilon
        value_plus = float(function(data))
        data[index] = original - epsilon
        value_minus = float(function(data))
        data[index] = original
        gradient[index] = (value_plus - value_minus) / (2.0 * epsilon)
        iterator.iternext()

    return gradient
```

Explanation:

1. `function` must return a Python `float` (scalar loss).
2. Perturb one coordinate at a time; restore it before moving on.
3. Central differences cancel the leading $O(\varepsilon)$ truncation term.

---

## **`check_tensor_gradient()`**

Runs analytic `.backward()`, then rebuilds the same scalar loss from raw NumPy data for the finite difference sweep.

```python
def check_tensor_gradient(make_loss: Callable[[Tensor], Tensor], tensor: Tensor, *, epsilon: float = 1e-5, absolute_tolerance: float = 1e-4, relative_tolerance: float = 1e-3) -> tuple[bool, float]:
    tensor.zero_grad()
    loss = make_loss(tensor)
    loss.backward()
    assert tensor.grad is not None
    analytic = tensor.grad.copy()

    def function(data: NDArray[np.float64]) -> float:
        probe = Tensor(data, requires_grad=False)

        return float(make_loss(probe).data)

    numeric = finite_difference_gradient(function, tensor.data.copy(), epsilon=epsilon)
    relative_error = float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12))
    passed = bool(np.allclose(analytic, numeric, atol=absolute_tolerance, rtol=relative_tolerance))

    return passed, relative_error
```

Explanation:

1. `make_loss(tensor)` must return a scalar `Tensor` that depends on `tensor`.
2. Analytic path uses autograd; numeric path does not need `requires_grad`.
3. Returns `(passed, relative_error)` so you can assert and inspect the gap.

---

## **Check**

```python
from src import Tensor
from src.gradcheck import check_tensor_gradient

x = Tensor([0.2, -0.5, 1.1], requires_grad=True)
passed, relative_error = check_tensor_gradient(lambda tensor: tensor.relu().power(2).sum(), x)
print(passed, relative_error)
```
