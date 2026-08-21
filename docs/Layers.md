# **Layers**

Parameterize the affine map $y = xW + b$. A `Module` owns learnable `Tensor` parameters; `Linear` is the first layer that uses them.

> *`src/modules.py`*

---

## **Math**

Affine map for a batch of inputs:

$$
y = xW + b
$$

Shapes:

- $x \in \mathbb{R}^{N \times d_{\mathrm{in}}}$
- $W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$
- $b \in \mathbb{R}^{d_{\mathrm{out}}}$
- $y \in \mathbb{R}^{N \times d_{\mathrm{out}}}$

He-style initialization for ReLU networks:

$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\mathrm{in}}}}\right)
$$

Bias starts at zero.

---

## **The `Module` Class**

Base class for anything with learnable parameters.

| Method | Role |
|--------|------|
| `parameters()` | Collect every `Tensor` with `requires_grad=True` on this module |
| `zero_grad()` | Zero `.grad` on all parameters |
| `train_mode()` / `eval_mode()` | Toggle training flag (used later for dropout, batch norm, etc.) |
| `forward()` | Define the forward pass |
| `__call__()` | Calls `forward()` |

```python
class Module:
    def parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                parameters.append(value)

            elif isinstance(value, Module):
                parameters.extend(value.parameters())

            elif isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, Module):
                        parameters.extend(item.parameters())

                    elif isinstance(item, Tensor) and item.requires_grad:
                        parameters.append(item)

        return parameters

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train_mode(self) -> None:
        self._training = True

    def eval_mode(self) -> None:
        self._training = False

    @property
    def training(self) -> bool:
        return getattr(self, "_training", True)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
```

Explanation:

1. Optimizers call `model.parameters()` to get every weight and bias.
2. Each training step starts with `zero_grad()` so gradients do not accumulate across minibatches.
3. Subclasses implement `forward()`; callers use `module(x)` instead.

---

## **The `Linear` Class**

Fully connected layer: `in_features` → `out_features`. Weight shape is `(in_features, out_features)` so batched input `(N, in_features)` maps to `(N, out_features)` via `matmul`.

| Field | Shape | Role |
|-------|-------|------|
| `weight` | `(in_features, out_features)` | Learnable matrix $W$ |
| `bias` | `(out_features,)` | Learnable vector $b$ (optional) |

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, random_generator: np.random.Generator | None = None) -> None:
        super().__init__()
        random_generator = random_generator or np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(random_generator.normal(0.0, scale, size=(in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        output = input.matmul(self.weight)
        if self.bias is not None:
            output = output.add(self.bias)

        return output
```

Explanation:

1. `forward` is only `matmul` plus broadcast `add` — autograd from step 2 handles gradients.
2. `bias=False` drops the bias term for layers that only need $xW$.
3. `random_generator` matches `Tensor.randn` for reproducible init in tests.

---

## **Check**

```python
from src import Tensor
from src.gradcheck import check_tensor_gradient
from src.modules import Linear

layer = Linear(3, 2)
input = Tensor([[1.0, 0.0, -1.0]], requires_grad=True)
output = layer(input)
print(output.shape)

passed, relative_error = check_tensor_gradient(lambda tensor: layer(tensor).sum(), input)
print(passed, relative_error)
```
