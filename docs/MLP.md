# **MLP**

Stack affine maps and activations into a multilayer perceptron. `Sequential` runs modules in order and collects every layer's parameters for the optimizer.

> *`src/modules.py`*

---

## **Math**

A feedforward MLP with $L$ layers:

$$
f(x) = W_L\,\sigma(\cdots\sigma(W_1 x + b_1)\cdots) + b_L
$$

Hidden layers apply a nonlinearity $\sigma$ (ReLU, Tanh, …). Without $\sigma$, stacked linears collapse to one affine map.

Shapes for a batch:

- Input: $x \in \mathbb{R}^{N \times d_{\mathrm{in}}}$
- Hidden width: $h$
- Output: $f(x) \in \mathbb{R}^{N \times d_{\mathrm{out}}}$

---

## **The `Sequential` Class**

Chains modules left to right. Overrides `parameters()` so nested `Linear` weights are visible to SGD / Adam.

```python
class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.layers = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.layers:
            output = layer(output)

        return output

    def parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = []
        for layer in self.layers:
            parameters.extend(layer.parameters())

        return parameters

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)
```

Explanation:

1. `forward` is a left fold: each layer's output is the next layer's input.
2. `parameters()` walks `self.layers` instead of relying only on `__dict__` recursion.
3. `__iter__` lets you inspect or loop over layers.

---

## **Building an MLP**

```python
Sequential(
    Linear(d_in, h),
    ReLU(),
    Linear(h, h),
    ReLU(),
    Linear(h, d_out),
)
```

Typical pattern: `Linear` → activation → … → final `Linear` that emits logits (no Softmax when the loss folds it in).

---

## **Check**

```python
from src import Tensor, Linear, ReLU, Sequential, Adam, BCEWithLogitsLoss
from src.gradcheck import check_tensor_gradient

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
)
x = Tensor([[0.1, -0.2], [0.3, 0.4]])
y = model(x)
print(y.shape)
print(len(model.parameters()))

x = Tensor([[0.5, -1.0]], requires_grad=True)
passed, relative_error = check_tensor_gradient(lambda tensor: model(tensor).sum(), x)
print(passed, relative_error)
```
