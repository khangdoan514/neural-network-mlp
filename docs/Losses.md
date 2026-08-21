# **Losses**

Scalar objectives $L(\hat y, y)$ that turn network outputs into something you can call `.backward()` on. Each loss is a `Module` whose `forward` returns a scalar `Tensor`.

> *`src/losses.py`*

---

## **Math**

**MSE** (regression)

$$
L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

**BCE with logits** (binary classification, stable form)

For logits $z$ and targets $y \in \{0,1\}$:

$$
L = \frac{1}{N}\sum_i \Big(\max(z_i,0) - z_i y_i + \log(1 + e^{-|z_i|})\Big)
$$

**Cross entropy** (multi-class, from logits)

For logits $z \in \mathbb{R}^{N \times C}$ and class indices $c_i$:

$$
L = -\frac{1}{N}\sum_i \log \mathrm{softmax}(z_i)_{c_i}
$$

Using logits + log-softmax (or the BCE-with-logits identity) avoids an explicit Softmax layer and stays numerically stable.

---

## **`MSELoss`**

Mean squared error: `mean((prediction - target)^2)`.

```python
class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor | np.ndarray) -> Tensor:
        target_tensor = target if isinstance(target, Tensor) else Tensor(target)

        return prediction.subtract(target_tensor).power(2).mean()
```

---

## **`BCEWithLogitsLoss`**

Binary cross entropy on raw logits. Uses the stable rewrite above so large positive or negative logits do not overflow.

```python
class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, target: Tensor | np.ndarray) -> Tensor:
        target_tensor = target if isinstance(target, Tensor) else Tensor(target)
        absolute_logits = logits.relu().add(logits.negate().relu())
        loss = logits.relu().subtract(logits.multiply(target_tensor)).add(absolute_logits.negate().exp().add(1.0).log())

        return loss.mean()
```

Explanation:

1. `|z|` is built as `relu(z) + relu(-z)`.
2. Targets should be floats in $\{0,1\}$ with the same shape as `logits`.
3. Prefer this over `Sigmoid` + BCE when training binary classifiers.

---

## **`CrossEntropyLoss`**

Multi-class cross entropy from logits. `logits` has shape `(N, C)`; `target` has integer class indices of shape `(N,)`.

```python
class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: Tensor | np.ndarray) -> Tensor:
        if isinstance(target, Tensor):
            target_index = target.data.astype(int).reshape(-1)

        else:
            target_index = np.asarray(target, dtype=int).reshape(-1)

        maximum = np.max(logits.data, axis=1, keepdims=True)
        shifted = logits.subtract(maximum)
        log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
        log_probabilities = shifted.subtract(log_sum_exp)
        batch_size = target_index.shape[0]
        rows = np.arange(batch_size)
        chosen = log_probabilities.get_item((rows, target_index))

        return chosen.mean().negate()
```

Explanation:

1. Subtract the row max before `exp` / `log` (log-sum-exp trick).
2. Index the true class log-probability with `get_item`, then mean and negate.
3. Gradients flow back into `logits` through the autograd graph from step 02.

---

## **Check**

```python
from src import Tensor, MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from src.gradcheck import check_tensor_gradient

prediction = Tensor([[0.5], [-0.2]], requires_grad=True)
mse = MSELoss()(prediction, [[1.0], [0.0]])
print(float(mse.data))

logits = Tensor([[0.2], [-0.5], [1.0]], requires_grad=True)
passed, relative_error = check_tensor_gradient(lambda tensor: BCEWithLogitsLoss()(tensor, [[1.0], [0.0], [1.0]]), logits)
print(passed, relative_error)

logits = Tensor([[0.1, 0.2, 0.3], [0.5, -0.1, 0.0]], requires_grad=True)
ce = CrossEntropyLoss()(logits, [2, 0])
print(float(ce.data))
```
