# **Optimizers**

Turn $\nabla_\theta L$ into parameter updates. After `loss.backward()`, call `optimizer.step()` to change each parameter in place.

> *`src/optimizers.py`*

---

## **Math**

**SGD**

$$
\theta \leftarrow \theta - \eta g
$$

**SGD with momentum**

$$
v \leftarrow \mu v + g, \qquad \theta \leftarrow \theta - \eta v
$$

**Adam**

$$
m \leftarrow \beta_1 m + (1-\beta_1)g
$$

$$
v \leftarrow \beta_2 v + (1-\beta_2)g^{2}
$$

$$
\hat{m} = \frac{m}{1-\beta_1^{t}}, \quad \hat{v} = \frac{v}{1-\beta_2^{t}}
$$

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
$$

Optional L2 weight decay adds $\lambda \theta$ to $g$ before the update.

---

## **The `Optimizer` Class**

Shared bookkeeping for every optimizer.

```python
class Optimizer:
    def __init__(self, parameters: Iterable[Tensor]) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError
```

Explanation:

1. Only tensors with `requires_grad=True` are tracked.
2. `zero_grad()` clears `.grad` before the next minibatch.
3. Subclasses implement `step()`.

---

## **`SGD`**

Plain SGD, or SGD with momentum when `momentum > 0`.

```python
class SGD(Optimizer):
    def __init__(self, parameters: Iterable[Tensor], learning_rate: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            gradient = parameter.grad
            if self.weight_decay != 0.0:
                gradient = gradient + self.weight_decay * parameter.data

            if self.momentum != 0.0:
                self.velocity[index] = self.momentum * self.velocity[index] + gradient
                gradient = self.velocity[index]

            parameter.data = parameter.data - self.learning_rate * gradient
```

---

## **`Adam`**

Adaptive moments with bias correction.

```python
class Adam(Optimizer):
    def __init__(self, parameters: Iterable[Tensor], learning_rate: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.timestep = 0
        self.first_moment = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.second_moment = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        self.timestep += 1
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            gradient = parameter.grad
            if self.weight_decay != 0.0:
                gradient = gradient + self.weight_decay * parameter.data

            self.first_moment[index] = self.beta1 * self.first_moment[index] + (1.0 - self.beta1) * gradient
            self.second_moment[index] = self.beta2 * self.second_moment[index] + (1.0 - self.beta2) * (gradient * gradient)
            first_moment_hat = self.first_moment[index] / (1.0 - self.beta1**self.timestep)
            second_moment_hat = self.second_moment[index] / (1.0 - self.beta2**self.timestep)
            parameter.data = parameter.data - self.learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + self.epsilon)
```

Explanation:

1. `first_moment` / `second_moment` are $m$ and $v$.
2. Bias correction divides by $1-\beta^{t}$ so early steps are not shrunk toward zero.
3. Updates write directly into `parameter.data`; the autograd graph is not involved in the step.

---

## **Check**

```python
import numpy as np
from src import Tensor, Adam, SGD

parameter = Tensor([1.0, 2.0], requires_grad=True)
parameter.grad = np.array([0.5, -0.25])

optimizer = Adam([parameter], learning_rate=0.1)
before = parameter.data.copy()
optimizer.step()
print(before, parameter.data)

sgd = SGD([parameter], learning_rate=0.01, momentum=0.9)
parameter.grad = np.array([0.1, 0.1])
sgd.step()
```
