# **Experiments**

Toy datasets and demos that prove the MLP stack works end to end. Data helpers live in `src/data.py`; runnable scripts live under `examples/`.

> *`src/data.py` · `examples/`*

---

## **Datasets**

### **`make_xor()`**

Four Gaussian blobs at the corners of the unit square. Labels follow $a \oplus b$, so a linear classifier fails.

```python
def make_xor(
    samples_per_class: int = 50,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    center_labels = np.array([0.0, 1.0, 1.0, 0.0])
    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for center, label in zip(centers, center_labels, strict=True):
        points = center + noise * random_generator.standard_normal((samples_per_class, 2))
        inputs_list.append(points)
        targets_list.append(np.full((samples_per_class, 1), label))
    inputs = np.vstack(inputs_list).astype(np.float64)
    targets = np.vstack(targets_list).astype(np.float64)
    permutation = random_generator.permutation(len(inputs))
    return inputs[permutation], targets[permutation]
```

### **`make_moons()`**

Two interleaving half-circles — a smooth nonlinear binary boundary.

```python
def make_moons(
    sample_count: int = 300,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    half = sample_count // 2
    theta = np.linspace(0, np.pi, half)
    first_moon = np.column_stack([np.cos(theta), np.sin(theta)])
    second_moon = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    inputs = np.vstack([first_moon, second_moon]) + noise * random_generator.standard_normal(
        (half * 2, 2)
    )
    targets = np.concatenate([np.zeros((half, 1)), np.ones((half, 1))])
    permutation = random_generator.permutation(len(inputs))
    return inputs[permutation].astype(np.float64), targets[permutation].astype(np.float64)
```

### **`make_spirals()`**

$K$ interleaved spirals for multiclass softmax + cross entropy. Targets are integer class indices of shape `(N,)`.

```python
def make_spirals(
    samples_per_class: int = 150,
    class_count: int = 3,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for class_index in range(class_count):
        radius = np.linspace(0.1, 1.0, samples_per_class)
        angle = np.linspace(
            class_index * 2.5,
            (class_index + 1) * 2.5,
            samples_per_class,
        ) + noise * random_generator.standard_normal(samples_per_class)
        points = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
        inputs_list.append(points)
        targets_list.append(np.full(samples_per_class, class_index, dtype=int))
    inputs = np.vstack(inputs_list).astype(np.float64)
    targets = np.concatenate(targets_list)
    permutation = random_generator.permutation(len(inputs))
    return inputs[permutation], targets[permutation]
```

---

## **Demos**

| Script | Task | Loss | Target |
|--------|------|------|--------|
| `examples/train_xor.py` | binary XOR | `BCEWithLogitsLoss` | accuracy ≥ 0.95 |
| `examples/train_moons.py` | binary moons | `BCEWithLogitsLoss` | accuracy ≥ 0.90 |
| `examples/train_spirals.py` | 3-class spirals | `CrossEntropyLoss` | accuracy ≥ 0.85 |

```bash
python examples/train_xor.py
python examples/train_moons.py
python examples/train_spirals.py
```

---

## **Check**

```python
from src import Linear, ReLU, Sequential, Adam, BCEWithLogitsLoss, train
from src.data import make_xor
from src.training import accuracy_binary

inputs, targets = make_xor(samples_per_class=40, noise=0.05, seed=0)
model = Sequential(Linear(2, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))
history = train(
    model,
    BCEWithLogitsLoss(),
    Adam(model.parameters(), learning_rate=0.05),
    inputs,
    targets,
    epochs=120,
    batch_size=32,
    metric_function=accuracy_binary,
    verbose=False,
)
print(history.metric[-1])
```