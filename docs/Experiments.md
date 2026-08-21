# **Experiments**

Toy datasets and demos that prove the MLP stack works end to end. Data helpers live in `src/data.py`; runnable scripts live under `examples/`.

> *`src/data.py` · `examples/`*

---

## **Datasets**

### **`xor()`**

Four Gaussian blobs at the corners of the unit square. Labels follow $a \oplus b$, so a linear classifier fails.

```python
def xor(samples_per_class: int = 50, noise: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
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

### **`moons()`**

Two interleaving half-circles — a smooth nonlinear binary boundary.

```python
def moons(sample_count: int = 300, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    half = sample_count // 2
    theta = np.linspace(0, np.pi, half)
    first_moon = np.column_stack([np.cos(theta), np.sin(theta)])
    second_moon = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    inputs = np.vstack([first_moon, second_moon]) + noise * random_generator.standard_normal((half * 2, 2))
    targets = np.concatenate([np.zeros((half, 1)), np.ones((half, 1))])
    permutation = random_generator.permutation(len(inputs))

    return inputs[permutation].astype(np.float64), targets[permutation].astype(np.float64)
```

### **`spirals()`**

$K$ interleaved spirals for multiclass softmax + cross entropy. Targets are integer class indices of shape `(N,)`.

```python
def spirals(samples_per_class: int = 150, class_count: int = 3, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for class_index in range(class_count):
        radius = np.linspace(0.1, 1.0, samples_per_class)
        angle = np.linspace(class_index * 2.5, (class_index + 1) * 2.5, samples_per_class) + noise * random_generator.standard_normal(samples_per_class)
        points = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
        inputs_list.append(points)
        targets_list.append(np.full(samples_per_class, class_index, dtype=int))

    inputs = np.vstack(inputs_list).astype(np.float64)
    targets = np.concatenate(targets_list)
    permutation = random_generator.permutation(len(inputs))

    return inputs[permutation], targets[permutation]
```

### **`mnist()`**

Classic IDX handwritten digits. Images flatten to `(N, 784)` and scale to $[0, 1]$.

```python
def mnist(directory: Path | None = None, *, train: bool = True, flatten: bool = True, normalize: bool = True, limit: int | None = None, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    directory = directory or MNIST_DIRECTORY
    if download:
        fetch(directory)

    prefix = "train" if train else "t10k"
    images = readIdx(resolve(f"{prefix}-images-idx3-ubyte", directory))
    labels = readIdx(resolve(f"{prefix}-labels-idx1-ubyte", directory))
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    inputs = images.astype(np.float64)
    if normalize:
        inputs = inputs / 255.0

    if flatten:
        inputs = inputs.reshape(inputs.shape[0], -1)

    return inputs, labels.astype(np.int64)
```

---

## **Demos**

| Script | Task | Loss | Target |
|--------|------|------|--------|
| `examples/train_xor.py` | binary XOR | `BCEWithLogitsLoss` | accuracy ≥ 0.95 |
| `examples/train_moons.py` | binary moons | `BCEWithLogitsLoss` | accuracy ≥ 0.90 |
| `examples/train_spirals.py` | 3-class spirals | `CrossEntropyLoss` | accuracy ≥ 0.85 |
| `examples/train_mnist.py` | MNIST digits | `CrossEntropyLoss` | test accuracy ≥ 0.95 |

```bash
python examples/train_xor.py
python examples/train_moons.py
python examples/train_spirals.py
python examples/train_mnist.py
```

```python
from src.data import mnist, fetch

fetch()
train_inputs, train_targets = mnist(train=True)
test_inputs, test_targets = mnist(train=False)
```

---

## **Check**

```python
from src import Linear, ReLU, Sequential, Adam, BCEWithLogitsLoss, train
from src.data import xor
from src.training import accuracy_binary

inputs, targets = xor(samples_per_class=40, noise=0.05, seed=0)
model = Sequential(Linear(2, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))
history = train(model, BCEWithLogitsLoss(), Adam(model.parameters(), learning_rate=0.05), inputs, targets, epochs=120, batch_size=32, metric_function=accuracy_binary, verbose=False)
print(history.metric[-1])
```