from __future__ import annotations
import gzip
import struct
import urllib.request
from pathlib import Path
import numpy as np

MNIST_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "mnist"
MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

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

def resolve(file_name: str, directory: Path) -> Path:
    candidates = (
        directory / file_name,
        directory / f"{file_name}.gz",
        directory / file_name.replace("-idx", ".idx"),
        directory / f"{file_name.replace('-idx', '.idx')}.gz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"missing MNIST file '{file_name}' under {directory}")

def fetch(directory: Path | None = None) -> Path:
    directory = directory or MNIST_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    for file_name, url in MNIST_URLS.items():
        archive_path = directory / file_name
        extracted_path = directory / file_name.removesuffix(".gz")
        if archive_path.exists() or extracted_path.exists():
            continue

        urllib.request.urlretrieve(url, archive_path)

    for archive_path in directory.glob("*.gz"):
        extracted_path = directory / archive_path.name.removesuffix(".gz")
        if extracted_path.exists():
            continue

        with gzip.open(archive_path, "rb") as archive_file, open(extracted_path, "wb") as output_file:
            output_file.write(archive_file.read())

    return directory

def readIdx(path: Path) -> np.ndarray:
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rb") as file_handle:
        zero, data_type, dimension_count = struct.unpack(">HBB", file_handle.read(4))
        if zero != 0:
            raise ValueError(f"invalid IDX header in {path}")

        if data_type != 0x08:
            raise ValueError(f"unsupported IDX data type in {path}")

        shape = struct.unpack(">" + "I" * dimension_count, file_handle.read(4 * dimension_count))
        values = np.frombuffer(file_handle.read(), dtype=np.uint8)

        return values.reshape(shape)

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