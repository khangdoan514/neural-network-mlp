from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from src.modules import Module
from src.optimizers import Optimizer
from src.tensor import Tensor

@dataclass
class TrainingHistory:
    loss: list[float]
    metric: list[float]

def accuracy_binary(logits: Tensor, target: np.ndarray) -> float:
    prediction = (logits.data >= 0.0).astype(np.float64)
    return float(np.mean(prediction.reshape(-1) == target.reshape(-1)))

def accuracy_multiclass(logits: Tensor, target: np.ndarray) -> float:
    prediction = np.argmax(logits.data, axis=1)
    return float(np.mean(prediction == target.reshape(-1)))

def train(
    model: Module,
    loss_function: Module,
    optimizer: Optimizer,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int | None = None,
    metric_function: Callable[[Tensor, np.ndarray], float] | None = None,
    random_generator: np.random.Generator | None = None,
    verbose: bool = True,
    log_every: int = 10,
) -> TrainingHistory:
    random_generator = random_generator or np.random.default_rng(0)
    sample_count = inputs.shape[0]
    batch_size = batch_size or sample_count
    history = TrainingHistory(loss=[], metric=[])

    for epoch in range(1, epochs + 1):
        permutation = random_generator.permutation(sample_count)
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets[permutation]
        batch_losses: list[float] = []
        batch_metrics: list[float] = []

        for start in range(0, sample_count, batch_size):
            batch_inputs = shuffled_inputs[start : start + batch_size]
            batch_targets = shuffled_targets[start : start + batch_size]
            optimizer.zero_grad()
            prediction = model(Tensor(batch_inputs))
            loss = loss_function(prediction, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.data))
            if metric_function is not None:
                batch_metrics.append(metric_function(prediction.detach(), batch_targets))

        epoch_loss = float(np.mean(batch_losses))
        epoch_metric = float(np.mean(batch_metrics)) if batch_metrics else float("nan")
        history.loss.append(epoch_loss)
        history.metric.append(epoch_metric)

        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            if metric_function is None:
                print(f"epoch {epoch:4d} | loss {epoch_loss:.6f}")
            else:
                print(f"epoch {epoch:4d} | loss {epoch_loss:.6f} | metric {epoch_metric:.4f}")

    return history
