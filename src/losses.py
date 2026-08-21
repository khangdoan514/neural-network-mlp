from __future__ import annotations
import numpy as np
from src.modules import Module
from src.tensor import Tensor

class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor | np.ndarray) -> Tensor:
        target_tensor = target if isinstance(target, Tensor) else Tensor(target)

        return prediction.subtract(target_tensor).power(2).mean()

class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, target: Tensor | np.ndarray) -> Tensor:
        target_tensor = target if isinstance(target, Tensor) else Tensor(target)
        absolute_logits = logits.relu().add(logits.negate().relu())
        loss = logits.relu().subtract(logits.multiply(target_tensor)).add(absolute_logits.negate().exp().add(1.0).log())

        return loss.mean()

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