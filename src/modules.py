from __future__ import annotations
from collections.abc import Iterator
import numpy as np
from src.tensor import Tensor

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

class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()

class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()

class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()

class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        return input.softmax(axis=self.axis)

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