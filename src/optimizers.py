from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from src.tensor import Tensor

class Optimizer:
    def __init__(self, parameters: Iterable[Tensor]) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

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
