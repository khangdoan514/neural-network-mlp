from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

def as_data(x: ArrayLike | Tensor) -> NDArray[np.float64]:
    if isinstance(x, Tensor):
        return x.data

    return np.asarray(x, dtype=np.float64)

def unbroadcast(grad: NDArray[np.float64], shape: tuple[int, ...]) -> NDArray[np.float64]:
    if grad.shape == shape:
        return grad

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for index, (grad_size, shape_size) in enumerate(zip(grad.shape, shape, strict=True)):
        if shape_size == 1 and grad_size != 1:
            grad = grad.sum(axis=index, keepdims=True)

    return grad.reshape(shape)

class Tensor:
    def __init__(self, data: ArrayLike | Tensor, *, requires_grad: bool = False, children: Iterable[Tensor] = (), operation: str = "", gradient_function: Callable[[], None] | None = None) -> None:
        if isinstance(data, Tensor):
            self.data = data.data.copy()

        else:
            self.data = np.asarray(data, dtype=np.float64)

        self.requires_grad = bool(requires_grad) or any(child.requires_grad for child in children)
        self.grad: NDArray[np.float64] | None = np.zeros_like(self.data) if self.requires_grad else None
        self.parents = tuple(children)
        self.operation = operation
        self.gradient_function = gradient_function or (lambda: None)

    @staticmethod
    def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        return Tensor(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        return Tensor(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, random_generator: np.random.Generator | None = None) -> Tensor:
        random_generator = random_generator or np.random.default_rng()

        return Tensor(random_generator.standard_normal(shape, dtype=np.float64), requires_grad=requires_grad)

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> NDArray[np.float64]:
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, gradient: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward called on a tensor that does not require grad")

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("backward for non-scalar requires an explicit gradient")

            gradient = np.ones_like(self.data)

        else:
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.shape != self.data.shape:
                raise ValueError(f"gradient shape {gradient.shape} != tensor shape {self.data.shape}")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return

            visited.add(id(node))
            for parent in node.parents:
                build(parent)

            topo.append(node)

        build(self)
        self.grad = gradient.copy()
        for node in reversed(topo):
            node.gradient_function()

    def binary(self, other: ArrayLike | Tensor, *, forward: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], backward_for_a: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], backward_for_b: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], operation: str) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other)
        out_data = forward(self.data, right.data)
        out = Tensor(out_data, requires_grad=self.requires_grad or right.requires_grad, children=(self, right), operation=operation)

        def local_backward() -> None:
            assert out.grad is not None
            if self.requires_grad:
                assert self.grad is not None
                self.grad += unbroadcast(backward_for_a(out.grad, self.data, right.data), self.data.shape)

            if right.requires_grad:
                assert right.grad is not None
                right.grad += unbroadcast(backward_for_b(out.grad, self.data, right.data), right.data.shape)

        out.gradient_function = local_backward

        return out

    def add(self, other: ArrayLike | Tensor) -> Tensor:
        return self.binary(other, forward=lambda left, right: left + right, backward_for_a=lambda gradient, left, right: gradient, backward_for_b=lambda gradient, left, right: gradient, operation="add")

    def add_right(self, other: ArrayLike | Tensor) -> Tensor:
        return self.add(other)

    def subtract(self, other: ArrayLike | Tensor) -> Tensor:
        return self.binary(other, forward=lambda left, right: left - right, backward_for_a=lambda gradient, left, right: gradient, backward_for_b=lambda gradient, left, right: -gradient, operation="sub")

    def subtract_right(self, other: ArrayLike | Tensor) -> Tensor:
        return Tensor(other).subtract(self)

    def multiply(self, other: ArrayLike | Tensor) -> Tensor:
        return self.binary(other, forward=lambda left, right: left * right, backward_for_a=lambda gradient, left, right: gradient * right, backward_for_b=lambda gradient, left, right: gradient * left, operation="mul")

    def multiply_right(self, other: ArrayLike | Tensor) -> Tensor:
        return self.multiply(other)

    def divide(self, other: ArrayLike | Tensor) -> Tensor:
        return self.binary(other, forward=lambda left, right: left / right, backward_for_a=lambda gradient, left, right: gradient / right, backward_for_b=lambda gradient, left, right: -gradient * left / (right**2), operation="div")

    def divide_right(self, other: ArrayLike | Tensor) -> Tensor:
        return Tensor(other).divide(self)

    def power(self, exponent: float) -> Tensor:
        power_value = float(exponent)
        out = Tensor(self.data**power_value, requires_grad=self.requires_grad, children=(self,), operation="pow")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad * power_value * (self.data ** (power_value - 1.0))

        out.gradient_function = local_backward

        return out

    def negate(self) -> Tensor:
        return self.multiply(-1.0)

    def matmul(self, other: ArrayLike | Tensor) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ right.data, requires_grad=self.requires_grad or right.requires_grad, children=(self, right), operation="matmul")

        def local_backward() -> None:
            assert out.grad is not None
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad @ right.data.T

            if right.requires_grad:
                assert right.grad is not None
                right.grad += self.data.T @ out.grad

        out.gradient_function = local_backward

        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, children=(self,), operation="transpose")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                if axes is None:
                    self.grad += out.grad.T

                else:
                    inverse = np.argsort(axes)
                    self.grad += out.grad.transpose(inverse)

        out.gradient_function = local_backward

        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, children=(self,), operation="sum")

        def local_backward() -> None:
            if not self.requires_grad:
                return

            assert self.grad is not None and out.grad is not None
            gradient = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * gradient

                return

            if not keepdims:
                gradient = np.expand_dims(gradient, axis=axis)

            self.grad += np.ones_like(self.data) * gradient

        out.gradient_function = local_backward

        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.data.size

        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.data.shape[axis_index] for axis_index in axes]))

        return self.sum(axis=axis, keepdims=keepdims).divide(float(count))

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, children=(self,), operation="reshape")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad.reshape(self.data.shape)

        out.gradient_function = local_backward

        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, children=(self,), operation="relu")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad * (self.data > 0.0)

        out.gradient_function = local_backward

        return out

    def tanh(self) -> Tensor:
        values = np.tanh(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, children=(self,), operation="tanh")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad * (1.0 - values**2)

        out.gradient_function = local_backward

        return out

    def sigmoid(self) -> Tensor:
        values = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(values, requires_grad=self.requires_grad, children=(self,), operation="sigmoid")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad * values * (1.0 - values)

        out.gradient_function = local_backward

        return out

    def exp(self) -> Tensor:
        values = np.exp(self.data)
        out = Tensor(values, requires_grad=self.requires_grad, children=(self,), operation="exp")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad * values

        out.gradient_function = local_backward

        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, children=(self,), operation="log")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                self.grad += out.grad / self.data

        out.gradient_function = local_backward

        return out

    def softmax(self, axis: int = -1) -> Tensor:
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=axis, keepdims=True)
        out = Tensor(probabilities, requires_grad=self.requires_grad, children=(self,), operation="softmax")

        def local_backward() -> None:
            if not self.requires_grad:
                return

            assert self.grad is not None and out.grad is not None
            gradient = out.grad
            sum_gradient_probability = np.sum(gradient * probabilities, axis=axis, keepdims=True)
            self.grad += probabilities * (gradient - sum_gradient_probability)

        out.gradient_function = local_backward

        return out

    def representation(self) -> str:
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        operation_info = f", operation={self.operation!r}" if self.operation else ""

        return f"Tensor(shape={self.shape}{grad_info}{operation_info})\n{self.data}"

    def get_item(self, item: Any) -> Tensor:
        out_data = self.data[item]
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), operation="getitem")

        def local_backward() -> None:
            if self.requires_grad:
                assert self.grad is not None and out.grad is not None
                np.add.at(self.grad, item, out.grad)

        out.gradient_function = local_backward

        return out