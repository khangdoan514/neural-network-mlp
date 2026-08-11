from __future__ import annotations
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np
from src.tensor import Tensor

def finite_difference_gradient(
    function: Callable[[NDArray[np.float64]], float],
    data: NDArray[np.float64],
    *,
    epsilon: float = 1e-5,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(data, dtype=np.float64)
    iterator = np.nditer(data, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(data[index])
        data[index] = original + epsilon
        value_plus = float(function(data))
        data[index] = original - epsilon
        value_minus = float(function(data))
        data[index] = original
        gradient[index] = (value_plus - value_minus) / (2.0 * epsilon)
        iterator.iternext()
    return gradient

def check_tensor_gradient(
    make_loss: Callable[[Tensor], Tensor],
    tensor: Tensor,
    *,
    epsilon: float = 1e-5,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-3,
) -> tuple[bool, float]:
    tensor.zero_grad()
    loss = make_loss(tensor)
    loss.backward()
    assert tensor.grad is not None
    analytic = tensor.grad.copy()

    def function(data: NDArray[np.float64]) -> float:
        probe = Tensor(data, requires_grad=False)
        return float(make_loss(probe).data)

    numeric = finite_difference_gradient(function, tensor.data.copy(), epsilon=epsilon)
    relative_error = float(
        np.linalg.norm(analytic - numeric)
        / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12)
    )
    passed = bool(
        np.allclose(analytic, numeric, atol=absolute_tolerance, rtol=relative_tolerance)
    )
    return passed, relative_error
