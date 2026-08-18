import numpy as np
from src.gradcheck import check_tensor_gradient
from src.tensor import Tensor

def test_add_mul_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    loss = a.multiply(b).add(a).sum()
    loss.backward()
    assert np.allclose(a.grad, b.data + 1.0)
    assert np.allclose(b.grad, a.data)

def test_matmul_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 1.0], [-1.0, 2.0]], requires_grad=True)
    loss = a.matmul(b).sum()
    loss.backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

def test_relu_gradcheck():
    input_tensor = Tensor(np.array([-1.0, 0.5, 2.0]), requires_grad=True)
    passed, relative_error = check_tensor_gradient(
        lambda tensor: tensor.relu().power(2).sum(),
        input_tensor,
    )
    assert passed, relative_error

def test_tanh_sigmoid_gradcheck():
    input_tensor = Tensor(np.array([-0.3, 0.2, 1.0]), requires_grad=True)
    passed_tanh, _ = check_tensor_gradient(lambda tensor: tensor.tanh().sum(), input_tensor)
    input_tensor2 = Tensor(np.array([-0.3, 0.2, 1.0]), requires_grad=True)
    passed_sigmoid, _ = check_tensor_gradient(lambda tensor: tensor.sigmoid().sum(), input_tensor2)
    assert passed_tanh and passed_sigmoid

def test_softmax_sums_to_one():
    input_tensor = Tensor([[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]], requires_grad=True)
    probabilities = input_tensor.softmax(axis=1)
    assert np.allclose(probabilities.data.sum(axis=1), 1.0)

def test_broadcast_add_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)
    loss = a.add(b).sum()
    loss.backward()
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.array([2.0, 2.0]))