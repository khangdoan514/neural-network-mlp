import numpy as np
from src.gradcheck import check_tensor_gradient
from src.losses import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from src.modules import Linear, ReLU, Softmax, Sequential
from src.optimizers import Adam, SGD
from src.tensor import Tensor

def test_linear_shape():
    layer = Linear(4, 3)
    input_tensor = Tensor(np.ones((2, 4)))
    output = layer(input_tensor)
    assert output.shape == (2, 3)
    assert len(layer.parameters()) == 2

def test_linear_gradcheck():
    layer = Linear(3, 2)
    input_tensor = Tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    passed, relative_error = check_tensor_gradient(lambda tensor: layer(tensor).sum(), input_tensor)
    assert passed, relative_error

def test_activation_modules():
    input_tensor = Tensor([[-1.0, 0.5, 2.0]], requires_grad=True)
    passed_relu, relative_error = check_tensor_gradient(lambda tensor: ReLU()(tensor).power(2).sum(), input_tensor)
    assert passed_relu, relative_error

    logits = Tensor([[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]], requires_grad=True)
    probabilities = Softmax(axis=1)(logits)
    assert np.allclose(probabilities.data.sum(axis=1), 1.0)

def test_sequential_forward():
    model = Sequential(Linear(2, 5), ReLU(), Linear(5, 1))
    output = model(Tensor([[0.1, -0.2], [0.3, 0.4]]))
    assert output.shape == (2, 1)
    assert len(model.parameters()) == 4

def test_sequential_gradcheck():
    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
    input_tensor = Tensor([[0.5, -1.0]], requires_grad=True)
    passed, relative_error = check_tensor_gradient(lambda tensor: model(tensor).sum(), input_tensor)
    assert passed, relative_error

def test_mse_decreases_with_sgd():
    model = Sequential(Linear(1, 8), ReLU(), Linear(8, 1))
    inputs = np.linspace(-1, 1, 40).reshape(-1, 1)
    targets = (0.5 * inputs + 0.1).astype(np.float64)
    optimizer = SGD(model.parameters(), learning_rate=0.05)
    loss_function = MSELoss()
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        prediction = model(Tensor(inputs))
        loss = loss_function(prediction, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] * 0.5

def test_bce_and_ce_finite():
    logits = Tensor([[0.2], [-0.5], [1.0]], requires_grad=True)
    targets = np.array([[1.0], [0.0], [1.0]])
    loss = BCEWithLogitsLoss()(logits, targets)
    loss.backward()
    assert np.isfinite(loss.data)

    logits2 = Tensor([[0.1, 0.2, 0.3], [0.5, -0.1, 0.0]], requires_grad=True)
    targets2 = np.array([2, 0])
    loss2 = CrossEntropyLoss()(logits2, targets2)
    loss2.backward()
    assert np.isfinite(loss2.data)

def test_loss_gradcheck():
    prediction = Tensor([[0.5], [-0.2]], requires_grad=True)
    target = [[1.0], [0.0]]
    passed_mse, relative_error_mse = check_tensor_gradient(lambda tensor: MSELoss()(tensor, target), prediction)
    assert passed_mse, relative_error_mse

    logits = Tensor([[0.2], [-0.5], [1.0]], requires_grad=True)
    passed_bce, relative_error_bce = check_tensor_gradient(lambda tensor: BCEWithLogitsLoss()(tensor, [[1.0], [0.0], [1.0]]), logits)
    assert passed_bce, relative_error_bce

def test_adam_steps():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([0.5, -0.25])
    optimizer = Adam([parameter], learning_rate=0.1)
    previous_data = parameter.data.copy()
    optimizer.step()
    assert not np.allclose(parameter.data, previous_data)