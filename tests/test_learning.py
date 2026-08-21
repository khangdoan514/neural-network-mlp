from src.data import xor
from src.losses import BCEWithLogitsLoss
from src.modules import Linear, ReLU, Sequential
from src.optimizers import Adam
from src.training import accuracy_binary, train

def test_xor_learns():
    inputs, targets = xor(samples_per_class=40, noise=0.05, seed=0)
    model = Sequential(Linear(2, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))
    history = train(model, BCEWithLogitsLoss(), Adam(model.parameters(), learning_rate=0.05), inputs, targets, epochs=120, batch_size=32, metric_function=accuracy_binary, verbose=False)
    assert history.metric[-1] >= 0.90
    assert history.loss[-1] < history.loss[0]