import sys
from pathlib import Path
import numpy as np
from src import Adam, CrossEntropyLoss, Linear, ReLU, Sequential, Tensor, train
from src.data import mnist
from src.training import accuracy_multiclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    train_inputs, train_targets = mnist(train=True, flatten=True, normalize=True)
    test_inputs, test_targets = mnist(train=False, flatten=True, normalize=True)

    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 10))
    history = train(model, CrossEntropyLoss(), Adam(model.parameters(), learning_rate=0.001), train_inputs, train_targets, epochs=5, batch_size=128, metric_function=accuracy_multiclass, log_every=1)

    model.eval_mode()
    logits = model(Tensor(test_inputs))
    predictions = np.argmax(logits.data, axis=1)
    test_accuracy = float(np.mean(predictions == test_targets))

    print(f"train accuracy: {history.metric[-1]:.3f}")
    print(f"test accuracy:  {test_accuracy:.3f}")
    print("sample predictions:")
    for index in range(10):
        print(f"  true={int(test_targets[index])} pred={int(predictions[index])}")

if __name__ == "__main__":
    main()