import sys
from pathlib import Path
from src import Adam, CrossEntropyLoss, Linear, ReLU, Sequential, train
from src.data import make_spirals
from src.training import accuracy_multiclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    inputs, targets = make_spirals(samples_per_class=120, class_count=3, noise=0.15, seed=0)
    model = Sequential(
        Linear(2, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 3),
    )
    history = train(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), learning_rate=0.02),
        inputs,
        targets,
        epochs=200,
        batch_size=64,
        metric_function=accuracy_multiclass,
        log_every=25,
    )
    print(f"final accuracy: {history.metric[-1]:.3f}")

if __name__ == "__main__":
    main()