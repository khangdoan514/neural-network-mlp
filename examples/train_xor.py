import sys
from pathlib import Path
from src import Adam, BCEWithLogitsLoss, Linear, ReLU, Sequential, train
from src.data import make_xor
from src.training import accuracy_binary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    inputs, targets = make_xor(samples_per_class=50, noise=0.08, seed=0)
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 1),
    )
    history = train(
        model,
        BCEWithLogitsLoss(),
        Adam(model.parameters(), learning_rate=0.05),
        inputs,
        targets,
        epochs=150,
        batch_size=32,
        metric_function=accuracy_binary,
        log_every=25,
    )
    print(f"final accuracy: {history.metric[-1]:.3f}")

if __name__ == "__main__":
    main()