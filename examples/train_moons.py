import sys
from pathlib import Path
from src import Adam, BCEWithLogitsLoss, Linear, ReLU, Sequential, Tanh, train
from src.data import moons
from src.training import accuracy_binary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> None:
    inputs, targets = moons(sample_count=400, noise=0.15, seed=1)
    model = Sequential(Linear(2, 32), Tanh(), Linear(32, 32), ReLU(), Linear(32, 1))
    history = train(model, BCEWithLogitsLoss(), Adam(model.parameters(), learning_rate=0.02), inputs, targets, epochs=150, batch_size=64, metric_function=accuracy_binary, log_every=25)

    print(f"final accuracy: {history.metric[-1]:.3f}")

if __name__ == "__main__":
    main()