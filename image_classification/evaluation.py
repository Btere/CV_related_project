from pathlib import Path

import torch
from dataset import load_dataset
from model import FashionMnistModel, evaluate_model

def main() -> None:
    """Loads a model and runs evaluaton on it."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionMnistModel(device).to(device)
    

    _, _, X_val, y_val = load_dataset()
    model.evaluate_model(X_val, y_val)


if __name__ == "__main__":
    main()