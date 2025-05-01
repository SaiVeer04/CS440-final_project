import logging
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TorchNN(nn.Module):
    def __init__(self, input_size: int, hidden1: int,
                 hidden2: int, output_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TrainerConfig:
    def __init__(self,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 10,
                 device: Optional[str] = None) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")


def make_dataloader(
        X: torch.Tensor, y: torch.Tensor,
        batch_size: int, shuffle: bool = True
) -> DataLoader:
    if X.size(0) != y.size(0):
        raise ValueError("Features and labels must have same first dimension")
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_and_evaluate(
        model: nn.Module,
        X_train: torch.Tensor, y_train: torch.Tensor,
        X_val: torch.Tensor, y_val: torch.Tensor,
        config: TrainerConfig
) -> Dict[str, Any]:
    try:
        # Prepare data loaders
        train_loader = make_dataloader(X_train, y_train, config.batch_size, shuffle=True)
        val_loader = make_dataloader(X_val, y_val, config.batch_size, shuffle=False)

        device = torch.device(config.device)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

        logger.info("Starting training...")
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch}/{config.epochs} — loss: {avg_loss:.4f}")

        # Evaluation helper function
        def evaluate(loader: DataLoader) -> float:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            return correct / total if total > 0 else 0.0

        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)
        logger.info(f"Training complete. Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
        }

    except Exception as e:
        logger.exception("Error during training/evaluation")
        raise

