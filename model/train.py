from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


@dataclass
class Metrics:
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    preds = []
    golds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy().tolist())
            golds.extend(y.cpu().numpy().tolist())

    return Metrics(
        loss=total_loss / len(loader.dataset),
        acc=accuracy_score(golds, preds),
        precision=precision_score(golds, preds, average="binary", zero_division=0),
        recall=recall_score(golds, preds, average="binary", zero_division=0),
        f1=f1_score(golds, preds, average="binary", zero_division=0)
    )


def train_model(model, train_loader, val_loader, test_loader, args, model_name):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_f1 = -1
    best_state = None
    patience_count = 0

    print(f"\n===== Training {model_name} on {device} =====")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"{model_name} Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item() * y.size(0)

        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={total_loss / len(train_loader.dataset):.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc:.4f} | "
            f"val_f1={val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"[{model_name} Test] "
        f"Accuracy={test_metrics.acc:.4f}, "
        f"Precision={test_metrics.precision:.4f}, "
        f"Recall={test_metrics.recall:.4f}, "
        f"F1={test_metrics.f1:.4f}"
    )

    return test_metrics