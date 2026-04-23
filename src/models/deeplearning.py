import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score,
    fbeta_score, brier_score_loss
)

# Torch datasets
def make_loader(X, y, batch_size=4096, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t),
                    batch_size=batch_size, shuffle=shuffle)

class FraudMLP(nn.Module):
    """
    Simple residual MLP for tabular fraud detection.
    BatchNorm + Dropout for regularisation.
    """
    def __init__(self, in_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))   # raw logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
    

def train_mlp(DEVICE, model, train_loader, y_train, test_loader, y_test,
              pos_weight_val, epochs=30, lr=1e-3, patience=5):
    
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=2)

    best_prauc  = 0
    best_state  = None
    patience_cnt = 0
    history     = []

    model.to(DEVICE)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        # eval
        model.eval()
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                logits = model(X_batch.to(DEVICE))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        y_prob_ep = np.array(all_probs)
        prauc     = average_precision_score(y_test, y_prob_ep)
        brier     = brier_score_loss(y_test, y_prob_ep)
        scheduler.step(prauc)

        history.append({'epoch': epoch, 'loss': total_loss/len(y_train),
                        'pr_auc': prauc, 'brier': brier})

        print(f"Epoch {epoch:03d} | loss={total_loss/len(y_train):.5f} "
              f"| PR-AUC={prauc:.5f} | Brier={brier:.5f}")

        # early stopping
        if prauc > best_prauc:
            best_prauc  = prauc
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def find_optimal_threshold(y_true, y_prob, beta=2):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]
    denom = (beta**2 * p) + r
    fbeta = np.where(denom > 0, (1 + beta**2) * p * r / denom, 0)
    best_idx = np.argmax(fbeta)
    return thresholds[best_idx]

def evaluate(label, y_true, y_prob, threshold, beta=2):
    y_pred = (y_prob >= threshold).astype(int)
    
    print(f"-- {label} (thr={threshold:.4f}) --")
    print(f"  PR-AUC    : {average_precision_score(y_true, y_prob):.5f}")
    print(f"  Brier     : {brier_score_loss(y_true, y_prob):.5f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F{beta}        : {fbeta_score(y_true, y_pred, beta=beta):.4f}")

    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))