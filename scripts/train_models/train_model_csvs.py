import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== CONFIG =====
CSV_30S = "./data/csv/features_30s.csv"
CSV_3S = "./data/csv/features_3s.csv"

MODEL_DIR_30S = "./models/mlp_30s"
MODEL_DIR_3S = "./models/mlp_3s"

RESULTS_DIR_30S = "./results/mlp_30s"
RESULTS_DIR_3S = "./results/mlp_3s"

for d in [MODEL_DIR_30S, MODEL_DIR_3S, RESULTS_DIR_30S, RESULTS_DIR_3S]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== REDE MLP EM TORCH =====
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ===== TREINAMENTO =====
def train_mlp(csv_path, model_dir, results_dir):
    print(f"\n=== Treinando modelo usando {csv_path} ===\n")

    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ["filename", "segment", "genre"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["genre"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    X_test_t = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    model = MLP(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 50
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == yb).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(X_train)

        model.eval()
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == yb).sum().item()

        test_loss = total_loss / len(test_loader)
        test_acc = correct / len(X_test)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

    # ===== SALVA MODELO =====
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    np.save(os.path.join(model_dir, "label_encoder.npy"), encoder.classes_)

    # ===== CURVAS =====
    plt.figure()
    plt.plot(train_acc_history, label="train")
    plt.plot(test_acc_history, label="test")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_history, label="train")
    plt.plot(test_loss_history, label="test")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()

    # ===== PREDICTIONS =====
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device)).argmax(1).cpu().numpy()

    report = classification_report(y_test, preds, target_names=encoder.classes_)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    misclassified = [
        f"True: {encoder.classes_[t]}  Pred: {encoder.classes_[p]}"
        for t, p in zip(y_test, preds) if t != p
    ]

    with open(os.path.join(results_dir, "misclassifications.txt"), "w") as f:
        f.write("\n".join(misclassified))

    print(f"\n✅ Resultados salvos em {results_dir}\n")


# ===== EXECUTA OS DOIS =====
train_mlp(CSV_30S, MODEL_DIR_30S, RESULTS_DIR_30S)
train_mlp(CSV_3S, MODEL_DIR_3S, RESULTS_DIR_3S)

print("\n🎯 TODOS OS MODELOS TORCH TREINADOS COM SUCESSO!\n")
