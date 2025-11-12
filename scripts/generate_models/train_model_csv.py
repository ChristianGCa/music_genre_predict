"""
train_mlp_models.py
===================

Treina dois modelos MLP:
 - Um com dados de 30 segundos
 - Outro com dados de 3 segundos

Gera gráficos e relatórios automáticos:
 - Curva de acurácia e perda
 - Matriz de confusão
 - Classificação detalhada
 - Lista de erros

Autor: [Seu Nome]
Data: [Data Atual]
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_30S = "./data/csv/features_30s.csv"
CSV_3S = "./data/csv/features_3s.csv"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train_model(csv_path, model_name):
    print(f"\nTreinando modelo: {model_name}")
    df = pd.read_csv(csv_path)

    result_path = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(result_path, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in ['filename', 'genre', 'segment']]
    X = df[feature_cols].values.astype('float32')
    y = df['genre'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)


    model = MLPModel(input_size=X.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)


    history_loss, history_acc = [], []


    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")


    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            _, preds = torch.max(outputs, 1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = (y_true == y_pred).mean()
    print(f"Acurácia no teste: {test_acc:.4f}")


    plt.figure(figsize=(8, 5))
    plt.plot(history_acc, label='Treino - Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.title(f"Acurácia durante o treino - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_loss, label='Treino - Loss', color='red')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title(f"Perda durante o treino - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "loss_curve.png"))
    plt.close()


    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Matriz de confusão - {model_name}")
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()


    report = classification_report(y_true, y_pred, target_names=le.classes_)
    with open(os.path.join(result_path, "classification_report.txt"), "w") as f:
        f.write(report)
    print("Relatório salvo.")


    errors_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "predicted_label": le.inverse_transform(y_pred)
    })
    errors_df = errors_df[errors_df["true_label"] != errors_df["predicted_label"]]
    print(f"Total de erros: {len(errors_df)}")


    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

    return le


if __name__ == "__main__":
    le_30s = train_model(CSV_30S, "mlp_30s")
    le_3s = train_model(CSV_3S, "mlp_3s")
