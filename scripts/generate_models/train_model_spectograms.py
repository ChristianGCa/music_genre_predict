import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Configurações gerais ---
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset ---
class SpectrogramDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# --- Modelo CNN ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# --- Função para carregar os dados ---
def load_spectrogram_dataset(base_dir):
    """
    Lê todos os espectrogramas PNG já gerados.
    Espera estrutura:
    base_dir/
        blues/
        classical/
        ...
    """
    genres = [g for g in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, g))]
    X, y = [], []

    for genre in genres:
        genre_dir = os.path.join(base_dir, genre)
        count = 0
        for f in os.listdir(genre_dir):
            if f.endswith(".png"):
                X.append(os.path.join(genre_dir, f))
                y.append(genres.index(genre))
                count += 1
        print(f"{genre:<12} → {count} imagens")

    print(f"\n{base_dir}: {len(X)} imagens encontradas ({len(genres)} gêneros)\n")
    return X, y, genres


# --- Função de treinamento ---
def train_model(spectro_dir, model_path):
    print(f"\nIniciando treinamento com espectrogramas em: {spectro_dir}")
    X, y, genres = load_spectrogram_dataset(spectro_dir)
    if len(X) == 0:
        print(f"Nenhuma imagem encontrada em {spectro_dir}.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = SpectrogramDataset(X_train, y_train, transform)
    val_ds = SpectrogramDataset(X_val, y_val, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNNModel(num_classes=len(genres)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Modelo inicializado ({len(genres)} classes).")

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"Época{epoch+1}/{EPOCHS}")
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # --- Progresso do batch ---
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                percent = 100 * (batch_idx + 1) / len(train_loader)
                print(f" Batch {batch_idx+1:03}/{len(train_loader)} ({percent:5.1f}%) | Loss: {loss.item():.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # --- Avaliação ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Época {epoch+1}/{EPOCHS} | Loss médio: {avg_loss:.4f} | Treino: {train_acc:.4f} | Validação: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Modelo salvo: {model_path} (Val Acc = {val_acc:.4f})\n")

    print(f"\nMelhor Val Acc = {best_acc:.4f}\n")


# --- Execução principal ---
if __name__ == "__main__":
    # Modelo com espectrogramas de 3 segundos
    train_model("./data/spectrograms_3s", "cnn_3s.pth")

    # Modelo com espectrogramas de 30 segundos
    train_model("./data/spectrograms_30s", "cnn_30s.pth")
