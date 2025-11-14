import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from audio_utils import load_spectrogram_as_tensor

# Parâmetros
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (256, 256)
LEARNING_RATE = 1e-4

# Pastas dos espectrogramas
SPECTROGRAMS = {
    'cnn_3s': 'data/spectograms_3s',
    'cnn_30s': 'data/spectograms_30s',
}

GENRES = sorted(os.listdir(SPECTROGRAMS['cnn_3s']))

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, genres, img_size=(512, 512), transform=None, samples=None):
        self.samples = [] if samples is None else samples
        self.genres = genres
        self.img_size = img_size
        self.transform = transform
        if samples is None:
            for genre in genres:
                genre_dir = os.path.join(root_dir, genre)
                for fname in os.listdir(genre_dir):
                    if fname.lower().endswith('.png'):
                        self.samples.append((os.path.join(genre_dir, fname), genre))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, genre = self.samples[idx]
        # Load PIL image here and apply transforms (gives more flexibility for augmentation)
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        else:
            # fallback to helper loader
            tensor = load_spectrogram_as_tensor(img_path, self.img_size)
            return tensor.squeeze(0), self.genres.index(genre)
        label = self.genres.index(genre)
        return img, label

class ImprovedCNN(nn.Module):
    """A small, deeper CNN with BatchNorm and Dropout and adaptive pooling."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(spectrogram_type, save_path):
    print(f'Treinando modelo para espectrogramas de {spectrogram_type}...')
    # Transforms: keep images grayscale but apply sensible augmentations on time/frequency axes
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomApply([transforms.RandomAffine(degrees=8, translate=(0.05, 0.05))], p=0.6),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Build a base dataset to collect sample list (no transform)
    base_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=None)
    samples = base_dataset.samples
    # Create train/val datasets that share the same samples ordering but have different transforms
    train_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=train_transform, samples=samples)
    val_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=val_transform, samples=samples)
    # Split train/val (80/20)
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    # When creating train/val subsets, we need the train transform applied only on train set
    train_set = torch.utils.data.Subset(train_dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(val_dataset, val_idx.tolist())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc = 0.0
    # Scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_set)
        train_losses.append(epoch_loss)
    # Validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / len(val_set)
        val_losses.append(val_loss)
        val_acc = correct / total if total > 0 else 0
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        # Step scheduler
        scheduler.step(val_acc)
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Novo melhor modelo salvo (val_acc={best_val_acc:.4f}) em {save_path}')
    # If model wasn't saved by scheduler loop (no improvement), save final state
    if best_val_acc == 0.0:
        torch.save(model.state_dict(), save_path)
        print(f'Modelo final salvo em {save_path}')
    # Avaliação final e métricas
    results_dir = os.path.join("results", spectrogram_type)
    os.makedirs(results_dir, exist_ok=True)
    # Curvas de loss e acurácia
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()
    plt.figure()
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
    plt.close()
    # Confusion matrix e classification report
    y_true, y_pred, misclassified = [], [], []
    model.eval()
    val_idx_list = val_idx.tolist()
    sample_counter = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            y_true.extend(labels_np)
            y_pred.extend(preds_np)
            for i in range(len(labels_np)):
                if preds_np[i] != labels_np[i]:
                    idx_in_dataset = val_idx_list[sample_counter + i]
                    misclassified.append({
                        'file': val_dataset.samples[idx_in_dataset][0],
                        'true': GENRES[int(labels_np[i])],
                        'pred': GENRES[int(preds_np[i])]
                    })
            sample_counter += len(labels_np)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    # Classification report
    report = classification_report(y_true, y_pred, target_names=GENRES)
    if not isinstance(report, str):
        report = str(report)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    # Misclassifications
    if misclassified:
        df = pd.DataFrame(misclassified)
        df.to_csv(os.path.join(results_dir, 'misclassifications.csv'), index=False)
    else:
        with open(os.path.join(results_dir, 'misclassifications.csv'), 'w') as f:
            f.write('file,true,pred\n')

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train_model('cnn_30s', 'models/cnn_30s.pth')
    train_model('cnn_3s', 'models/cnn_3s.pth')
