import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from audio_utils import load_spectrogram_as_tensor

import torch.nn.functional as F
import random

def spec_augment(tensor, time_mask_param=30, freq_mask_param=15, num_time_masks=1, num_freq_masks=1):
    # tensor shape: (C=1, H=freq, W=time)
    if not isinstance(tensor, torch.Tensor):
        return tensor
    _, H, W = tensor.shape
    t = tensor.clone()
    for _ in range(num_time_masks):
        t_len = random.randint(0, min(time_mask_param, W))
        if t_len == 0:
            continue
        t_start = random.randint(0, max(0, W - t_len))
        t[:, :, t_start:t_start + t_len] = 0
    for _ in range(num_freq_masks):
        f_len = random.randint(0, min(freq_mask_param, H))
        if f_len == 0:
            continue
        f_start = random.randint(0, max(0, H - f_len))
        t[:, f_start:f_start + f_len, :] = 0
    return t

BATCH_SIZE = 32
EPOCHS = 40
IMG_SIZE = (256, 256)
LEARNING_RATE = 1e-4

SPECTROGRAMS = {
    'cnn_3s': 'data/spectograms_3s',
    'cnn_30s': 'data/spectograms_30s',
}

os.makedirs(SPECTROGRAMS['cnn_3s'], exist_ok=True)
os.makedirs(SPECTROGRAMS['cnn_30s'], exist_ok=True)

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
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        else:
            # fallback to helper loader
            tensor = load_spectrogram_as_tensor(img_path, self.img_size)
            return tensor.squeeze(0), self.genres.index(genre)
        label = self.genres.index(genre)
        return img, label

    def get_group(self, idx):
        """Return a group id for the sample (used to avoid leakage between segments of same track).
        Group is derived from filename before the final underscore, e.g. pop.00007_3.png -> pop.00007
        If no underscore exists, falls back to filename without extension.
        """
        path, _ = self.samples[idx]
        fname = os.path.basename(path)
        name = os.path.splitext(fname)[0]
        if '_' in name:
            return name.rsplit('_', 1)[0]
        return name

class ImprovedCNN(nn.Module):
    """A small, deeper CNN with BatchNorm and Dropout and adaptive pooling."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # height and width / 2

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
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: spec_augment(x, time_mask_param=40, freq_mask_param=20, num_time_masks=2, num_freq_masks=2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    base_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=None)
    samples = base_dataset.samples

    train_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=train_transform, samples=samples)
    val_dataset = SpectrogramDataset(SPECTROGRAMS[spectrogram_type], GENRES, IMG_SIZE, transform=val_transform, samples=samples)

    num_samples = len(samples)
    if spectrogram_type == 'cnn_3s':
        # Group indices by group id
        group_to_indices = {}
        for i in range(num_samples):
            g = base_dataset.get_group(i)
            group_to_indices.setdefault(g, []).append(i)
        groups = list(group_to_indices.keys())
        np.random.shuffle(groups)
        train_idx_list = []
        cnt = 0
        target_train = int(0.8 * num_samples)
        for g in groups:
            if cnt >= target_train:
                break
            idxs = group_to_indices[g]
            train_idx_list.extend(idxs)
            cnt += len(idxs)
        train_idx = np.array(sorted(train_idx_list), dtype=int)
        # rest -> val
        all_idx = np.arange(num_samples)
        val_idx = np.setdiff1d(all_idx, train_idx)
    else:
        # stratified split by class label to keep per-class proportions stable (use simple per-class split)
        idxs_by_class = {c: [] for c in range(len(GENRES))}
        for i, (_, genre) in enumerate(samples):
            idxs_by_class[GENRES.index(genre)].append(i)
        train_idx_list, val_idx_list = [], []
        for c, idxs in idxs_by_class.items():
            idxs = np.array(idxs)
            np.random.shuffle(idxs)
            split = int(0.8 * len(idxs))
            train_idx_list.extend(idxs[:split].tolist())
            val_idx_list.extend(idxs[split:].tolist())
        train_idx = np.array(sorted(train_idx_list), dtype=int)
        val_idx = np.array(sorted(val_idx_list), dtype=int)
    # When creating train/val subsets, we need the train transform applied only on train set
    train_set = torch.utils.data.Subset(train_dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(val_dataset, val_idx.tolist())

    train_labels = []
    for idx in train_idx.tolist():
        _, genre = samples[idx]
        train_labels.append(GENRES.index(genre))
    class_sample_count = np.array([train_labels.count(i) for i in range(len(GENRES))])
    # avoid divide by zero
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weights_per_class = 1.0 / class_sample_count
    sample_weights = np.array([weights_per_class[label] for label in train_labels], dtype=float)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights_tensor.tolist(), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCNN(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc = 0.0
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
        
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Novo melhor modelo salvo (val_acc={best_val_acc:.4f}) em {save_path}')
    if best_val_acc == 0.0:
        torch.save(model.state_dict(), save_path)
        print(f'Modelo final salvo em {save_path}')

    results_dir = os.path.join("results", spectrogram_type)
    os.makedirs(results_dir, exist_ok=True)

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
