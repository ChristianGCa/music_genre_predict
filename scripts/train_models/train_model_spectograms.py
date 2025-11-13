import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configurações gerais ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECTROGRAMS_30S_DIR = "./data/spectograms_30s"
SPECTROGRAMS_3S_DIR = "./data/spectograms_3s"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
EPOCHS = 20
BATCH_SIZE = 4
LR = 0.0001
IMAGE_SIZE = 512

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512 -> 256
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_data(base_dir):
    """Carrega caminhos de imagens e labels dos espectrogramas"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {base_dir}\n")
    
    image_paths = []
    labels = []
    genre_to_idx = {}
    
    genres = sorted([d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))])
    
    if not genres:
        raise ValueError(f"Nenhum gênero encontrado em: {base_dir}")
    
    print(f"\nGêneros encontrados: {genres}")
    
    for idx, genre in enumerate(genres):
        genre_to_idx[genre] = idx
        genre_path = os.path.join(base_dir, genre)
        
        # Listar imagens do gênero
        genre_images = [f for f in os.listdir(genre_path) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"{genre}: {len(genre_images)} imagens")
            
        for img_file in genre_images:
            image_paths.append(os.path.join(genre_path, img_file))
            labels.append(idx)
    
    if not image_paths:
        raise ValueError(f"Nenhuma imagem encontrada em {base_dir}!")
    
    return image_paths, labels, genre_to_idx

def train_model(spec_dir, model_name):
    print(f"Treinando modelo: {model_name}")
    
    result_path = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(result_path, exist_ok=True)
    
    image_paths, labels, genre_to_idx = load_data(spec_dir)
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}
    num_classes = len(genre_to_idx)

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Transformações (data augmentation para treino)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Datasets e DataLoaders
    train_dataset = SpectrogramDataset(X_train, y_train, transform=train_transform)
    test_dataset = SpectrogramDataset(X_test, y_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Modelo, loss e otimizador
    model = SpectrogramCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Histórico
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        
        # Treino
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels_batch) in enumerate(train_loader):
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Salvar histórico
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Treino {model_name}| Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
        
        # Scheduler
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"⚠ Learning rate ajustado: {prev_lr:.6f} → {new_lr:.6f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Modelo salvo. Val Acc: {best_val_acc:.4f}")
        
        print()
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = (y_true == y_pred).mean()
    
    print(f"Acurácia final no teste: {test_acc:.4f}")
    print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Treino - Acurácia', linewidth=2)
    plt.plot(history['val_acc'], label='Validação - Acurácia', linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title(f"Acurácia durante o treino - {model_name}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "accuracy_curve.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Treino - Loss', color='red', linewidth=2)
    plt.plot(history['val_loss'], label='Validação - Loss', color='orange', linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f"Perda durante o treino - {model_name}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "loss_curve.png"), dpi=150)
    plt.close()
    
    cm = confusion_matrix(y_true, y_pred)
    genre_names = [idx_to_genre[i] for i in range(num_classes)]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title(f"Matriz de confusão - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=genre_names)
    report_path = os.path.join(result_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Acurácia Final no Teste: {test_acc:.4f}\n")
        f.write(report)
    print(f"Relatório salvo em: {report_path}")
    
    errors_data = []
    for i in range(len(y_test)):
        if y_true[i] != y_pred[i]:
            errors_data.append({
                'image_path': X_test[i],
                'true_label': idx_to_genre[y_true[i]],
                'predicted_label': idx_to_genre[y_pred[i]]
            })
    
    if errors_data:
        import pandas as pd
        errors_df = pd.DataFrame(errors_data)
        errors_path = os.path.join(result_path, "misclassifications.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Total de erros: {len(errors_df)} ({len(errors_df)/len(y_test)*100:.2f}%)")
        print(f"Erros salvos em: {errors_path}")
    else:
        print("Nenhum erro de classificação!")

    print(f"Modelo salvo em: {best_model_path}")
    return genre_to_idx

if __name__ == "__main__":
    if os.path.exists(SPECTROGRAMS_30S_DIR):
        try:
            genre_map_30s = train_model(SPECTROGRAMS_30S_DIR, "cnn_30s")
        except Exception as e:
            print(f"\nErro ao treinar modelo 30s: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nPulando treinamento 30s - diretório não encontrado")
    
    if os.path.exists(SPECTROGRAMS_3S_DIR):
        try:
            genre_map_3s = train_model(SPECTROGRAMS_3S_DIR, "cnn_3s")
        except Exception as e:
            print(f"\nErro ao treinar modelo 3s: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nulando treinamento 3s - diretório não encontrado")
    
    print(f"\nModelos salvos em: {MODELS_DIR}")
    print(f"Resultados salvos em: {RESULTS_DIR}")