import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
LR = 0.001
BATCH_SIZE = 16
IMG_SIZE = (128, 128)

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_save(DATA_DIR, RESULTS_DIR, model_name="cnn_model"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ----------------------------------------------------
    # NOVO: Função para salvar a amostra da imagem transformada
    def save_sample_image(dataset, result_dir):
        # Apenas pega a primeira imagem do dataset
        sample_img_tensor, _ = dataset[0] 
        
        # O tensor de entrada é (1, H, W).
        # Para salvar a imagem, precisamos remover a normalização.
        # Desfaz a normalização (X * std + mean) para 1 canal (Grayscale)
        mean = 0.5
        std = 0.5
        denormalized_tensor = sample_img_tensor * std + mean
        
        # Clipa os valores para garantir que fiquem no intervalo [0, 1]
        denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
        
        # Converte para PIL Image e salva
        # O tensor é (C, H, W). Precisamos de (H, W) para PIL Image Grayscale (L)
        img_to_save = transforms.ToPILImage()(denormalized_tensor.cpu())
        
        sample_path = os.path.join(result_dir, f"{model_name}_sample_processed.png")
        img_to_save.save(sample_path)
        print(f"Amostra de espectrograma processado salva em: {sample_path}")
        
    # ----------------------------------------------------

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # NOVO: Chamada da função para salvar a amostra
    if len(dataset) > 0:
        save_sample_image(dataset, RESULTS_DIR)
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    num_classes = len(dataset.classes)
    class_names = dataset.classes
    print(f"Gêneros detectados em {DATA_DIR}: {class_names}")

    model = CNNModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    acc_history = []


    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        print(f"{model_name} | Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")


    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")


    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, NUM_EPOCHS+1), loss_history, marker='o')
    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(range(1, NUM_EPOCHS+1), acc_history, marker='o', color='orange')
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_metrics.png"))
    plt.close()


    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão')
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()


    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(RESULTS_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"Resultados salvos em: {RESULTS_DIR}\n")


train_and_save(DATA_DIR="./data/spectrograms_30s",
               RESULTS_DIR="./results/cnn_30s",
               model_name="cnn_30s")

train_and_save(DATA_DIR="./data/spectrograms_3s",
               RESULTS_DIR="./results/cnn_3s",
               model_name="cnn_3s")
