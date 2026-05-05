import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score
import numpy as np
import time
import copy
import os

# 1. 설정 및 하이퍼파라미터
data_dir = "archive/PokemonData"
batch_size = 32
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 및 데이터로더
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = datasets.ImageFolder(data_dir)
num_classes = len(full_dataset.classes)

# 80% train, 20% validation/test
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

def calculate_metrics(labels, preds):
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    return precision, recall

# 3. 모델 학습 함수
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            elif phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                precision, recall = calculate_metrics(all_labels, all_preds)
                print(f'Val Precision: {precision:.4f} Val Recall: {recall:.4f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==================== 실험 4가지 설정 ====================
experiments = {}
criterion = nn.CrossEntropyLoss()

# 1. Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model_A = CustomCNN(num_classes).to(device)
optimizer_A = optim.Adam(model_A.parameters(), lr=0.001)

# 2. ResNet18 (From Scratch)
model_B = models.resnet18(weights=None)
model_B.fc = nn.Linear(model_B.fc.in_features, num_classes)
model_B = model_B.to(device)
optimizer_B = optim.Adam(model_B.parameters(), lr=0.001)

# 3. ResNet18 (Feature Extractor)
model_C = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model_C.parameters():
    param.requires_grad = False
model_C.fc = nn.Linear(model_C.fc.in_features, num_classes)
model_C = model_C.to(device)
optimizer_C = optim.Adam(model_C.fc.parameters(), lr=0.001)

# 4. ResNet18 (Fine-tuning - ALL)
model_D = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_D.fc = nn.Linear(model_D.fc.in_features, num_classes)
model_D = model_D.to(device)
optimizer_D = optim.Adam(model_D.parameters(), lr=0.001)

# 훈련 실행
if __name__ == '__main__':
    print("Training Model A: Custom CNN")
    model_A, hist_a = train_model(model_A, criterion, optimizer_A, num_epochs)

    print("Training Model B: ResNet From Scratch")
    model_B, hist_b = train_model(model_B, criterion, optimizer_B, num_epochs)

    print("Training Model C: ResNet Feature Extractor")
    model_C, hist_c = train_model(model_C, criterion, optimizer_C, num_epochs)

    print("Training Model D: ResNet Fine-tuning")
    model_D, hist_d = train_model(model_D, criterion, optimizer_D, num_epochs)

    # 최고성능 모델 저장
    torch.save(model_D.state_dict(), 'best_model.pth')

    # [과제 제출을 위한 결과 시각화 코드 삽입 부분]
    print("Generating learning curves...")
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Validation Accuracy 비교
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist_a['val_acc'], label='Model A (Custom CNN)')
    plt.plot(epochs_range, hist_b['val_acc'], label='Model B (From Scratch)')
    plt.plot(epochs_range, hist_c['val_acc'], label='Model C (Feature Extractor)')
    plt.plot(epochs_range, hist_d['val_acc'], label='Model D (Fine-tuning)')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 2. Validation Loss 비교
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist_a['val_loss'], label='Model A (Custom CNN)')
    plt.plot(epochs_range, hist_b['val_loss'], label='Model B (From Scratch)')
    plt.plot(epochs_range, hist_c['val_loss'], label='Model C (Feature Extractor)')
    plt.plot(epochs_range, hist_d['val_loss'], label='Model D (Fine-tuning)')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Learning curve saved as 'learning_curve.png'")
