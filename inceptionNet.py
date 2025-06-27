import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import shutil
import csv
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Global hyperparameters
INITIAL_LR = 0.001  # Initial learning rate for Adam
MAX_LR = 0.005      # Maximum learning rate for CyclicLR
BATCH_SIZE = 32     # Batch size (adjustable: 16 or 64 based on GPU memory)
WEIGHT_DECAY = 0.0001  # L2 regularization strength
DROPOUT_RATE = 0.2  # Dropout rate for classifier layers
FREEZE_PERCENTAGE = 0.8  # Percentage of feature extraction layers to freeze (0 to 1)
NUM_EPOCHS = 200    # Maximum number of epochs
LABEL_SMOOTHING = 0.1  # Label smoothing for CrossEntropyLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "Potato Leaf Disease Dataset in Uncontrolled Environment"  # Update this to your local path if needed
try:
    dataset = datasets.ImageFolder(data_dir, transform=val_test_transforms)
    print(f"Classes: {dataset.classes}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Check if data_dir '{data_dir}' exists and contains class folders.")
    raise

# Split dataset into train (80%), validation (10%), and test (10%)
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size
print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

# Apply train transforms to train dataset
train_dataset.dataset.transform = train_transforms

# Create data loaders
try:
    # Set num_workers=0 for Windows compatibility; adjust based on your system
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
except Exception as e:
    print(f"Error creating DataLoader: {e}")
    raise

# Load GoogLeNet model
model = models.googlenet(weights='IMAGENET1K_V1')

# Freeze specified percentage of feature extraction layers
total_params = sum(1 for _ in model.parameters())
freeze_count = int(FREEZE_PERCENTAGE * total_params)
param_count = 0
for param in model.parameters():
    if param_count < freeze_count:
        param.requires_grad = False
    else:
        param.requires_grad = True
    param_count += 1
print(f"Froze {freeze_count}/{total_params} parameters ({FREEZE_PERCENTAGE*100:.1f}%)")

# Modify classifier with additional dropout for regularization
num_classes = len(dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(256, num_classes)
)

model = model.to(device)
print("Model loaded and moved to device")

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=INITIAL_LR/10, max_lr=MAX_LR, step_size_up=len(train_loader) * 2, mode='triangular'
)
scaler = GradScaler()  # For mixed precision training

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.best_loss = val_loss

# Initialize CSV file for training history
history_file = 'training_history.csv'
output_dir = "Potato_Model_Output"  # Local directory to save outputs
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
history_file_path = os.path.join(output_dir, history_file)
checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')

try:
    with open(history_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Precision', 'Recall', 'F1'])
    print(f"Training history will be saved to {history_file_path}")
except Exception as e:
    print(f"Error initializing CSV file: {e}")
    raise

# Training loop with history logging, train accuracy, and tqdm
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    early_stopping = EarlyStopping(patience=10)
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        train_preds, train_true = [], []
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(train_true, train_preds)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        preds, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Log metrics to CSV
        try:
            with open(history_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_loss, train_accuracy, val_loss, val_accuracy, precision, recall, f1])
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            raise

        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        early_stopping(val_loss, model, path=checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model
    try:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    return model

# Test function
def test_model(model, test_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# Train and test
print("Starting training...")
try:
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    print("Training completed. Evaluating on test set...")
    test_model(model, test_loader)
except Exception as e:
    print(f"Error during training or testing: {e}")
    raise

# Save history and checkpoint to local output directory
print("Saving history and checkpoint to local directory...")
try:
    # Files are already saved in output_dir during training
    print(f"Files saved successfully to {output_dir}")
except Exception as e:
    print(f"Error saving files: {e}")
    raise