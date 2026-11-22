# -*- coding: utf-8 -*-
"""
Bridge Type Recognition using EfficientNetB4.

This script trains an image classification model to recognize different types
of bridges (None, beam, arch, cable, suspension) using PyTorch and EfficientNetB4.

Authors: Jianwei Zhang, Chao Yu, Xiya Liu
Affiliation: College of Civil Engineering, Tongji University

Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths (Updated to relative paths for GitHub)
train_data_path = os.path.join("data", "train")
test_data_path = os.path.join("data", "test")

# Class mapping
class_mapping = {
    "None": 1,
    "beam": 2, 
    "arch": 3,
    "cable": 4,
    "suspension": 5
}

# Reverse class mapping (for display)
reverse_class_mapping = {
    1: "None",
    2: "beam",
    3: "arch", 
    4: "cable",
    5: "suspension"
}

# Custom dataset class
class BridgeDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []
        
        # Load training data
        if is_train:
            for class_name, label in class_mapping.items():
                class_path = os.path.join(data_dir, class_name)
                if os.path.exists(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith('.jpg'):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(label - 1)  # Convert to 0-based index
        # Load test data
        else:
            if os.path.exists(data_dir):
                for img_name in os.listdir(data_dir):
                    if img_name.lower().endswith('.png'):
                        self.images.append(os.path.join(data_dir, img_name))
                        self.labels.append(-1)  # Unknown labels for test set
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Read image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            # Return a black image as substitute
            image = Image.new('RGB', (380, 380), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path  # Return image path for result display

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # EfficientNetB4 recommended input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create EfficientNetB4 model
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetB4Classifier, self).__init__()
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        
        # Replace classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, _ in train_pbar:  # Ignore image paths
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels, _ in val_pbar:  # Ignore image paths
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bridge_classifier.pth')
            print(f'  Saved best model, Validation Accuracy: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Enhanced test function
def test_model_with_detailed_results(model, test_loader, test_dataset):
    model.eval()
    predictions = []
    probabilities = []
    image_paths = []
    
    with torch.no_grad():
        for images, _, paths in test_loader:  # Get image paths
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            image_paths.extend(paths)
    
    return predictions, probabilities, image_paths

# Visualize test results
def visualize_test_results(test_dataset, predictions, probabilities, image_paths, num_display=10):
    # Create results directory
    os.makedirs('test_results', exist_ok=True)
    
    # 1. Detailed prediction results table
    print("\n" + "="*80)
    print("Test Set Detailed Prediction Results")
    print("="*80)
    
    results = []
    for i, (img_path, pred, prob) in enumerate(zip(image_paths, predictions, probabilities)):
        predicted_class = reverse_class_mapping[pred + 1]
        confidence = prob[pred] * 100
        results.append({
            'Index': i+1,
            'Image Name': os.path.basename(img_path),
            'Predicted Class': predicted_class,
            'Confidence': f'{confidence:.2f}%',
            'Class Probabilities': {reverse_class_mapping[j+1]: f'{p*100:.2f}%' for j, p in enumerate(prob)}
        })
        
        if i < num_display:  # Only show first few detailed results
            print(f"{i+1:2d}. {os.path.basename(img_path):<20} -> {predicted_class:<12} (Confidence: {confidence:.2f}%)")
    
    # 2. Statistical information
    print("\n" + "="*50)
    print("Test Results Statistics")
    print("="*50)
    
    class_counts = {}
    for pred in predictions:
        class_name = reverse_class_mapping[pred + 1]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"{class_name:<12}: {count:2d} images ({percentage:.1f}%)")
    
    # 3. Confidence distribution
    confidences = [prob[pred] * 100 for pred, prob in zip(predictions, probabilities)]
    avg_confidence = np.mean(confidences)
    max_confidence = np.max(confidences)
    min_confidence = np.min(confidences)
    
    print(f"\nConfidence Statistics:")
    print(f"Average Confidence: {avg_confidence:.2f}%")
    print(f"Maximum Confidence: {max_confidence:.2f}%")
    print(f"Minimum Confidence: {min_confidence:.2f}%")
    
    # 4. Visualize sample prediction results
    display_indices = min(num_display, len(test_dataset))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i in range(display_indices):
        img_path = image_paths[i]
        pred = predictions[i]
        prob = probabilities[i]
        
        # Read and display image
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(f'{os.path.basename(img_path)}\nPredicted: {reverse_class_mapping[pred+1]}\nConfidence: {prob[pred]*100:.1f}%')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Confidence distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(avg_confidence, color='red', linestyle='--', label=f'Average Confidence: {avg_confidence:.2f}%')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Number of Images')
    plt.title('Test Set Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_results/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Save detailed results to file
    with open('test_results/detailed_results.txt', 'w', encoding='utf-8') as f:
        f.write("Bridge Type Recognition Test Results Detailed Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Test Images: {len(predictions)}\n")
        f.write(f"Test Time: {np.datetime64('now')}\n\n")
        
        f.write("Class Distribution:\n")
        for class_name, count in class_counts.items():
            percentage = (count / len(predictions)) * 100
            f.write(f"  {class_name}: {count} images ({percentage:.1f}%)\n")
        
        f.write(f"\nConfidence Statistics:\n")
        f.write(f"  Average Confidence: {avg_confidence:.2f}%\n")
        f.write(f"  Maximum Confidence: {max_confidence:.2f}%\n")
        f.write(f"  Minimum Confidence: {min_confidence:.2f}%\n\n")
        
        f.write("Detailed Prediction Results:\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(results):
            f.write(f"{i+1:2d}. {result['Image Name']:<20} -> {result['Predicted Class']:<12} (Confidence: {result['Confidence']})\n")
    
    print(f"\nDetailed results saved to 'test_results' folder")
    return results

def main():
    # Create datasets
    print("Loading datasets...")
    train_dataset = BridgeDataset(train_data_path, transform=train_transform, is_train=True)

    # Split training and validation sets (9:1)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create test set
    test_dataset = BridgeDataset(test_data_path, transform=test_transform, is_train=False)

    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Initialize model
    model = EfficientNetB4Classifier(num_classes=5)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model (comment this part if you already have a trained model)
    print("Starting model training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1
    )

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('best_bridge_classifier.pth'))

    # Perform detailed prediction on test set
    print("Performing test set prediction...")
    test_predictions, test_probabilities, image_paths = test_model_with_detailed_results(model, test_loader, test_dataset)

    # Display detailed test results
    detailed_results = visualize_test_results(test_dataset, test_predictions, test_probabilities, image_paths)

    print(f"\nTraining and testing completed!")
    print(f"Best model saved as 'best_bridge_classifier.pth'")
    print(f"Detailed test results saved to 'test_results' folder")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()