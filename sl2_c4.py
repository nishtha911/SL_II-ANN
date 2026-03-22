#MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=False,
                                            transform=transforms.ToTensor())
# Data loaders
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size, 
                            shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size, 
                            shuffle=False)
# Neural Network Model (1 hidden layer)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out 
model = NeuralNet(28*28, 500, 10).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Test the model
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
# Plot some test images with their predicted labels
def plot_images(images, labels, preds):
    fig = plt.figure(figsize=(10, 8))
    for i in range(12):
        ax = fig.add_subplot(3, 4, i+1)
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {labels[i]}\nPred: {preds[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
test_images = []
test_labels = []
test_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_images.extend(images.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())
        if len(test_images) >= 12:
            break
plot_images(test_images[:12], test_labels[:12], test_preds[:12])


