# -*- coding: utf-8 -*-
"""multilayer_perceptron.py

A Multilayer Perceptron implementation example using Pytorch.
This example does handwritten digit recognition using the MNIST database.
(http://yann.lecun.com/exdb/mnist/)

"""

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

# Network hyperparameters
input_size = 28*28
hidden_layer_size= 512
output_size = 10
num_epochs = 2
batch_size = 100
learning_rate = 1e-3
learning_rate_decay = 0.8
num_workers = 0

# Load MNIST train and test datasets 
def load_dataset():
    train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transforms.ToTensor())
    
    test_size = len(test_data)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
    return train_loader,test_loader,test_size

train_loader,test_loader,test_size = load_dataset()

# function to update learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Create model 
class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):

        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = MLP()
print(model)


# Specify loss function
criterion = nn.CrossEntropyLoss()

# Specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
model.train()
total_step = len(train_loader)

for epoch in range(num_epochs):

    # Loading each input batch
    for i, (images, labels) in enumerate(train_loader):

        # Outputs after forward pass
        outputs = model(images)

         # Calculate loss
        loss = criterion(outputs, labels)

        # Backprop to update model parameters 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Update learning rate for next epoch       
    learning_rate *= learning_rate_decay
    update_lr(optimizer, learning_rate)

# Test
model.eval()
correct=0
for images, labels in test_loader:

    # Compute predicted outputs (forward pass)
    output = model(images)

    # Calculate loss
    loss = criterion(output, labels)

    # Convert output probabilities to predicted class
    _, pred = torch.max(output, 1)

    # compare predictions to true labels
    correct += (pred == labels).sum().item()
    

# Test accuracy
print('Accuracy of the MLP on {} test images: {} %'.format(test_size, 100 * (correct / test_size)))