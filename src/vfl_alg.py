import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from torch.distributions.binomial import Binomial

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.densenet = torchvision.models.densenet169(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        return x

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.densenet = torchvision.models.densenet169(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        # Define new classification layers
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.BatchNorm1d(53760), # Output features from densenet + resnet + vgg
            nn.Linear(53760, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        return x

# Arguments and parameters
num_clients = 4
lr = 0.0001
lr_decay = 0.9
batch_size = 32
num_epochs = 5
quant_bin = 8 # quantization parameter
theta = 0.15 # DP noise parameter

# Make models for each client
models = []
optimizers = []
schedulers = []

for i in range(num_clients+1):
    if i == num_clients:
        model = ClientModel()
    else:
        model = ServerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    
    models.append(model)
    optimizers.append(optimizer)
    schedulers.append(scheduler)

criterion = nn.BCEWithLogitsLoss() # Binary cross-entropy

# Discrete differential privacy noise
def quantize(x, theta, m):
    p = torch.add(0.5, torch.mul(theta, x))

    binom = Binomial(m, p)
    noise = binom.sample()
    
    y = x.clone()
    y.data = noise
    
    return y

def dequantize(q, theta, m, n):
    det = torch.sub(q, m * n / 2)
    sum = torch.div(det, theta * m)
    return sum

# Load datasets
# each image has size 128x128
transform = transforms.ToTensor()

train_loaders = []
val_loaders = []
test_loaders =[]

for i in range(num_clients):
    train_data = torchvision.datasets.ImageFolder(root=f'C:/Users/huyfr/Documents/AI&Blockchain/project/SplitCovid19/client{i}', transform=transform)
    train_data = torch.utils.data.Subset(train_data, range(10))
    val_data = torch.utils.data.Subset(train_data, range(10, 20))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,  num_workers=4)
    train_loaders.append(train_loader)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,  num_workers=4)
    val_loaders.append(val_loader)
    test_data = torchvision.datasets.ImageFolder(root=f'C:/Users/huyfr/Documents/AI&Blockchain/project/SplitCovid19/client{i}', transform=transform)
    test_data = torch.utils.data.Subset(test_data, range(10))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loaders.append(test_loader)

# train without blockchain
def train_without_blockchain():
    
    targets = None
    embeddings_grad = [None] * num_clients
    embeddings_nograd = [None] * num_clients
    
    completed = False
    train_iterators = []
    for i in range(num_clients):
        train_iterators.append(iter(train_loaders[i]))
    
    while not completed:
        # At party side
        for i in range(num_clients):
            # get current batch
            item = next(train_iterators[i], -1)
            
            if item == -1:
                completed = True
                break
            
            inputs, targets = item
        
            # generate embedding
            embeddings_grad[i] = models[i](inputs)
            with torch.no_grad():
                embeddings_nograd[i] = models[i](inputs)

            # add differential privacy noise
            embeddings_nograd[i] = quantize(embeddings_nograd[i], theta, quant_bin)
        
        if completed:
            break

        # At server side
        with torch.no_grad():
            sum_nograd = torch.sum(torch.stack(embeddings_nograd),axis=0)

        # dequantize the discrete sum into continuous sum
        sum_nograd = dequantize(sum_nograd, theta, quant_bin, num_clients)
        sum_grad = torch.sum(torch.stack(embeddings_grad),axis=0)
        sum_grad.data = sum_nograd

        # compute outputs
        outputs = models[num_clients](sum_grad)
        loss = criterion(outputs, targets)

        # parties and server compute gradient and do SGD step
        for i in range(num_clients + 1):
            optimizers[i].zero_grad()
        loss.backward()
        for i in range(num_clients + 1):
            optimizer[i].step()

        # parties and server calculate new learning rate
        for i in range(num_clients + 1):
            schedulers[i].step()

# evaluation function for validation and testing
def evaluate(mode):
    # validation or testing
    data_iterators = []
    for i in range(num_clients):
        if mode == 'validation':
            data_iterators.append(iter(val_loaders[i]))
        else:
            data_iterators.append(iter(test_loaders[i]))
    
    # initialize variables
    embeddings = [None] * num_clients
    targets = None
    completed = False
    total = 0
    correct = 0
    total_loss = 0
    n = 0
    
    while not completed:
        # At party side
        for i in range(num_clients):
            # get current batch
            item = next(data_iterators[i], -1)
            
            if item == -1:
                completed = True
                break
            
            inputs, targets = item

            # generate embedding
            embeddings[i] = models[i](inputs)
        
        if completed:
            break

        # At server side
        embedding_sum = torch.sum(torch.stack(embeddings),axis=0)

        # compute outputs
        outputs = models[num_clients](embedding_sum)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        # compute accuracy
        correct += (predicted == targets).sum()
        total += targets.size(0)

        # compute loss
        total_loss += loss
        n += 1

    accuracy = 100 * correct / total
    loss = total_loss / n

    return (accuracy, loss)

# initial loss
test_accuracy, test_loss = evaluate(mode = 'test')
print('Initial test loss: {:.2f} \t Initial test accuracy: {:.2f}'.format(test_loss, test_accuracy))

# main training loop
for epoch in range(num_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, num_epochs))

    train_without_blockchain()
    val_accuracy, val_loss = evaluate(mode = 'validation')
    test_accuracy, test_loss = evaluate(mode = 'test')
    
    print('Val Loss: {:.2f} \t Val Accuracy: {:.2f} \t Test Loss: {:.2f} \t Test Accuracy: {:.2f}'.format(val_loss, val_accuracy, test_loss, test_accuracy))