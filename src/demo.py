import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.binomial import Binomial

from models import ClientModel2Layers, ServerModel
from Blockchain_and_VFL_Integration import BlockchainVFLIntegrator

import os
import argparse
import time

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

# train function
def train_with_blockchain():
    
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

            # send embeddings (embeddings_nograd[i]) to smart contract
            client_parameters = [list(map(int, row)) for row in embeddings_nograd[i].tolist()]
            blockchain_vfl_integrator.update_client_weights(blockchain_vfl_integrator.client_accounts[i], client_parameters)
        
        if completed:
            break

        # At server side
        # retrieve the embedding sum from smart contract
        blockchain_vfl_integrator.aggregate_weights()
        with torch.no_grad():
            sum_nograd = torch.tensor(blockchain_vfl_integrator.get_aggregated_weights())

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
            optimizers[i].step()
    
        # parties and server calculate new learning rate
        for i in range(num_clients + 1):
            schedulers[i].step()
            
    del train_iterators


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
    
    del data_iterators
    accuracy = 100 * correct / total
    loss = total_loss / n

    return (accuracy, loss)


if __name__ == "__main__":
    CONTRACT_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "Aggregator.sol"
    # Create the blockchain integrator
    blockchain_vfl_integrator = BlockchainVFLIntegrator(4, CONTRACT_SOURCE)
    
    # Arguments and parameters
    parser = argparse.ArgumentParser(description='VFL')
    parser.add_argument('--theta', default=0.1, type=float, metavar='T', help='Noise value (in range [0, 0.25]). Default is 0.1')
    parser.add_argument('--datasize', default=1.0, type=float, metavar='T', help='Datasize size (0.25, 0.5, or 1.0). Default is 1.0')
    args = parser.parse_args()
    
    num_clients = 4
    lr = 0.0001
    lr_decay = 0.9
    batch_size = 10
    num_epochs = 5
    quant_bin = 8 # quantization parameter
    theta = args.theta # DP noise parameter

    
    # Make models for each client
    models = []
    optimizers = []
    schedulers = []

    for i in range(num_clients+1):
        if i == num_clients:
            model = ServerModel()
        else:
            model = ClientModel2Layers()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    criterion = nn.CrossEntropyLoss()

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),])

    train_loaders = []
    val_loaders = []
    test_loaders = []

    train_val_permute = None
    train_permute = None
    val_permute = None
    test_permute = None
    
    if args.datasize == 1.0:
        # train (800); val (150); test (200)
        train_val_permute = np.random.permutation(np.arange(989))
        train_permute = train_val_permute[:800]
        val_permute = train_val_permute[800:950]
        test_permute = np.random.permutation(np.arange(563))[:200]
    elif args.datasize == 0.5:
        # train (400); val (100); test (100)
        train_val_permute = np.random.permutation(np.arange(989))
        train_permute = train_val_permute[:400]
        val_permute = train_val_permute[400:500]
        test_permute = np.random.permutation(np.arange(563))[:100]
    elif args.datasize == 0.25:
        # train (200); val (50); test (50)
        train_val_permute = np.random.permutation(np.arange(989))
        train_permute = train_val_permute[:200]
        val_permute = train_val_permute[200:250]
        test_permute = np.random.permutation(np.arange(563))[:50]

    for i in range(num_clients):
        train_dataset = torchvision.datasets.ImageFolder(root=f'SplitCovid19/client{i}/train', transform=transform)
        train_data = torch.utils.data.Subset(train_dataset, indices=train_permute)
        val_data = torch.utils.data.Subset(train_dataset, indices=val_permute)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,  num_workers=1)
        train_loaders.append(train_loader)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,  num_workers=1)
        val_loaders.append(val_loader)
        test_dataset = torchvision.datasets.ImageFolder(root=f'SplitCovid19/client{i}/test', transform=transform)
        test_data = torch.utils.data.Subset(test_dataset, indices=test_permute)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loaders.append(test_loader)    

    # initial loss
    test_accuracy, test_loss = evaluate(mode = 'test')
    print('Initial test loss: {:.2f} \t Initial test accuracy: {:.2f}'.format(test_loss, test_accuracy))

    # main training loop
    for epoch in range(num_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, num_epochs))

        start = time.time()
        train_with_blockchain()
        print('Time taken: %.2f sec.' % (time.time() - start))

        val_accuracy, val_loss = evaluate(mode = 'validation')
        test_accuracy, test_loss = evaluate(mode = 'test')

        print('Val Loss: {:.2f} \t Val Accuracy: {:.2f} \t Test Loss: {:.2f} \t Test Accuracy: {:.2f}'.format(val_loss, val_accuracy, test_loss, test_accuracy))
