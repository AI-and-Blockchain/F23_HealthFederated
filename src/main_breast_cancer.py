import argparse
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.binomial import Binomial

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

import os
from Blockchain_and_VFL_Integration import BlockchainVFLIntegrator

# Arguments and parameters
parser = argparse.ArgumentParser(description='VFL without Blockchain')
parser.add_argument('--num_clients', type=int, help='Number of clients to split data between vertically (5 or 10)', default=5)
parser.add_argument('--quant_bin', type=int, help='Number of quantization buckets', default=0)
parser.add_argument('--theta', type=float, metavar='T', help='Noise value (in range [0, 0.25])', default=0.00)
parser.add_argument('--withblockchain', type=bool, help='With or without blockchain. Default is False', default=False)
args = parser.parse_args()

num_clients = args.num_clients
quant_bin = args.quant_bin
theta = args.theta

blockchain_vfl_integrator = None
if args.withblockchain:
    CONTRACT_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "Aggregator.sol"
    # Create the blockchain integrator
    blockchain_vfl_integrator = BlockchainVFLIntegrator(num_clients, CONTRACT_SOURCE)

# parameters
batch_size = 10
num_epochs = 10
lr = 0.001
lr_decay = 0.9
print_freq = 50
num_features = 30
coords_per = int(num_features/num_clients)
embedding_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pkl_auprc = []
pkl_auroc = []
pkl_loss = []

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_size = 450
test_size = 110

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train[:train_size,:], dtype=torch.float32)
y_train = torch.tensor(y_train[:train_size], dtype=torch.float32)
X_test = torch.tensor(X_test[:test_size,:], dtype=torch.float32)
y_test = torch.tensor(y_test[:test_size], dtype=torch.float32)

# Split client data
X_trains, X_tests = [], []
for i in range(num_clients):
    X_trains.append(X_train[:,coords_per*i:coords_per*(i+1)])
    X_tests.append(X_test[:,coords_per*i:coords_per*(i+1)])

# Make models for each client
class Bottom(nn.Module):
    def __init__(self, input_size, output_size):
        super(Bottom, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = torch.tanh(out)
        return out

class Top(nn.Module):
    def __init__(self, input_size):
        super(Top, self).__init__()
        self.linear = nn.Linear(input_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

top_models = []
bot_models = []
top_optimizers = []
bot_optimizers = []
top_schedulers = []
bot_schedulers = []

for i in range(num_clients):
    top_model = None
    top_model = Top(input_size=embedding_size)
    top_optimizer = torch.optim.SGD(top_model.parameters(), lr=lr)
    top_scheduler = ExponentialLR(top_optimizer, gamma=lr_decay)
    top_models.append(top_model)
    top_optimizers.append(top_optimizer)
    top_schedulers.append(top_scheduler)
    top_model.to(device)
    cudnn.benchmark = True

    bot_model = None
    bot_model = Bottom(input_size=coords_per, output_size=embedding_size)
    bot_optimizer = torch.optim.SGD(bot_model.parameters(), lr=lr)
    bot_scheduler = ExponentialLR(bot_optimizer, gamma=lr_decay)
    bot_models.append(bot_model)
    bot_optimizers.append(bot_optimizer)
    bot_schedulers.append(bot_scheduler)
    bot_model.to(device)
    cudnn.benchmark = True

criterion = nn.BCELoss()

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

def train():
    for step in range(0, train_size, batch_size):

        labels = y_train[step:step+batch_size]
        try:
            labels = labels.cuda(device)
        except RuntimeError:
            labels = labels.cpu()
        labels = Variable(labels)

        # Exchange embeddings
        H_orig = [None] * num_clients
        with torch.no_grad():
            H_nograd = [None] * num_clients
        for i in range(num_clients):
            inputs = X_trains[i][step:step+batch_size]
            try:
                inputs = inputs.cuda(device)
            except RuntimeError:
                inputs = inputs.cpu()
            inputs = Variable(inputs)
            H_orig[i] = bot_models[i](inputs)
            with torch.no_grad():
                H_nograd[i] = bot_models[i](inputs)
            
            # Compress embedding / Quantization
            if quant_bin > 0:
                H_nograd[i] = quantize(H_nograd[i], theta, quant_bin)

                if args.withblockchain:
                    client_parameters = [list(map(int, row)) for row in H_nograd[i].tolist()]
                    blockchain_vfl_integrator.update_client_weights(blockchain_vfl_integrator.client_accounts[i], client_parameters)
        
        # embedding summation
        if args.withblockchain:
            # retrieve the embedding sum from smart contract
            blockchain_vfl_integrator.aggregate_weights()
            with torch.no_grad():
                sum_nograd = torch.tensor(blockchain_vfl_integrator.get_aggregated_weights())
        else:
            with torch.no_grad():
                sum_nograd = torch.sum(torch.stack(H_nograd),axis=0)

        if quant_bin > 0:
            sum_nograd = dequantize(sum_nograd, theta, quant_bin, num_clients)
        
        sum_embedding = torch.sum(torch.stack(H_orig),axis=0)
        if quant_bin > 0:
            sum_embedding.data = sum_nograd

        # compute output
        outputs = []
        for i in range(num_clients):
            outputs.append(top_models[i](sum_embedding))
        loss = criterion(outputs[0][:, 1], labels)

        # compute gradient and do SGD step
        for i in range(num_clients):
            bot_optimizers[i].zero_grad()
            top_optimizers[i].zero_grad()
        loss.backward()
        for i in range(num_clients):
            bot_optimizers[i].step()
            top_optimizers[i].step()

        if step % print_freq == 0:
            print("\tLoss: %.4f" % (loss.item()))

def test():
    all_auprc = []
    all_auroc = []
    all_loss = []

    for step in range(0, test_size, batch_size):
        labels = y_train[step:step+batch_size]
        try:
            labels = labels.cuda(device)
        except RuntimeError:
            labels = labels.cpu()
        labels = Variable(labels)

        # Exchange embeddings
        H_orig = [None] * num_clients
        with torch.no_grad():
            H_nograd = [None] * num_clients
        for i in range(num_clients):
            inputs = X_tests[i][step:step+batch_size]
            try:
                inputs = inputs.cuda(device)
            except RuntimeError:
                inputs = inputs.cpu()
            inputs = Variable(inputs)
            H_orig[i] = bot_models[i](inputs)
            with torch.no_grad():
                H_nograd[i] = bot_models[i](inputs)
            
            # Compress embedding / Quantization
            if quant_bin > 0:
                H_nograd[i] = quantize(H_nograd[i], theta, quant_bin)

                if args.withblockchain:
                    client_parameters = [list(map(int, row)) for row in H_nograd[i].tolist()]
                    blockchain_vfl_integrator.update_client_weights(blockchain_vfl_integrator.client_accounts[i], client_parameters)
        
        # embedding summation
        if args.withblockchain:
            # retrieve the embedding sum from smart contract
            blockchain_vfl_integrator.aggregate_weights()
            with torch.no_grad():
                sum_nograd = torch.tensor(blockchain_vfl_integrator.get_aggregated_weights())
        else:
            with torch.no_grad():
                sum_nograd = torch.sum(torch.stack(H_nograd),axis=0)

        if quant_bin > 0:
            sum_nograd = dequantize(sum_nograd, theta, quant_bin, num_clients)
        
        sum_embedding = torch.sum(torch.stack(H_orig),axis=0)
        if quant_bin > 0:
            sum_embedding.data = sum_nograd

        # client 0 with labels compute output
        with torch.no_grad():
            output = top_models[0](sum_embedding)
            all_loss.append(criterion(output[:, 1], labels).item())
            all_auroc.extend(output[:, 1].cpu().numpy())
            all_auprc.extend(output[:, 1].cpu().numpy())

    loss = sum(all_loss) / len(all_loss)
    auroc = roc_auc_score(y_test.numpy(), all_auroc)
    precision, recall, _ = precision_recall_curve(y_test.numpy(), all_auprc)
    auprc = auc(recall, precision)

    return (loss, auroc, auprc)

def run_save_test():
    test_loss, test_auroc, test_auprc = test()
    print('Loss: {:.2f} \tAUROC {:.2f} \tAUPRC {:.2f}'.format(test_loss, test_auroc, test_auprc))

    pkl_loss.append(test_loss)
    pkl_auprc.append(test_auprc)
    pkl_auroc.append(test_auroc)

    try:
        os.makedirs("./results/breast_cancer")
    except OSError:
        pass
    pickle.dump(pkl_loss, open(f'results/breast_cancer/loss_client{num_clients}_quant{quant_bin}_theta{theta}.pkl', 'wb'))
    pickle.dump(pkl_auprc, open(f'results/breast_cancer/auprc_client{num_clients}_quant{quant_bin}_theta{theta}.pkl', 'wb'))
    pickle.dump(pkl_auroc, open(f'results/breast_cancer/auroc_client{num_clients}_quant{quant_bin}_theta{theta}.pkl', 'wb'))

# initial loss
print('Initial test result')
run_save_test()

# main training loop
for epoch in range(num_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, num_epochs))
    train()
    run_save_test()