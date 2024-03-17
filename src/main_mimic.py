import numpy as np
import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.distributions.binomial import Binomial

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models.in_hospital_mortality import utils
from sklearn.metrics import f1_score

# Arguments and parameters
parser = argparse.ArgumentParser(description='VFL without Blockchain')
parser.add_argument('--quant_bin', type=int, help='Number of quantization buckets', default=0)
parser.add_argument('--theta', type=float, metavar='T', help='Noise value (in range [0, 0.25])', default=0.00)
parser.add_argument('--load_data', type=bool, help='First time load data', default=False)
args = parser.parse_args()

num_clients = 4
quant_bin = args.quant_bin
theta = args.theta

# ML parameters
batch_size = 100
num_epochs = 1000
lr = 0.0001
print_freq = 10

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.manual_seed(707412115)

pkl_loss = []
pkl_f1 =[]

class LSTMBottom(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMBottom, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, output_size)
        out = torch.tanh(out)

        return out

class LSTMTop(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMTop, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)

        return out


class MimicDataset(Dataset):
    def __init__(self, X, coords_per):
        self.data = X[0]
        self.labels = X[1]
        self.coords_per = coords_per

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input0': self.data[idx][:,:self.coords_per],
            'input1': self.data[idx][:,self.coords_per:2*self.coords_per],
            'input2': self.data[idx][:,2*self.coords_per:3*self.coords_per],
            'input3': self.data[idx][:,3*self.coords_per:],
            'label': self.labels[idx]
        }
        return sample


def preprocess_data():
    # Load datasets
    # train 5918 ; val 1332 ; test 1379
    # 60 time frames, 76 features each
    train_raw, val_raw, test_raw = None, None, None

    if args.load_data:
        train_reader = InHospitalMortalityReader(dataset_dir='in-hospital-mortality/train',
                                                listfile='in-hospital-mortality/train_listfile.csv',
                                                period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir='scratch/in-hospital-mortality/train',
                                            listfile='in-hospital-mortality/val_listfile.csv',
                                            period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir='scratch/in-hospital-mortality/test',
                                                listfile='in-hospital-mortality/test_listfile.csv',
                                                period_length=48.0)

        discretizer = Discretizer(timestep=0.8,
                                store_masks=True,
                                impute_strategy='previous',
                                start_time='zero')

        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'ihm_ts1.0.input_str-previous.start_time-zero.normalizer'
        normalizer_state = os.path.join('mimic3-benchmarks-master/mimic3models/in_hospital_mortality', normalizer_state)
        normalizer.load_params(normalizer_state)

        train_raw = utils.load_data(train_reader, discretizer, normalizer)
        val_raw = utils.load_data(val_reader, discretizer, normalizer)
        test_raw = utils.load_data(test_reader, discretizer, normalizer)

        pickle.dump(train_raw, open('in-hospital-mortality/train_raw.pkl', 'wb'))
        pickle.dump(val_raw, open('in-hospital-mortality/val_raw.pkl', 'wb'))
        pickle.dump(test_raw, open('in-hospital-mortality/test_raw.pkl', 'wb'))

    else:
        train_raw = pickle.load(open('datasets/mimic-iii/train_raw.pkl', 'rb'))
        val_raw = pickle.load(open('datasets/mimic-iii/val_raw.pkl', 'rb'))
        test_raw = pickle.load(open('datasets/mimic-iii/test_raw.pkl', 'rb'))

    return (train_raw, val_raw, test_raw)

num_features = 76
coords_per = int(num_features/num_clients) # 19 features per client
embedding_size = 16
num_classes = 2

train_raw, val_raw, test_raw = preprocess_data()
train_set = MimicDataset(train_raw, coords_per=coords_per)
val_set = MimicDataset(val_raw, coords_per=coords_per)
test_set = MimicDataset(test_raw, coords_per=coords_per)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Make models for each client
top_models = []
bot_models = []
top_optimizers = []
bot_optimizers = []

for i in range(num_clients):
    top_model = None
    top_model = LSTMTop(input_size=embedding_size, output_size=num_classes)
    top_optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)
    top_models.append(top_model)
    top_optimizers.append(top_optimizer)
    top_model.to(device)
    top_model.float()
    cudnn.benchmark = True

    bot_model = None
    bot_model = LSTMBottom(input_size=coords_per, hidden_size=256, num_layers=1, output_size=embedding_size)
    bot_optimizer = torch.optim.Adam(bot_model.parameters(), lr=lr)
    bot_models.append(bot_model)
    bot_optimizers.append(bot_optimizer)
    bot_model.to(device)
    bot_model.float()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

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

def run_model(mode):

    data_loader = None
    if mode == 'train':
        data_loader = train_loader
    elif mode == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader

    all_labels = []
    all_predictions = []
    all_loss = []

    Hs = np.empty((len(data_loader), num_clients), dtype=object)
    Hs.fill([])
    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])

    for step, batch in enumerate(data_loader):
        labels = batch['label']
        labels = labels.to(device)
        labels = Variable(labels)

        # Exchange embeddings
        H_orig = [None] * num_clients
        with torch.no_grad():
            H_nograd = [None] * num_clients
        for i in range(num_clients):
            x_local = batch[f'input{i}']
            x_local = x_local.to(device)
            x_local = x_local.float()
            x_local = Variable(x_local)
            H_orig[i] = bot_models[i](x_local)
            with torch.no_grad():
                H_nograd[i] = bot_models[i](x_local)
            
            # Compress embedding / Quantization
            if quant_bin > 0:
                H_nograd[i] = quantize(H_nograd[i], theta, quant_bin)
        
        # embedding summation
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
        loss = criterion(outputs[0], labels)

        if mode == 'train':
            # compute gradient and do SGD step
            for i in range(num_clients):
                bot_optimizers[i].zero_grad()
                top_optimizers[i].zero_grad()
            loss.backward()
            for i in range(num_clients):
                bot_optimizers[i].step()
                top_optimizers[i].step()
        else:
            _, prediction = torch.max(outputs[0], 1)
            all_predictions.extend(prediction.numpy())
            all_labels.extend(labels.numpy())
            all_loss.append(loss.item())

        if (step + 1) % print_freq == 0:
            print("\tServer Iter [%d/%d] Loss: %.4f" % (step + 1, len(data_loader), loss.item()))

    if mode != 'train':
        f1 = f1_score(all_labels, all_predictions)
        loss = sum(all_loss) / len(all_loss)
        return (loss, f1)

def run_and_save(mode):
    test_loss, test_f1 = run_model(mode = mode)
    print('{} loss: {:.2f} \tF1 {:.2f}'.format(mode, test_loss, test_f1))

    pkl_loss.append(test_loss)
    pkl_f1.append(test_f1)

    pickle.dump(pkl_loss, open(f'results/mimic/{mode}_loss_client{num_clients}_quant{quant_bin}_theta{theta}.pkl', 'wb'))
    pickle.dump(pkl_f1, open(f'results/mimic/{mode}_f1_client{num_clients}_quant{quant_bin}_theta{theta}.pkl', 'wb'))

# initial loss
print('Initial test result')
run_and_save(mode = 'test')

# main training loop
for epoch in range(num_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, num_epochs))

    run_model(mode = 'train')
    run_and_save(mode = 'val')
    run_and_save(mode = 'test')