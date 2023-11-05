"""
VFL algorithm
"""

import torch

# Set up input arguments
parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='./')
parser.add_argument('--num_clients', type=int, help='Number of clients to split data between vertically',
                        default=4)
parser.add_argument('--epochs', default=600, type=int, metavar='N', help='number of total epochs to run (default: 600)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--quant_bin', type=int, help='Number of quantization buckets', default=0)
parser.add_argument('--theta', default=0.25, type=float,
                    metavar='T', help='theta value (default: 0.25)')
parser.add_argument('-r', '--resume', type=str, help='path to latest checkpoint (default: none)', default='')

# Parse input arguments
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_clients = args.num_clients

'''
Task 1: Differential Privacy algorithm to add noise and quantize each values into integer to reduce communication cost.
quantize() function adds noises at the client side.
dequantize() function happends at the server side to convert the integer model update sum to an equivalent floating point model update sum.
This algorithm is proven to reduce significant communication cost with negligible lost in accuracy.
Linh: This may need modification, I can take care of this task.
'''

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

def save_checkpoint(state, filename):
    torch.save(state, filename)

''' Load dataset task '''
dset_train = None
train_loader = None

dset_val = None
test_loader = None

classes = dset_train.classes
print(len(classes), classes)

losses = []
accs_train = []
accs_test = []

best_acc = 0.0
best_loss = 0.0
start_epoch = 0

models = []
optimizers = []



'''
Task 2: Initialize the neural network model for clients and server
'''
# Make models for each client
for i in range(num_clients+1):
    if i == num_clients:
        model = None
    else:
        model = None
    optimizer = None

    models.append(model)
    optimizers.append(optimizer)

# Loss and Optimizer
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
coords_per = 16

'''
Optionally resume from a checkpoint.
This is in case the training need to be run multiple times due to time limit on AiMOS.
Ignore this step for now.
'''
if args.resume != '':
    for i in range(args.num_clients+1):
        if args.quant_bin == 0:
            cpfile = os.path.join('checkpoints/{}_{}clients_{}lr_noquant.pth.tar'.format(i, args.num_clients, args.lr))
        else:
            cpfile = os.path.join('checkpoints/{}_{}clients_{}lr_{}bins_{}theta.pth.tar'.format(i, args.num_clients, args.lr, args.quant_bin, args.theta))
        if os.path.isfile(cpfile):
            print("=> loading checkpoint")
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(cpfile, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            start_epoch = checkpoint["epoch"]
            models[i].load_state_dict(checkpoint["state_dict"])
            optimizers[i].load_state_dict(checkpoint["optimizer"])
            losses = pickle.load(open(f'loss_cifar_lr{args.lr}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            accs_train = pickle.load(open(f'accs_train_cifar_lr{args.lr}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            accs_test = pickle.load(open(f'accs_test_cifar_lr{args.lr}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found")

'''
Task 3: Evaluate and save current loss and accuracy
'''
def save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, step, train_size):
    avg_train_acc, avg_loss = eval(models, train_loader)
    avg_test_acc, _ = eval(models, test_loader)

    losses.append(avg_loss)
    accs_train.append(avg_train_acc)
    accs_test.append(avg_test_acc)

    print('Iter [%d/%d]: Test Acc: %.2f - Train Acc: %.2f - Loss: %.4f' 
            % (step + 1, train_size, avg_test_acc.item(), avg_train_acc.item(), avg_loss.item()))


'''
Task 4: Training algorithm
'''
def train(models, optimizers):
    """
    Train all clients on all batches 
    """

    train_size = len(train_loader)
    server_model = models[-1]
    #server_optimizer = optimizers[-1]

    Hs = np.empty((len(train_loader), num_clients), dtype=object)
    Hs.fill([])
    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])

    for step, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)
        # Exchange embeddings
        H_orig = [None] * num_clients
        with torch.no_grad():
            H_nograd = [None] * num_clients
        for i in range(num_clients):
            # split images into equal portions
            r = math.floor(i/2)
            c = i % 2
            section = coords_per
            x_local = inputs[:,:,
                            section*r:section*(r+1),
                            section*c:section*(c+1)]
            x_local = torch.transpose(x_local,0,1)
            H_orig[i] = models[i](x_local)
            with torch.no_grad():
                H_nograd[i] = models[i](x_local)
            
            # Compress embedding / Quantization
            if args.quant_bin > 0:
                H_nograd[i] = quantize(H_nograd[i], args.theta, args.quant_bin)
        
        # embedding summation
        with torch.no_grad():
            sum_nograd = torch.sum(torch.stack(H_nograd),axis=0)
        if args.quant_bin > 0:
            sum_nograd = dequantize(sum_nograd, args.theta, args.quant_bin, args.num_clients)
        
        sum_embedding = torch.sum(torch.stack(H_orig),axis=0)
        if args.quant_bin > 0:
            sum_embedding.data = sum_nograd

        # compute output
        outputs = server_model(sum_embedding)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        for i in range(args.num_clients+1):
            optimizers[i].zero_grad()
        loss.backward()
        if next(models[0].parameters()).grad is None:
            print("No gradient in embedding model!")
        elif next(models[0].parameters()).grad[0][0][0][0] == 0:
            print("Zero gradient!")
        for i in range(args.num_clients+1):
            optimizers[i].step()

        if (step + 1) % args.print_freq == 0:
            print("\tServer Iter [%d/%d] Loss: %.4f" % (step + 1, train_size, loss.item()))


'''
Task 5: Calculate loss and accuracy for a given data_loader
'''
def eval(models, data_loader):
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for _, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # Get current embeddings
            H_new = [None] * num_clients
            for i in range(num_clients):
                # split images into equal portions
                r = math.floor(i/2)
                c = i % 2
                section = coords_per
                x_local = inputs[:,:,
                                section*r:section*(r+1),
                                section*c:section*(c+1)]
                x_local = torch.transpose(x_local,0,1)
                H_new[i] = models[i](x_local)
            # compute output
            outputs = models[-1](torch.sum(torch.stack(H_new),axis=0))
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss

# Get initial loss/accuracy
if start_epoch == 0:
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, 0, len(train_loader))
# Training / Eval loop
train_size = len(train_loader)
for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print(f'Clients: {args.num_clients}, Quant_bin: {args.quant_bin}, Theta: {args.theta}')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    train(models, optimizers, epoch)
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, epoch, train_size)

    '''
    # Uncomment if need to save checkpoint
    # Checkpoints will be saved to `checkpoints` folder, make sure it exists
    for i in range(args.num_clients+1):
        if args.quant_bin == 0:
            save_filename = os.path.join('checkpoints/cifar_results/{}_{}clients_{}lr_noquant.pth.tar'.format(i, args.num_clients, args.lr))
        else:
            save_filename = os.path.join('checkpoints/cifar_results/{}_{}clients_{}lr_{}bins_{}theta.pth.tar'.format(i, args.num_clients, args.lr, args.quant_bin, args.theta))
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": models[i].state_dict(),
                "optimizer": optimizers[i].state_dict(),
            },
            filename=save_filename,
        )
        print(f"saved to '{save_filename}'")

    print('Time taken: %.2f sec.' % (time.time() - start))
    '''
