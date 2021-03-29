# This is a simple example of comparing a standard cross entropy loss
# and using the differentiable argmax with Fenchel Young loss.
# The standard PyTorch tutorial code is used as a baseline, with minor
# changes to show how to use the differentable optimizer.

# Comparison results of a simple network, trained over 50 epochs
#                   |          MODE
# Results  (%)      |   0       1     2
# ----------------------------------------
# Accuracy of     0 :   77     51     75
# Accuracy of     1 :   81     52     68
# Accuracy of     2 :   45     17     47
# Accuracy of     3 :   45     25     46
# Accuracy of     4 :   58     28     63
# Accuracy of     5 :   51     32     60
# Accuracy of     6 :   65     60     69
# Accuracy of     7 :   72     53     67
# Accuracy of     8 :   75     61     67
# Accuracy of     9 :   67     52     80
#
# Note: 
# These results are not meant to be SOTA, but simply to show how once
#   can use these perturbed optimizers.
# Mode 0 = Cross entropy loss
# Mode 1 = argmax with MSE Loss
# Mode 2 = argmax with Fenchel Young Loss


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import perturbations
import fenchel_young as fy

BATCH_SIZE = 32
EPOCHS = 50

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def argmax(x, axis=-1):
  	return F.one_hot(torch.argmax(x, dim=axis), list(x.shape)[axis]).float()

pert_argmax = perturbations.perturbed(argmax, num_samples=100, sigma=0.5, noise='gumbel', batched=True, device=device)


def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for data in data_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # If using argmax, we need to convert single index into one-hot of length 10
        if not isinstance(criterion, nn.CrossEntropyLoss):
            labels = F.one_hot(labels, 10).float()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        if isinstance(criterion, nn.MSELoss):
            outputs = pert_argmax(outputs)
        loss = criterion(outputs, labels).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def test(model, data_loader):
    model.eval()
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    mode_choices = [0, 1, 2]
    parser = argparse.ArgumentParser(description='Perturbations Demo.')
    parser.add_argument("--mode", dest='mode', choices=mode_choices, default=0, type=int,
        help="0: Cross Entropy Loss, 1: argmax with MSE, 2: argmax with Fenchel Young Loss")
    args = parser.parse_args()
    assert args.mode in mode_choices

    model = Net()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set loss from mode
    if args.mode == 0:
        print("Using cross entropy loss")
        criterion = nn.CrossEntropyLoss()
    elif args.mode == 1:
        print("Using argmax with MSE loss")
        criterion = nn.MSELoss()
    else:
        print("Using argmax with Fenchel Young loss")
        criterion = fy.FenchelYoungLoss(argmax, num_samples=100, sigma=0.5, noise='gumbel', batched=True, device=device)
    
    # Train
    for epoch in range(EPOCHS):
        loss = train_epoch(model, trainloader, criterion, optimizer)
        print('Epoch {:>3}, Loss: {:.4f}'.format(epoch+1, loss))

    # Test
    test(model, testloader)

