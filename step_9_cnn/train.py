from utils import Fer2013Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import numpy as np
import torch
import time


trainset = Fer2013Dataset('X_train.npy', 'Y_train.npy')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = Fer2013Dataset('X_test.npy', 'Y_test.npy')
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def evaluate(net, dataset):
    X = Variable(torch.from_numpy(dataset.samples).float())
    _, output = torch.max(net(X), 1)
    Yhat = output.data.numpy()
    return np.sum(Yhat == dataset.labels) / dataset.labels.shape[0]


t0 = time.time()
best_test_acc = 0

for epoch in range(20):  # loop over the dataset multiple times
    break
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data['image'].float(), data['label'].long()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:
            test_acc = evaluate(net, testset)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            print('[%d, %5d] loss: %.3f train acc: ? val acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / i, test_acc))
            running_loss = 0.0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, best_test_acc == test_acc)
    print('[%d] time: %.3f s' % (epoch, (time.time() - t0)/ (epoch + 1)))

train_acc = evaluate(net, trainset)
print('Training accuracy: %.3f' % train_acc)

print('Finished Training')
