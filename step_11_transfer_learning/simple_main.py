"""
Convolutional neural network for face authentication
"""

from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import cv2


class MeDataset(Dataset):
    """Me dataset.

    Each sample is 1 x 1 x 48 x 48, and each label is a scalar.
    """

    def __init__(self, sample_path: str, label_path: str, transform=None):
        """
        Args:
            sample_path: Path to `.npy` file containing samples nxd.
            label_path: Path to `.npy` file containign labels nx1.
        """
        self._samples = np.load(sample_path)
        self._labels = np.load(label_path)
        self._samples = self._samples.reshape((-1, 1, 48, 48))

        self.X = Variable(torch.from_numpy(self._samples)).float()
        self.Y = Variable(torch.from_numpy(self._labels)).float()
        self.transform = transform

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        image = self._samples[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': self._labels[idx]}


def transform_train(image):
    # if np.random.random() > 0.5:
    #     image = cv2.flip(image.reshape(48, 48), 1).reshape(1, 48, 48).astype(np.int64)
    return image

trainset = MeDataset('me_X_train.npy', 'me_Y_train.npy', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10)


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


pretrained_model = torch.load('model_best.pth')
net.load_state_dict(pretrained_model['state_dict'])

for param in net.parameters():
    param.requires_grad = False
net.fc3 = nn.Linear(48, 2)

Y = trainset.Y.data.numpy().astype(np.int)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        Yhat = np.argmax(net(trainset.X).data.numpy(), axis=1)
        train_acc = float(np.sum(Yhat == Y) / Y.shape[0])
        print('[%d, %5d] loss: %.3f train acc %.3f' % (
            epoch, i, running_loss / (i + 1), train_acc))

# data = next(iter(trainloader))
# inputs = Variable(data['image'].float())
# labels = Variable(data['label'].long())
# print(inputs[0][0][0][:10])
# outputs = net(inputs)
# X = outputs.data.numpy()

# w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(np.eye(2)[np.ravel(Y)])
# Yhat = np.argmax(X.dot(w), 1)
# train_acc = float(np.sum(Yhat == Y) / Y.shape[0])
# print(train_acc)