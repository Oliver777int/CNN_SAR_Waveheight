import numpy as np
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.optim as optim
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from tifffile import imread
import time

# Set to true if you want to build the data
rebuild_data = False
if torch.cuda.is_available():
    # device = torch.device("cpu")
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


class Build_Dataset():
    sarbilder = r'D:\Sar_download2'
    count = 0
    training_data = []

    def make_training_data(self):
        for f in tqdm(os.listdir(self.sarbilder)):
            try:
                path = os.path.join(self.sarbilder, f)
                img = imread(path)
                pxlimg=np.array(img)

                p = f.split('.tif')[0]
                label = float(p.split('_')[5])
                self.training_data.append([pxlimg, label])
                self.count += 1
            except Exception as e:
                print(str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('antal sarbilder i dataset = ', self.count)


# The neural network with 3 conv and 2 fully connected layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))    # Conv layer
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, (5, 5))   # Conv layer 2
        self.batchnorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 32, (5, 5))  # Conv layer 3
        self.batchnorm3 = nn.BatchNorm2d(32)

        # Intermediary part that finds the value of self._to_linear
        x = torch.randn(400, 400).view(-1, 1, 400, 400)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 64)  # Fully connected layer 1
        self.fc2 = nn.Linear(64, 1)                # Fully connected layer 2

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.batchnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.batchnorm2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.batchnorm3(x)

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print(self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test(size=32):
    random_start = np.random.randint(len(test_x)-size)
    x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
    x, y = x.view(-1, 1, 400, 400).to(device), y.to(device)
    with torch.no_grad():
        val_loss = fwd_pass(x, y)
    return val_loss


def train():
    BATCH_SIZE = 30
    EPOCHS = 4
    with open(f'{MODEL_NAME}.log', 'a') as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
                # print(i, i+BATCH_SIZE)
                batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, 400, 400)
                batch_y = train_y[i:i + BATCH_SIZE]

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                loss = fwd_pass(batch_x, batch_y, train=True)
                if i % BATCH_SIZE/2 == 0:
                    val_loss = test(size=BATCH_SIZE)
                    f.write(f'{MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss), 4)}, {round(float(val_loss), 4)}\n')

            print(f'Epoch: {epoch}. Loss: {loss}. Validation Loss: {val_loss}')


def fwd_pass(x, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(x)
    #alpha = np.array(outputs.cpu())
    #beta = np.array(y.cpu())
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return loss #, alpha, beta


def create_loss_graph(model_name):
    contents = open(f"{model_name}.log", "r").read().split('\n')
    times = []
    losses = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, loss, val_loss = c.split(",")

            times.append(float(timestamp))
            losses.append(float(loss))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    # ax1.plot(times, accuracies, label="acc")
    # ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()
#val_loss=test(size=8)
#print(val_loss)

# Only rebuilds the data if rebuild_data is true
if rebuild_data:
    make_data = Build_Dataset()
    make_data.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True)    # Load in pixelimages and labels

# plt.imshow(training_data[0][0], cmap='gray')
# plt.show()

# Separates data into testing and training, both the images and the labels

x = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, 400, 400)
x = (x - torch.mean(x))/torch.std(x)
y = torch.Tensor(np.array([i[1] for i in training_data])).view(-1, 1)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

# print(len(train_x))
# print(len(test_x))


MODEL_NAME = f'model-{int(time.time())}'
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

print(MODEL_NAME)
train()
create_loss_graph(MODEL_NAME)




