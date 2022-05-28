import numpy as np
import time
import random
import csv
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tifffile import imread
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from plot_fun import density_scatter


# Rebuild the entire dataset
rebuild_data = False

# Load the previous model, or a previously saved one by replacing my_checkpoint.pth.tar
load_model = True

# Saves the model as my_checkpoint.pth.tar in project directory
save_model = True

# Shows a Histogram of the significant wave height
show_histogram = False

# Train the model
run_training = True

# Fully evaluates the model
fully_evaluate_model = True

# Perform a buoy evaluation (requires buoy data)
buoy_evaluation = False

n = 200     # Pixel size of sub image
BATCH_SIZE = 10
EPOCHS = 1
learning_rate = 0.0001
MODEL_NAME = f'model-{int(time.time())}'

# Will run on GPU if you have cuda installed and enabled, otherwise CPU (GPU much Faster)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


class Build_Dataset():
    path = r'D:\CNN_storage\s1_output_dataset'
    count = 0
    training_data = []          # Stores pixel values and labels
    saved_numpy_arrays = 1      # Tracks the number of saved numpy arrays

    def make_training_data(self):
        # Read and shuffle the folder names
        folder = os.listdir(self.path)
        random.shuffle(folder)
        for f in tqdm(folder):
            try:
                path = os.path.join(self.path, f)

                # Read tif files
                img = imread(path)
                pxl_img = np.array(img)
                label = float(f.split('.tif')[0])
                self.training_data.append([pxl_img, label])
                self.count += 1

                # If numpy array holds 10000 images, then create a new array.
                if self.count % 10000 == 0:
                    np.random.shuffle(self.training_data)
                    np.save('training_data_' + str(self.saved_numpy_arrays) + '.npy', self.training_data)
                    self.training_data = []
                    self.saved_numpy_arrays += 1

            except Exception as e:
                print(str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data_' + str(self.saved_numpy_arrays) + '.npy', self.training_data)
        np.save('saved_numpy_arrays.npy', self.saved_numpy_arrays)
        print('Number of sub images in dataset: ', self.count)
        print('Number of saved numpy arrays: ', self.saved_numpy_arrays)


# The Convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (5, 5))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (5, 5))
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (5, 5))
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, (5, 5))
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)

        # Intermediary part that finds the value of self._to_linear
        x = torch.randn(n, n).view(-1, 1, n, n)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 1)                # Fully connected layer 2

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.batchnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.batchnorm2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.batchnorm3(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = self.batchnorm4(x)
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = self.batchnorm5(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            #print(self._to_linear)  # Input size of first fully connected layer
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def test(size=32):
    random_start = np.random.randint(len(test_x)-size)
    x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
    x, y = x.view(-1, 1, n, n).to(device), y.to(device)
    with torch.no_grad():
        val_loss = fwd_pass(x, y)
    return val_loss


def train():
    with open(f'{MODEL_NAME}.log', 'a') as f:
        loss_array = []
        val_loss_array = []

        # Splits the dataset into batch sizes. Training and testing in recorded in model.log
        for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
            batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, n, n)
            batch_y = train_y[i:i + BATCH_SIZE]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = fwd_pass(batch_x, batch_y, train=True)
            loss_array.append(loss.cpu().detach().numpy())

            if i % BATCH_SIZE/2 == 0:
                val_loss = test(size=BATCH_SIZE)
                val_loss_array.append(val_loss.cpu().detach().numpy())
                f.write(f'{MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss), 4)}, {round(float(val_loss), 4)}\n')

        # For the loss graph
        mean_val_loss=np.mean(val_loss_array)
        mean_loss=np.mean(loss_array)

        # These values are printed after training
        total_val_loss.append(mean_val_loss)
        total_train_loss.append(mean_loss)
        print(f'Training Loss: {mean_loss}. Validation Loss: {mean_val_loss}')


def fwd_pass(x, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(x)

    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
    return loss


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

    ax1.legend(loc=2)
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()


def buoy_eval():
    # Buoy data is generated in preproccesing and holds pixel values, time and position.
    boj_data = np.load('buoy_data.npy', allow_pickle=True)
    time = np.array([i[1] for i in boj_data])
    pos_1 = np.array([i[2] for i in boj_data])
    pos_2 = np.array([i[3] for i in boj_data])
    test_x = torch.Tensor(np.array([i[0] for i in boj_data])).view(-1, n, n)

    output_array = []
    for i in tqdm(range(0, len(test_x), BATCH_SIZE)):
        batch_x = test_x[i:i + BATCH_SIZE].view(-1, 1, n, n)
        batch_x = batch_x.to(device)
        with torch.no_grad():
            outputs = net(batch_x)
        outputs = outputs.detach().cpu().numpy()
        for j in range(len(outputs)):
            output_array.append(outputs[j])
    output_array = np.squeeze(output_array)

    # Writes the buoy evaluation to csv for comparison with buoy data. Choose path for output below
    with open(r'buoy_prediction.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(output_array)):
            data = [time[i], pos_1[i], pos_2[i], output_array[i]]
            writer.writerow(data)
    f.close()
    print("Buoy evaluation done")


def full_evaluation():
    test_x_filenames, test_y_filenames = [], []
    for i in range(np.load('saved_numpy_arrays.npy')):
        test_x_filenames.append('test_x_'+str(i+1)+'.npy')
        test_y_filenames.append('test_y_'+str(i+1)+'.npy')

    test_x = [np.load(f, allow_pickle=True) for f in tqdm(test_x_filenames)]  # Load in pixel images and labels
    test_x = np.concatenate(test_x)
    test_x = torch.Tensor(test_x).view(-1, n, n)

    test_y = [np.load(f, allow_pickle=True) for f in tqdm(test_y_filenames)]  # Load in pixel images and labels
    input_array = np.concatenate(test_y)

    output_array = []

    for i in tqdm(range(0, len(test_x), BATCH_SIZE)):

        batch_x = test_x[i:i + BATCH_SIZE].view(-1, 1, n, n)
        batch_x = batch_x.to(device)
        with torch.no_grad():
            outputs = net(batch_x)
        outputs = outputs.detach().cpu().numpy()

        for j in range(len(outputs)):
            output_array.append(outputs[j])

    output_array = np.squeeze(output_array)
    input_array = np.squeeze(input_array)
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    total = 0
    for k in range(len(output_array)):
        if abs(output_array[k]-input_array[k]) < 1:
            correct_1 += 1
        if abs(output_array[k]-input_array[k]) < 0.5:
            correct_2 += 1
        if abs(output_array[k] - input_array[k]) < 0.2:
            correct_3 += 1
        total += 1

    print("Accuracy within 20 centmeters:", round(correct_3 / total, 3))
    print("Accuracy within 50 centimeters:", round(correct_2 / total, 3))
    print("Accuracy within 100 centimeters:", round(correct_1 / total, 3))

    # Calculates RMSE, R-value and MAE
    mse = mean_squared_error(input_array, output_array)
    rmse = np.sqrt(mse)
    print(f"RMSE for full validation set: {rmse:.3f}")

    res = r2_score(input_array, output_array)
    res = np.sqrt(res)
    print(f"R-value for full validation set: {res:.3f}")

    mean_absolute_error = np.mean((output_array - input_array))
    print(f"Mean error: {mean_absolute_error:.3f}")

    mean_absolute_error = np.mean(np.abs(output_array - input_array))
    print(f"Mean absolute error: {mean_absolute_error:.3f}")

    density_scatter(input_array, output_array, bins=50)
    plt.show()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def load_dataset(iterations):
    filename = 'training_data_'+str(iterations+1)+'.npy'
    training_data = np.load(filename, allow_pickle=True)
    # Plot the first SAR pixel image using the code below
    # plt.imshow(training_data[0][0], cmap='gray')
    # plt.show()

    if show_histogram:
        bin = np.linspace(min(np.array([i[1] for i in training_data])), max(np.array([i[1] for i in training_data])), 60)
        plt.hist(np.array([i[1] for i in training_data]), bins=bin)
        plt.show()

    # Puts data and labels in torch format
    x = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, n, n)
    y = torch.Tensor(np.array([i[1] for i in training_data])).view(-1, 1)

    # Splits the dataset into training and testing using a random seed
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)
    print('training length: ', len(train_x))
    print('training label: ', len(train_y))
    print('test length: ', len(test_x))
    print('test label: ', len(test_y))
    return train_x, test_x, train_y, test_y


# Generate an instance of the Neural network
net = Net().to(device)
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

if rebuild_data:
    make_data = Build_Dataset()
    make_data.make_training_data()

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

if run_training:
    total_val_loss = []
    total_train_loss = []
    for epochs in range(EPOCHS):
        for iteration in range(np.load('saved_numpy_arrays.npy')):
            train_x, test_x, train_y, test_y = load_dataset(iteration)
            print('training on training_data_' + str(iteration + 1))
            train()

            if rebuild_data and epochs == 1:
                test_x_save = test_x.cpu().detach().numpy()
                test_y_save = test_y.cpu().detach().numpy()
                np.save('test_x_' + str(iteration + 1) + '.npy', test_x_save)
                np.save('test_y_' + str(iteration + 1) + '.npy', test_y_save)

    create_loss_graph(MODEL_NAME)
    print('Total Train Loss: ', np.mean(total_train_loss))
    print('Total Val Loss: ', np.mean(total_val_loss))

if fully_evaluate_model:
    full_evaluation()

if buoy_evaluation:
    buoy_eval()

if save_model:
    Question = input("Save model y or n: ")
    if Question == "y":
        checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
