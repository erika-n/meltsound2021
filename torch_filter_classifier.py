# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from scipy.io import wavfile
import sound_functions as sf





def loadData(input_folder, n_per_file=5, files=5, clip_size=1000, filter_n=10, filter_step=500):
    ### loaad data from folder, convert to filtered, shuffle with labels
    data = np.ndarray((n_per_file*files, filter_n, clip_size))
    labels = np.ndarray((n_per_file*files))
    speed = 10 #int(np.random.randint(4) + 1)


    for path, _, filenames in os.walk(input_folder):
        file_i = 0
        for i, filename in enumerate(filenames):
            if file_i >= files:
                break
            if filename[-4:] == '.wav':
                file_i += 1
                # label nums are in filenames (for consistency)
                label_num = int(filename[:2])
                wavdata = sf.getWav(os.path.join(path, filename))
                wavdata = wavdata[::speed]
                for j in range(n_per_file):
                    # create filtered clip
                    start =  j*clip_size#np.random.randint(wavdata.shape[0] - clip_size -1)
                    end = start + clip_size
                    clip = wavdata[start:end]
                    clip_data = sf.filterBank(clip, order=2, n=filter_n, step=filter_step)
                    data[int(i*n_per_file + j)] = clip_data
                    labels[int(i*n_per_file + j)] = label_num
                
            

 


    # shuffle
    p_data = np.random.permutation(data.shape[0])
    data = data[p_data]
    labels = labels[p_data]

    n_labels = np.max(labels) + 1


    # convert to torch tensors
    data = torch.from_numpy(data).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    return data, labels



class Net(nn.Module):
    def __init__(self, clip_size, n_y, filter_n):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=clip_size, out_features=100)
        # self.conv1 = nn.Conv1d(filter_n, 33, 3, stride=2)
        # self.pool = nn.MaxPool1d(3, stride=2)
        # # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 80)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(80*filter_n, 40)
        self.fc5 = nn.Linear(40, n_y)

    def forward(self, x):
       
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.pool(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x





def train_epoch(x_train, y_train, input_folder, optimizer, net, criterion, clip_size=100, batch_size=10):
    running_loss = 0.0
    

    n_batches = int((list(x_train.size())[0])/batch_size)
    # print('batch_size', batch_size)
    # print('n_batches', n_batches)
    # print('x_train size', x_train.size(), list(x_train.size())[0])
    # print(int((200 - 1)/100))
    # exit()
    for i in range(n_batches):
        # get the inputs; data is a list of [inputs, labels]
        inputs = x_train[batch_size*i:batch_size*(i + 1)]
        labels = y_train[batch_size*i:batch_size*(i + 1)]


        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
     
        outputs = net(inputs)


        #print('outputs', outputs)
 
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        #print('loss', loss.item())
        # print statistics
        running_loss += loss.item()

    return running_loss/n_batches

def test(input_folder, net, x_test, y_test, criterion, batch_size=10):

    # test whole dataset
    correct = 0
    total = 0

    n_batches = int((list(x_test.size())[0] - batch_size - 1)/batch_size)
    running_loss = 0

    #print(n_batches)
    with torch.no_grad():
        for i in range(n_batches):
            inputs = x_test[i*batch_size:(i + 1)*batch_size]
            labels = y_test[i*batch_size:(i + 1)*batch_size]
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    inputs = x_test[:batch_size]
    labels = y_test[:batch_size]
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    print('loss', running_loss/n_batches)
    print('labels[0]\t', labels)
    print('predicted[0]\t', predicted)
    print('accuracy:', 	100 * correct / total)
    print('\n')


def main():
    rate = 44100
    seconds=0.5
    clip_size = 1000 #int(seconds*rate)
    epochs = 1000
    batch_size = 10
    n_per_file = 50
    files = 5
    input_folder = '../sounds/songsinmyhead/b'
    model_path = './music_net.pth'
    train_net = True
    load_net = False
    learning_rate = 0.0001
    filter_n = 20
    filter_step = 500




    net = Net(clip_size, files + 1, filter_n)

    print(net)

    if load_net:
        net.load_state_dict(torch.load(model_path))

    if train_net:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        for i in range(epochs):
            x, y = loadData(input_folder, filter_n = filter_n, n_per_file=n_per_file, files=files, clip_size=clip_size, filter_step=500)

            loss = train_epoch(x, y, input_folder, optimizer,
                    net, criterion, clip_size=clip_size, batch_size = batch_size )

            print(i, loss)

            if i % 10 ==0:
                print("\n")
                print('train', i)
                test(input_folder, net, x, y, criterion)
                x, y = loadData(input_folder, filter_n = filter_n, n_per_file=n_per_file, files=files, clip_size=clip_size, filter_step=filter_step)
                print('test', i)
                test(input_folder, net, x, y, criterion)
                
                torch.save(net.state_dict(), model_path)
                print('---------')

    else:
        x, y = loadData(input_folder, n_per_file=n_per_file, files=files, clip_size=clip_size, filter_step=filter_sgep)
        test(input_folder, net, x, y, criterion)

if __name__ == '__main__':
    main()
