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


rate = 44100
# seconds=0.25
# frames_per_second = 30
fftwidth = 400
timewidth = 100
epochs = 1000
batch_size = 20
n_batches = 10
max_per_file = 200
files = 10
input_folder = '../sounds/songsinmyhead/b'
model_path = './models/music_net_fft3.pth'
train_net = True
load_net = False
learning_rate = 0.0001
filter_n = 30
filter_step = 200

def shuffle(data, labels, n):
    p_data = np.random.permutation(data.shape[0])
    data = data[p_data]
    labels = labels[p_data]
    return data[:n], labels[:n]


def loadData(input_folder):
    ### loaad data from folder, convert to filtered, shuffle with labels
    data = None
    labels = None

    for path, _, filenames in os.walk(input_folder):
        file_i = 0
        for i, filename in enumerate(filenames):

            if file_i >= files:
                break
            if filename[-4:] == '.wav':
                print('loading file ', filename)
                file_i += 1
                # label nums are in filenames (for consistency)
                label_num = int(filename[:2])
                wav_data = sf.getWav(os.path.join(path, filename))
                fft_data = sf.getFFT(wav_data, fftwidth, timewidth)
                print('fft data shape', fft_data.shape)
                
                if data is not None:
                    data = np.concatenate((data, fft_data))
                else:
                    data = fft_data

                label_nums = np.full((fft_data.shape[0]), label_num)
                if labels is not None:
                    labels = np.concatenate((labels, label_nums))
                else:
                    labels = label_nums
                # wavdata = wavdata[::speed]
                #file_start = np.random.randint(wavdata.shape[0] - n_per_file*clip_size - 1)
                # for j in range(max_per_file):
                #     # create filtered clip
                #     start =  j*clip_size#np.random.randint(wavdata.shape[0] - clip_size -1)
                #     end = start + clip_size
                #     clip = wavdata[start:end]
                #     clip_data = sf.filterBank(clip, order=2, n=filter_n, step=filter_step)
                #     data[int(i*max_per_file + j)] = clip_data
                #     labels[int(i*max_per_file + j)] = label_num
                
            

 

    print('data.shape', data.shape)

    # shuffle
    p_data = np.random.permutation(data.shape[0])
    data = data[p_data]
    labels = labels[p_data]

    n_labels = np.max(labels) + 1

    test_n = int(0.5*data.shape[0])
    test_data = data[:test_n]
    test_labels = labels[:test_n]
    train_data = data[test_n:]
    train_labels = labels[test_n:]

    # convert to torch tensors
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)



    return train_data, train_labels, test_data, test_labels



class MCNet(nn.Module):
    def __init__(self,  timewidth, fftwidth, n_y, channels=2):
        super(MCNet, self).__init__()

        self.conv1 = nn.Conv2d(2, 2, 3, stride=2, groups=2)
        self.pool1 = nn.MaxPool2d(3, 1)
        self.conv2 = nn.Conv2d(2, 2, 3, 2, groups=2)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.fc2 = nn.Linear(4032, 40)
        self.fc3 = nn.Linear(40, n_y)

    def forward(self, x):
       
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.pool1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





def train_epoch(net, x_train, y_train, criterion, optimizer,   batch_size=10):

    
    running_loss = 0.0
    

    n_batches = int((list(x_train.size())[0])/batch_size)
    # print('batch_size', batch_size)
    # print('n_batches', n_batches)
    # print('x_train size', x_train.size(), list(x_train.size())[0])
    # print(int((200 - 1)/100))

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

def test( net, x_test, y_test, criterion, batch_size=10):

    # test whole dataset
    correct = 0
    total = 0

    n_batches = int((list(x_test.size())[0] )/batch_size)
    print('x test size', x_test.size())
    print('test n batches', n_batches)

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

def loadNet():
    net = MCNet(timewidth, fftwidth, files + 1)

    print(net)

    net.load_state_dict(torch.load(model_path))
    return net


def main():





    net = MCNet(timewidth, fftwidth, files + 1)
    
    print(net)

    if load_net:
        net.load_state_dict(torch.load(model_path))

    if train_net:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        x_train, y_train, x_test, y_test = loadData(input_folder)

        print('x train shape', x_train.shape)
        print('y train shape', y_train.shape)
        print('x test shape', x_test.shape)
        print('y test shape', y_test.shape)


        for i in range(epochs):
            c_x_train, c_y_train = shuffle(x_train, y_train, n_batches*batch_size)



            loss = train_epoch(net, c_x_train, c_y_train, criterion, optimizer, batch_size = batch_size )

            print(i, loss)

            if i % 10 ==0:
                print("\n")
                print('train', i)
                test(net, c_x_train, c_y_train, criterion)
                print('test', i)
                c_x_test, c_y_test = shuffle(x_test, y_test, n_batches*batch_size)
                print('c x test shape', c_x_test.shape)
                print('c y test shape', c_y_test.shape)
 
                test( net, c_x_test, c_y_test, criterion)
                
                torch.save(net.state_dict(), model_path)
                print('---------')

    # else:
    #     x, y = loadData(input_folder, n_per_file=n_per_file, files=files, clip_size=clip_size, filter_step=filter_sgep)
    #     test(input_folder, net, x, y, criterion)

if __name__ == '__main__':
    main()
