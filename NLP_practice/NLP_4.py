import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
print(os.getcwd())
os.chdir('ML_study/PyTorchZeroToAll-master')
torch.manual_seed(0)

def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator/(denominator+1e-7)

# make dataset to input
def build_dataset(time_series, seq_length):
    X_data = []
    Y_data = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [-1]]
        print(_x, '->', _y)
        X_data.append(_x)
        Y_data.append(_y)
    return np.array(X_data), np.array(Y_data)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers = layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias = True)
    
    def forward(self,x):
        x, _status = self.rnn(x)
        x = self.fc(x[:,-1])
        return x
if __name__ == '__main__':
    seq_length = 7
    data_dim = 5
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.01
    iterations = 500

    xy = np.loadtxt('data-02-stock_daily.csv', delimiter = ',')
    xy = xy[::-1] # reverse order
    train_size = int(len(xy)*0.7)
    train_set = xy[0:train_size]
    test_set = xy[train_size-seq_length:]

    train_set = minmax_scaler(train_set)
    test_set = minmax_scaler(test_set)

    train_x, train_y = build_dataset(train_set, seq_length)
    test_x, test_y = build_dataset(test_set, seq_length)

    train_tensor_x = torch.FloatTensor(train_x)
    train_tensor_y = torch.FloatTensor(train_y)

    test_tensor_x = torch.FloatTensor(test_x)
    test_tensor_y = torch.FloatTensor(test_y)
    
    model = Net(data_dim, hidden_dim, output_dim, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    epochs = 500
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_tensor_x)
        loss = criterion(outputs, train_tensor_y)
        loss.backward()
        optimizer.step()
        print(i, loss.item())

    
    plt.plot(test_y)
    plt.plot(model(test_tensor_x).data.numpy())
    #plt.show()