import torch
import torch.optim as optim
import numpy as np
torch.manual_seed(0)

if __name__ == '__main__':
    char_set = ['h', 'i', 'e', 'l', 'o']
    input_size = len(char_set)
    hidden_size = len(char_set)

    x_data = [[0, 1, 0, 2, 3, 3]]
    x_one_hot = [[[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0]]]
    y_data = [[1, 0, 2, 3, 3, 4]] 
    train_x = torch.FloatTensor(x_one_hot)
    train_y = torch.LongTensor(y_data)

    learning_rate = 0.1
    epochs = 100
    rnn = torch.nn.RNN(input_size, hidden_size, batch_first = True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), learning_rate)

    for i in range(epochs):
        optimizer.zero_grad()
        outputs, _status = rnn(train_x)
        loss = criterion(outputs.view(-1, input_size), train_y.view(-1))
        loss.backward()
        optimizer.step()

        result = outputs.data.numpy().argmax(axis=2)
        result_str = ''.join([char_set[c] for c in np.squeeze(result)])
        print(i, "loss: ", loss.item(), "prediction: ", result, "true Y:", y_data, "prediction str: ", result_str)
        