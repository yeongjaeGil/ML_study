import torch
import torch.optim as optim
import numpy as np
torch.manual_seed(0)

if __name__ == '__main__':
    sample = ' if you want you'
    char_set = list(set(sample))
    char_dic = {c: i for i,c in enumerate(char_set)}
    print(char_dic)
    dic_size = len(char_dic)
    hidden_size = len(char_dic)
    learning_rate = 0.1
    sample_idx = [char_dic[c] for c in sample]
    x_data = [sample_idx[:]]
    x_one_hot = [np.eye(dic_size)[x] for x in x_data]
    y_data = [sample_idx[:]]

    x = torch.FloatTensor(x_one_hot)
    y = torch.LongTensor(y_data)

    epochs = 100
    rnn = torch.nn.RNN(dic_size, hidden_size, batch_first = True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), learning_rate)

    for i in range(epochs):
        optimizer.zero_grad()
        outputs, _status = rnn(x)
        #print(outputs)
        print(outputs.view(-1, dic_size))
        print(y.view(-1))
        loss = criterion(outputs.view(-1, dic_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        result = outputs.data.numpy().argmax(axis=2)
        result_str = ''.join([char_set[c] for c in np.squeeze(result)])
        #print(i, 'loss: ', loss.item(), "prediction: ", result, "true Y: ", y_data, 'prediction str: ', result_str)
