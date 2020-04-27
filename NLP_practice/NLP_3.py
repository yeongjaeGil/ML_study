import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn  = nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias = True)
    
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    sentence =( "if you want to build a ship, don't dtrum up people together to " \
                "collect wood and don't assign them tasks and work, but rather " \
                "teach them to long for the endless immensity of the sea.")
    char_set = list(set(sentence))
    char_dic = {c: i for i, c in enumerate(char_set)}
    dic_size = len(char_dic)
    hidden_size = len(char_dic)
    sequence_length = 10
    learning_rate = 0.1
    epochs = 100
    x_data = []
    y_data = []
    
    for i in range(0, len(sentence) - sequence_length):
        x_str = sentence[i:i + sequence_length]
        y_str = sentence[i + 1:i + sequence_length + 1]
        print(i, x_str, '->', y_str)

        x_data.append([char_dic[c] for c in x_str])
        y_data.append([char_dic[c] for c in y_str])
    x_one_hot = [np.eye(dic_size)[x] for x in x_data]
    
    x = torch.FloatTensor(x_one_hot)
    y = torch.LongTensor(y_data)
    model = Net(dic_size, hidden_size, 2)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    epoches = 100
    for i in range(epoches):
        optimizer.zero_grad()
        # 171 x 10 x 25 가 나오는 이유: 10개의 character씩 171문장이 있고 거기 label이 25개가 있다.
        outputs = model(x)
        # print(outputs)
        # print(outputs.size())
        loss = criterion(outputs.view(-1, dic_size), y.view(-1))
        loss.backward()
        optimizer.step()

        results = outputs.argmax(dim=2)
        predict_str = ""
        
        for j, result in enumerate(results):
            #print(result[-1])
            #print(j)
            #print(i, j, ''.join([char_set[t] for t in result]), loss.item())
            if j == 0:
                predict_str += ''.join([char_set[t] for t in result])
            else:
                # 예상되는 단어의 맨 마지막이 다음에 올 예측 값이라서 그렇다.
                predict_str += char_set[result[-1]]
        print(predict_str)