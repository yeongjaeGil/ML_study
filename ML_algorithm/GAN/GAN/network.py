import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super(Generator,self).__init__()
        self.layer0 = nn.Linear(latent_size, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, input_size)
        self.act = nn.ReLU()

    def forward(self, x):
        hidden = self.layer0(x)
        hidden = self.act(hidden)
        hidden = self.layer1(hidden)
        hidden = self.act(hidden)
        hidden = self.layer2(hidden)
        hidden = self.act(hidden)
        hidden = self.layer3(hidden)
        output = F.tanh(hidden)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.layer0 = nn.Linear(input_size,hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = self.layer0(x)
        print(hidden.shape)
        hidden = F.leaky_relu(hidden,0.2)
        print(hidden.shape)
        hidden = self.layer1(x)
        hidden = F.leaky_relu(hidden,0.2)
        hidden = self.layer2(x)
        output = F.sigmoid(hidden)
        return output

