
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from GAN.network import Discriminator
from GAN.network import Generator
import os
import torchvision
import torchvision.transforms as transforms

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class run_epoch(nn.Module):
    '''
    Manage model traning and evaluation
    '''

    def __init__(self, model, epoch, criterion, optimizer):
        super().__init__()
        self.model = model
        self.epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_graph = {'train':[],'test':[],'epoch':[]}

    def train(self, data, gpu = False):
        for inputs, outputs in data:
            if gpu:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            else:
                continue

            predict = self.model(inputs)
            loss = self.criterion(predict, outputs)
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def test(self, data, metric, gpu = False):
        for inputs, outputs in data:
            if gpu:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            else:
                continue
            
            predict = self.model(inputs)
            metric = metric(predict, outputs)
        
    def plot_loss(self, is_train=True):
        #이건 학습 종료 후 Logger를 사용해서 저장하고 읽어내자.
        '''
        plt.plot(loss_graph['epoch'], loss_graph['train']);
        plt.plot(loss_graph['epoch'], loss_graph['test']);
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
        '''
        pass

if __name__ == '__main__':
    input_size = 28*28
    latent_size = 64
    hidden_size = 256

    num_epochs = 200
    batch_size = 100

    sample_dir = "./result_gan/"

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))])
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                    #                    std=(0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    G = Generator(input_size,latent_size, hidden_size)
    D = Discriminator(input_size,hidden_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr = 0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0002)

    total_step = len(trainloader)





    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(trainloader):
            images = images.reshape(batch_size, -1).to(device)
            
            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ================================================================== #
            #                        Train the discriminator                     #
            # ================================================================== #

            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            d_loss = d_loss_real + d_loss_fake
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()
            
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            
            g_loss = criterion(outputs, real_labels)
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                            real_score.mean().item(), fake_score.mean().item()))
        
        # Save real images
        if (epoch+1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
        
        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    # Save the model checkpoints 
    torch.save(G.state_dict(), sample_dir + '/G.ckpt')
    torch.save(D.state_dict(), sample_dir + '/D.ckpt')