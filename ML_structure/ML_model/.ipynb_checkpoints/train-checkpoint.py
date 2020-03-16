import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network.CNN import CNNClassifier
from data_loader import dataLoader
from datasets import Datasets
import argparse
import yaml
import logging
from utils import get_logger
from utils import SummaryWriterDummy

if __name__ == "__main__":
    #formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logging.basicConfig(filename='logfile.log', level=logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logger = get_logger('ML_struct')
    parser = argparse.ArgumentParser(description='Character-level convolutional neural network for text classification')
    parser.add_argument('--config','--c', type=str, metavar='yaml FILE', help='where to load YAML configuration')
    args = parser.parse_args()
    print(args)
    print(args.config)
    yaml_file = args.config
    with open(yaml_file) as f:
        cfg = yaml.load(f)
    print(cfg)
    
    train_dataset, val_dataset = Datasets() # 이건 다른 path일때는 어떻게?
    train_loader, val_loader=dataLoader(train_dataset, val_dataset)
    cnn = CNNClassifier()
    device = torch.device("cuda")
    print(device)
    if torch.cuda.device_count() >1 :
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #cnn.to(device)
        cnn = nn.DataParallel(cnn)
        cnn.to(device)
    else:
        cnn.to(device)
        #mytensor = my_tensor.to(device)

    # loss
    criterion = nn.CrossEntropyLoss()
    # backpropagation method
    learning_rate = cfg['Model']['lr']#1e-3
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    # hyper-parameters
    use_cuda = torch.cuda.is_available()
    num_epochs = cfg['Model']['num_epochs']
    num_batches = len(train_loader)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            x, label = data
            if use_cuda:
                x = x.to(device)
                label = label.to(device)
            # grad init
            # test for multi-GPU
            #print("OUTSIDE: input size",x.size(),"output size",label.size())
            optimizer.zero_grad()
            # forward propagation
            model_output = cnn(x)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation 
            loss.backward()
            # weight update
            optimizer.step()
            # train_loss summary
            train_loss += loss.item()
            # del (memory issue)
            del loss
            del model_output
            # 학습과정 출력
            if (i+1) % 100 == 0: # every 100 mini-batches
                with torch.no_grad(): # very very very very important!!!
                    val_loss = 0.0
                    for j, val in enumerate(val_loader):
                        val_x, val_label = val
                        if use_cuda:
                            val_x = val_x.to(device)
                            val_label =val_label.to(device)
                        val_output = cnn(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss
                logger.info("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | val loss: {:.4f}".format(
                    epoch+1, num_epochs, i+1, num_batches, train_loss / 100, val_loss / len(val_loader)
                ))            

                train_loss_list.append(train_loss/100)
                val_loss_list.append(val_loss/len(val_loader))
                train_loss = 0.0
    
    save_path = "cnn_model.pth"
    #이걸로 Loss를 알아내야 한다.
    #writers = [SummaryWriterDummy(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]
    torch.save({'epoch': epoch,
                #'log': {
                #        'train': rs['train'].get_dict(),
                #        'valid': rs['valid'].get_dict(),
                #        'test': rs['test'].get_dict(),
                #        },
                'optimizer': optimizer.state_dict(),
                'model': cnn.state_dict(),
                }, save_path)
    logger.info('Done...!')
    
    