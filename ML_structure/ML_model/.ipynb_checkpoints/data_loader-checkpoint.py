import torch
def dataLoader(train_dataset, val_dataset):
    batch_size = 64
    shuffle = True
    use_cuda = torch.cuda.is_available()
    pin_memory = True
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,pin_memory=True)
    return train_loader, val_loader
# 이것도 하나씩 해서 분절하는 게 좋을 듯
'''
def dataLoader(dataset):
    batch_size = 64
    shuffle = True
    use_cuda = torch.cuda.is_available()
    pin_memory = True
    data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True, pin_memory=True)
    return data_loader
'''