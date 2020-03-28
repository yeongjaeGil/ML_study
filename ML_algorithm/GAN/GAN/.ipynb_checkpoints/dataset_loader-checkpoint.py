from torch.utils.data import Dataset
from torch import from_numpy, tensor
import numpy as np


class Dataset(Dataset):
    def __init__ (self,DIR):
        xy = np.loadtxt(DIR, delimiter = ',', dtype= np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:,0:-1])
        self.y_data = from_numpy(xy[:,[-1]])
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
    