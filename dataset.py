import torch
import numpy as np
from torch.utils.data import Dataset

class Mouse_Dataset(Dataset):
    def __init__(self, mk1_5_data, sal_data):
        
        self.data = np.concatenate((mk1_5_data, sal_data), axis=0)
        x_len = mk1_5_data.shape[2]
        y_len = mk1_5_data.shape[3]
        #print('image shape: ', x_len, y_len)
        
        self.data = np.reshape(self.data, (-1, x_len, y_len))
        self.labels = np.concatenate((np.ones(mk1_5_data.shape[0] * mk1_5_data.shape[1], dtype=np.compat.long),
                                      np.zeros(sal_data.shape[0] * sal_data.shape[1], dtype=np.compat.long)))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label