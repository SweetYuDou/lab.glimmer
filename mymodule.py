import torch
from torch import nn
from torch.utils.data import Dataset
import pickle


class MyDatasets(Dataset):
    def __init__(self, data_route,transform=None):
        with open(data_route, 'rb') as file:
            data_dict = pickle.load(file, encoding='iso-8859-1')

        raw_x = data_dict['x']
        raw_y = data_dict['y']
        self.data_x = raw_x.reshape((raw_x.shape[0], 3, 32, 32))
        self.data_y = raw_y.reshape((raw_y.shape[0]))

        self.transform = transform

    def __getitem__(self, index):
        x = self.data_x[index,:,:,:]
        y = self.data_y[index]
        x = torch.from_numpy(x).float() / 255
        y = int(y)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.data_x.shape[0]


# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        x = self.model(x)
        return x