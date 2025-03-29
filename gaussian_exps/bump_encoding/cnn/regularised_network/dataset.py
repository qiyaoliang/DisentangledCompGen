import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset Definition
class GaussianBumpDataset(Dataset):
    def __init__(self,
                 grid_size=32,
                 sigma = 1.0, mask = lambda x, y: (x > -1)):

        self.grid_size = grid_size
        self.sigma = sigma
        self.data = []
        self.targets = []

        # x, y coordinate of each grid patch
        xv, yv = np.meshgrid(np.linspace(0.5, grid_size - 0.5, grid_size), np.linspace(0.5, grid_size - 0.5, grid_size))

        # prepare data array
        xd, yd = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        xd, yd = xd.reshape(-1), yd.reshape(-1)
        for i in range(len(xd)):
            x = xd[i] + 0.5 # convert to coordinate number
            y = yd[i] + 0.5 # convert to coordinate number
            if mask(x, y):
                prep_data = np.zeros(grid_size * 2)
                prep_data[xd[i]] = 1
                prep_data[grid_size + yd[i]] = 1
                gaussian_image = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
                self.data.append(prep_data)  # Store x and y as a 2D vector
                self.targets.append(gaussian_image)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.num_samples = len(self.data)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)

    def get_all_data(self):
        # Return all data and targets as tensors
        inputs = torch.tensor(self.data, dtype=torch.float32)
        outputs = torch.tensor(self.targets, dtype=torch.float32)
        return inputs, outputs
