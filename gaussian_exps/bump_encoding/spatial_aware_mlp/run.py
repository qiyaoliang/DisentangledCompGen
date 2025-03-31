import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import time as time
import pandas as pd
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import BumpTransformer
from dataset import GaussianBumpDataset

fig_dir = "results"
os.makedirs(fig_dir, exist_ok=True)

sigma = 1.0 # spread of the spot
grid_size = 28 # size of the image grid
depth = 2
save_name = f'depth_{depth}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# visualise different mask regions
rect_thre = 14
masks = [lambda x, y: True,
         lambda x, y: (x < rect_thre) | (y < rect_thre),
         lambda x, y: not ((x < rect_thre) | (y < rect_thre)),
        ]

# Dataset and DataLoader
set_FU = GaussianBumpDataset(mask = masks[0], sigma = sigma, grid_size = grid_size)
set_ID = GaussianBumpDataset(mask = masks[1], sigma = sigma, grid_size = grid_size)
set_OD = GaussianBumpDataset(mask = masks[2], sigma = sigma, grid_size = grid_size)

inputs_FU, targets_FU = set_FU.get_all_data()
inputs_FU, targets_FU = inputs_FU.to(device), targets_FU.to(device)

dataloader_FU = DataLoader(set_FU, batch_size=32, shuffle=True)
dataloader_ID = DataLoader(set_ID, batch_size = 32, shuffle=True)
dataloader_OD = DataLoader(set_OD, batch_size = 32, shuffle=True)

num_epochs = 2000

torch.manual_seed(42)

criterion = nn.MSELoss()

model = BumpTransformer(input_dim = 56, 
                        embed_dim = 256, 
                        num_heads = 2,
                        num_layers = depth, 
                        grid_size = 28)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
total_params = sum(p.numel() for p in model.parameters())
print('total model parameters is: {0}'.format(total_params))

# training the modelAnalytical network
LOSS = []

for epoch in range(num_epochs):
    model.train()
    running_loss_ID = 0.0
    for batch_idx, (inputs_ID, targets_ID) in enumerate(dataloader_ID):
        inputs_ID, targets_ID = inputs_ID.to(device), targets_ID.to(device)
        optimizer.zero_grad()
        outputs_ID = model(inputs_ID)
        loss = criterion(outputs_ID, targets_ID)
        running_loss_ID += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_loss_ID = running_loss_ID / len(dataloader_ID)
    
    model.eval()  # Set the model to evaluation mode
    running_loss_OD = 0.0
    with torch.no_grad():  # No gradients needed during validation
        for batch_idx, (inputs_OD, targets_OD) in enumerate(dataloader_OD):
            inputs_OD, targets_OD = inputs_OD.to(device), targets_OD.to(device)
            outputs_OD = model(inputs_OD)  # Forward pass
            loss = criterion(outputs_OD, targets_OD)  # Compute the validation loss
            running_loss_OD += loss.item()
    avg_loss_OD = running_loss_OD / len(dataloader_OD)
    
    LOSS.append([avg_loss_ID, avg_loss_OD])
    print(f"Epoch [{epoch+1}/{num_epochs}], ID Loss: {avg_loss_ID:.6f}, OOD Loss : {avg_loss_OD:.6f}")
    
LOSS = np.array(LOSS)

torch.save(model.state_dict(), "results/model_{0}.pth".format(save_name))

plt.figure(figsize = (5, 3))
plt.plot(LOSS[:, 0], label = 'in-dist')
plt.plot(LOSS[:, 1], label = 'out-of-dist')
plt.yscale('log')
plt.ylim(1e-6, 1e0)
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.tight_layout()
plt.savefig('results/loss_{0}.png'.format(save_name))
plt.close()

inputs_ID, targets_ID = set_ID.get_all_data()
inputs_OD, targets_OD = set_OD.get_all_data()

inputs_ID, targets_ID = inputs_ID.to(device), targets_ID.to(device)
inputs_OD, targets_OD = inputs_OD.to(device), targets_OD.to(device)

sample_idx_ID = np.arange(len(inputs_ID))
np.random.shuffle(sample_idx_ID)
sample_idx_OD = np.arange(len(inputs_OD))
np.random.shuffle(sample_idx_OD)

fig, axes = plt.subplots(8, 2, figsize = (6, 24))
for i in range(len(axes)):
    res = model(inputs_ID[sample_idx_ID[i], :].unsqueeze(0)).cpu()
    res = res.detach().numpy()
    tar = targets_ID[sample_idx_ID[i], :].cpu()
    tar = tar.numpy()
    axes[i, 0].imshow(np.reshape(res, (grid_size, grid_size)), origin = 'lower', vmin = 0, vmax = 1)
    axes[i, 1].imshow(np.reshape(tar, (grid_size, grid_size)), origin = 'lower', vmin = 0, vmax = 1)
plt.tight_layout()
plt.savefig('results/examples_ID_{0}.png'.format(save_name))
plt.close()

fig, axes = plt.subplots(8, 2, figsize = (6, 24))
for i in range(len(axes)):
    res = model(inputs_OD[sample_idx_OD[i], :].unsqueeze(0)).cpu()
    res = res.detach().numpy()
    tar = targets_OD[sample_idx_OD[i], :].cpu()
    tar = tar.numpy()
    axes[i, 0].imshow(np.reshape(res, (grid_size, grid_size)), origin = 'lower', vmin = 0, vmax = 1)
    axes[i, 1].imshow(np.reshape(tar, (grid_size, grid_size)), origin = 'lower', vmin = 0, vmax = 1)
plt.tight_layout()
plt.savefig('results/examples_OD_{0}.png'.format(save_name))
plt.close()

#with open('results/activations_{0}.npy'.format(save_name), 'wb') as f:
#    for ll in range(len(model.activations)):
#        np.save(f, model.activations[ll].cpu().detach().numpy())