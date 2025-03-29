import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, grid_size = 64, hidden_layers = 4):
        super(model, self).__init__()
        
        self.grid_size = grid_size
        
        # token-like embedding of the digital input
        # for each input token, there is a (output_size x output_size) matrix
        # that we construct via eigendecomposition
        
        self.eigValX = nn.Parameter(torch.normal(0, 1/np.sqrt(grid_size), size = (grid_size, grid_size)))
        self.eigValY = nn.Parameter(torch.normal(0, 1/np.sqrt(grid_size), size = (grid_size, grid_size)))

        # left and right eigenvectors, randomly initialised
        
        self.LeftVeX = nn.Parameter(F.normalize(torch.normal(0, 1, size = (grid_size, grid_size, grid_size)), p = 2, dim = -2))
        self.LeftVeY = nn.Parameter(F.normalize(torch.normal(0, 1, size = (grid_size, grid_size, grid_size)), p = 2, dim = -2))
        self.RihtVeX = nn.Parameter(F.normalize(torch.normal(0, 1, size = (grid_size, grid_size, grid_size)), p = 2, dim = -1))
        self.RihtVeY = nn.Parameter(F.normalize(torch.normal(0, 1, size = (grid_size, grid_size, grid_size)), p = 2, dim = -1))

        layers = []
        for i in range(hidden_layers):
            layers.append(nn.PReLU())
            layers.append(nn.ConvTranspose2d(2, 2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1))
        self.CNN = nn.Sequential(*layers)

    def get_embeddings(self, x):
        
        # split x and y digital code
        inX, inY = torch.split(x, self.grid_size, dim=1)

        # get respective eigvals and left/right eigvectors
        lamX = torch.matmul(inX, self.eigValX)
        lamY = torch.matmul(inY, self.eigValY)
        lefX = F.normalize(torch.einsum("ni,ijk->njk", inX, self.LeftVeX), p = 2, dim = -2)
        lefY = F.normalize(torch.einsum("ni,ijk->njk", inY, self.LeftVeY), p = 2, dim = -2)
        rigX = F.normalize(torch.einsum("ni,ijk->njk", inX, self.RihtVeX), p = 2, dim = -1)
        rigY = F.normalize(torch.einsum("ni,ijk->njk", inY, self.RihtVeY), p = 2, dim = -1)
        
        # construct embedding matrices
        embX = torch.einsum("nij,nj,njk->nik", lefX, lamX, rigX)
        embY = torch.einsum("nij,nj,njk->nik", lefY, lamY, rigY)

        return embX, embY
    
    def forward(self, x):
        #self.activations = []
        
        embX, embY = self.get_embeddings(x)
        x = torch.cat((embX.unsqueeze(1), embY.unsqueeze(1)), dim = 1)
        x = self.CNN(x)
        
        # self.activations.append(z.clone())
        return x