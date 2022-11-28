import torch.nn as nn
import torch

class Generator(nn.Module):
    """A GAN generator with convolutions"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            # state size: 1-wide, zSize channels
            nn.ConvTranspose1d(zSize, 20, 512, 1, 0, bias=False),
            # state size: 512-wide, 10ch
            nn.ReLU(),
            nn.ConvTranspose1d(20, 10, 32, 2, 15, bias=False), #padding = (kernel_size/2) - 1
            # state size: 1024-wide, 10ch
            nn.ReLU(),
            nn.Conv1d(10, entitiesN + relationsN + entitiesN, 1024, 1, 0, bias=False),
            # state size: 1-wide, entN+relN+entN channels
            nn.BatchNorm1d(entitiesN + relationsN + entitiesN),
            nn.Tanh()
        )

    def forward(self, z):
        # takes latent vector and makes a one-hot encoded triple
        z = torch.unsqueeze(z, dim=-1)
        tripleEnc = self.model(z)
        tripleEnc = torch.squeeze(tripleEnc, dim=-1)
        return tripleEnc



class Discriminator(nn.Module):
    """A very simple GAN discriminator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(entitiesN + relationsN + entitiesN, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # takes a one-hot encoded triple and gives a binary classifications (real/synthetic)
        prediction = self.model(z)
        return prediction
