import torch.nn as nn
import torch

class Generator(nn.Module):
    """A very simple GAN generator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            # effectively a dense layer, while converting it from 1-wide with zSize channels to 512-wide 1ch
            nn.ConvTranspose1d(zSize, 1, 512, 1, 0, bias=False),
            nn.ReLU(),
            # "normal" transpose convolution
            nn.ConvTranspose1d(1, 1, 16, 2, 7, bias=False), #padding = (kernel_size/2) - 1
            nn.ReLU(),
            # another dense layer, this time from 1024-wide 1ch to 1-wide n-channel
            nn.Conv1d(1, entitiesN + relationsN + entitiesN, 1024, 1, 0, bias=False),
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
            # nn.Conv1d(entitiesN + relationsN + entitiesN, 256, 4, 1, 0, bias=False),
            nn.Linear(entitiesN + relationsN + entitiesN, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # takes a one-hot encoded triple and gives a binary classifications (real/synthetic)
        tripleEnc = self.model(z)
        return tripleEnc
