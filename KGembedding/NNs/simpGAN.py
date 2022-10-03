import torch.nn as nn

class Generator(nn.Module):
    """A very simple GAN generator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(zSize, 512),
            nn.ReLU(),
            nn.Linear(512, entitiesN + relationsN + entitiesN)
        )

    def forward(self, z):
        # takes latent vector and makes a one-hot encoded triple
        tripleEnc = self.model(z)
        return tripleEnc



class Discriminator(nn.Module):
    """A very simple GAN discriminator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(entitiesN + relationsN + entitiesN, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, z):
        # takes a one-hot encoded triple and gives a binary classifications (real/synthetic)
        tripleEnc = self.model(z)
        return tripleEnc