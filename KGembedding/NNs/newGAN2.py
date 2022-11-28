import torch.nn as nn

class Generator(nn.Module):
    """A slightly less simple GAN generator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(zSize, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, entitiesN + relationsN + entitiesN),
            nn.BatchNorm1d(entitiesN + relationsN + entitiesN),
            nn.Tanh()
        )

    def forward(self, z):
        # takes latent vector and makes a one-hot encoded triple
        tripleEnc = self.model(z)
        return tripleEnc



class Discriminator(nn.Module):
    """A slightly less simple GAN discriminator"""
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