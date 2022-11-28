import torch
import torch.nn as nn
from random import randrange

class Generator(nn.Module):
    """A random GAN generator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(zSize, entitiesN + relationsN + entitiesN),
            nn.Tanh()
        )
        self.entitiesN = entitiesN
        self.relationsN = relationsN

    def forward(self, z):
        tripleEncList = []
        for i in range(z.shape[0]):
            # takes latent vector and makes a one-hot encoded triple
            hID = randrange(0, self.entitiesN)
            rID = randrange(0, self.relationsN)
            tID = randrange(0, self.entitiesN)

            hEnc = nn.functional.one_hot(torch.arange(1) + hID, num_classes = self.entitiesN)
            rEnc = nn.functional.one_hot(torch.arange(1) + rID, num_classes = self.relationsN)
            tEnc = nn.functional.one_hot(torch.arange(1) + tID, num_classes = self.entitiesN)
        
            enc = torch.cat((hEnc, rEnc, tEnc), 1)
            enc = torch.squeeze(enc, dim=0)
            enc = enc.type(torch.cuda.FloatTensor)
            tripleEncList.append(enc)

        tripleEnc = torch.stack(tripleEncList)
        tripleEnc.clone().detach().requires_grad_(True)

        device = "cpu"
        if next(self.model.parameters()).is_cuda:
            device = "cuda:0"
        tripleEnc = tripleEnc.to(device)
        return tripleEnc



class Discriminator(nn.Module):
    """A random GAN discriminator"""
    def __init__(self, zSize:int, entitiesN:int, relationsN:int):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(entitiesN + relationsN + entitiesN, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # takes a one-hot encoded triple and gives a binary classifications (real/synthetic)
        prediction = []
        for i in range(z.shape[0]):
            prediction.append(float(randrange(0, 1+1)))

        prediction = torch.tensor(prediction, requires_grad=True)
        prediction = torch.unsqueeze(prediction, dim=-1)
        prediction = torch.unsqueeze(prediction, dim=-1)
        device = "cpu"
        if next(self.model.parameters()).is_cuda:
            device = "cuda:0"
        prediction = prediction.to(device)
        return prediction