from .Triple import *
import torch
import torch.nn as nn

# takes a triple (h,r,t) and encodes it with one-hot encoding
def encode(triple:Triple, entities, entN, relations, relN):
	hID = entities[triple.h]
	rID = relations[triple.r]
	tID = entities[triple.t]

	hEnc = nn.functional.one_hot(torch.arange(1) + hID, num_classes = entN)
	rEnc = nn.functional.one_hot(torch.arange(1) + rID, num_classes = relN)
	tEnc = nn.functional.one_hot(torch.arange(1) + tID, num_classes = entN)

	enc = torch.cat((hEnc, rEnc, tEnc))
	return enc


# takes a one-hot ecoded triple and converts it to (h,r,t)
def decode(triple:list, entities, entN, relations, relN):
	pass