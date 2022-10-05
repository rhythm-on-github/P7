import torch
import numpy as np
from torch.autograd import Variable


def train_discriminator(opt, Tensor, data, genData, device, discriminator, generator, optim_disc, loss_func, disc_losses):
	optim_disc.zero_grad()

	# Configure input
	real_data = Variable(data.type(Tensor))

	# get a label vector for true label
	real_batch_size = real_data.size(0)
	labels = torch.full((real_batch_size,), 1, dtype=torch.float, device=device)

	# use real data
	real_preds = discriminator(real_data).view(-1)
	real_loss = loss_func(real_preds, labels)

	# use generated data
	z = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], opt.latent_dim))))
	fake_data = generator(z)
	labels2 = torch.full((real_batch_size,), 0, dtype=torch.float, device=device)
	fake_preds = discriminator(fake_data.detach()).view(-1)
	fake_loss = loss_func(fake_preds, labels2)

	# combine
	loss_D = real_loss + fake_loss

	# optimize
	learningChoice = 'n' #do not disable learning
	if(learningChoice != 'y'):
		loss_D.backward()
		optim_disc.step()
		if(opt.clip_value != -1):
			for p in discriminator.parameters():
				p.data.clamp_(-opt.clip_value, opt.clip_value)
					
	#save misc data
	genData.append(fake_data)
	(real_losses, fake_losses, discriminator_losses) = disc_losses
	real_losses.append(real_loss.item())
	fake_losses.append(fake_loss.item())
	discriminator_losses.append(loss_D.item())

	#return things as needed elsewhere
	return real_batch_size




def train_generator(genData, device, discriminator, optim_gen, loss_func, real_batch_size, generator_losses):
	optim_gen.zero_grad()

	# fake labels are real for generator cost
	labels = torch.full((real_batch_size,), 1, dtype=torch.float, device=device)

	# run generator
	fake_data = genData[-1]
	output = discriminator(fake_data).view(-1)
	loss_G = loss_func(output, labels)

	# optimize
	learningChoice = 'n' #do not disalbe learning
	if(learningChoice != 'y'):
		loss_G.backward()
		optim_gen.step()

	generator_losses.append(loss_G.item())