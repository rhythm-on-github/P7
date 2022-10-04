import torch
from torch.autograd import Variable


def train_discriminator(self, opt, Tensor, data, genData, device, discriminator, generator, optim_disc, loss_func):
	optim_disc.zero_grad()

	# Configure input
	real_data = Variable(data.type(self.Tensor))

	# get a label vector for true label
	real_batch_size = real_data.size(0)
	labels = torch.full((real_batch_size,), 1, dtype=torch.float, device=device)

	# use real data
	real_preds = discriminator(real_imgs).view(-1)
	real_loss = loss_func(real_preds, labels)

	# use generated data
	z = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], opt.latent_dim, 1, 1))))
	fake_data = generator(z)
	labels2 = torch.full((real_batch_size,), 0, dtype=torch.float, device=device)
	fake_preds = discriminator(fake_data.detach()).view(-1)
	fake_loss = loss_func(fake_preds, labels2)

	# combine
	loss_D = real_loss + fake_loss

	# optimize
	learningChoice == 'n' #do not disable learning
	if(learningChoice != 'y'):
		loss_D.backward()
		optim_disc.step()
		if(opt.clip_value != -1):
			for p in discriminator.parameters():
				p.data.clamp_(-opt.clip_value, opt.clip_value)
					
	genData.append(fake_data)
	#self.real_losses.append(real_loss.item())
	#self.fake_losses.append(fake_loss.item())
	#self.discriminator_losses.append(loss_D.item())

	#return things as needed
	#maybe real_batch_size?