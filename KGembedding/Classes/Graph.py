import matplotlib.pyplot as plt

def saveGraph(path, G_losses, D_losses):
	plt.figure(figsize=(10, 5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(path)
	plt.close()