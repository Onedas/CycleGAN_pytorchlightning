import torch
import torch.nn as nn


def vanillaGANLoss():
	return nn.BCE()

def LSGANLoss():
	return nn.MSELoss()

def IdentityLoss():
	return nn.L1Loss()

def CycleLoss():
	return nn.L1Loss()

if __name__ == "__main__":

	x = torch.randn(3, 3, 50, 50)
	x_hat = torch.randn(3, 3, 50, 50)

	print(IdentityLoss(x_hat, x))
