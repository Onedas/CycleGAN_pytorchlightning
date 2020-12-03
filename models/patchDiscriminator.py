import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
	"""
		PatchGAN Discriminator
	"""

	def __init__(self, in_c, num_filter=64, num_layers=3, norm_type="instance"):
		super(PatchDiscriminator, self).__init__()
		"""
		Params
			in_c : the number of channels in input images
			num_filter : the number of filters in the conv layers
			num_layers : the number of conv layers
			norm_type : type of normalization layer
		"""

		#
		if norm_type == "batch":
			norm_layer = nn.BatchNorm2d
		elif norm_type == "instance":
			norm_layer = nn.InstanceNorm2d
		else:
			raise NotImplementedError("Check the normalization type")

		self.first = nn.Sequential(
			nn.Conv2d(in_c, num_filter, kernel_size = 4, stride = 2, padding=1),
			nn.LeakyReLU(0.2, True),
			)


		middle = [] 

		prev_f = num_filter
		now_f = num_filter
		for i in range(num_layers):
			now_f *= 2
			middle += [ nn.Conv2d(prev_f, now_f, kernel_size=4, stride=2, padding=1),
						norm_layer(now_f),
						nn.LeakyReLU(0.2, True)]

			prev_f = now_f

		self.middle = nn.Sequential(*middle)

		self.last = nn.Conv2d(prev_f, 1, kernel_size = 4, stride = 2, padding=1)

	def forward(self, x):
		out = self.first(x)
		out = self.middle(out)
		out = self.last(out)
		return out

if __name__ == "__main__":
	x = torch.Tensor(3, 3, 256, 256)
	D = PatchDiscriminator(3)
	print(D)

	print(D(x).shape)