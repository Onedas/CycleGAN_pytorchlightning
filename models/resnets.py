import torch
import torch.nn as nn


class ResNet(nn.Module):
	"""
		ResNet model for CycleGAN Generator
	"""

	def __init__(self, in_c, out_c, num_filter, 
				 norm_type = "batch",
				 n_blocks = 6,
				 use_dropout = False,
				 use_bias = False,
				 padding_type = "zero",
				 ):

		"""
		Parmas
			in_c : the number of input image channels (if rgb = 3)
			out_c : the number of output image channels 
			num_filter : the number of filters at first layers
			norm_type : type of normalization layer 
			n_blocks : the number of residual block
			...

		"""

		super(ResNet, self).__init__()
			
		if norm_type == "batch":
			norm_layer = nn.BatchNorm2d
		elif norm_type == "instance":
			norm_layer = nn.InstanceNorm2d
		# elif norm_type == "none":
		# 	norm_layer = lambda x:x
		else:
			raise NotImplementedError("Check the normalization type")

		###

		# First layer
		self.first = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_c, num_filter, kernel_size=7, padding=0, bias=use_bias),
			norm_layer(num_filter),
			nn.ReLU(inplace =True),
			)


		# Downsample layers
		downsample = []
		n_downsample = 2
		for i in range(n_downsample):
			mult = 2 ** i
			downsample += [nn.Conv2d(num_filter * mult, num_filter * mult * 2, kernel_size =3, stride = 2, padding = 1, bias = use_bias),
						   norm_layer(num_filter * mult * 2),
			      		   nn.ReLU(True),]
		mult = 2 ** n_downsample

		self.downsampling = nn.Sequential(*downsample)
		
		# ResnetBlock
		resnet_layers = []
		for i in range(n_blocks):
			resnet_layers += [ResNetBlock(num_filter * mult, 
											padding_type=padding_type,
											norm_layer = norm_layer,
											use_dropout = use_dropout,
											use_bias = use_bias,)]
		self.resnet_layers = nn.Sequential(*resnet_layers)

		# Upsampling layers
		upsample = []
		for i in range(n_downsample):
			mult = 2 ** (n_downsample - i)
			upsample += [nn.ConvTranspose2d(num_filter*mult, int(num_filter*mult/2),
							kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias=use_bias),
						norm_layer(int(num_filter*mult/2)),
						nn.ReLU(True)]

		self.upsampling = nn.Sequential(*upsample)
		
		# Last layer
		self.last = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(num_filter, out_c, kernel_size=7, padding=0),
			nn.Tanh()
			)

	def forward(self, x):
		x = self.first(x)
		x = self.downsampling(x)
		x = self.resnet_layers(x)
		x = self.upsampling(x)
		x = self.last(x)
		return x


class ResNetBlock(nn.Module):
	"""
		Residual block for ResNet
	"""
	def __init__(self, dim, padding_type, 
		norm_layer = nn.BatchNorm2d,
		use_dropout = False, 
		use_bias = False,
		):
		"""
		Params
			dim : the number of channels in features
			padding_type : type of padding (ex. zero, reflect, replicate)
			norm_layer : normalization layer module.	
		"""

		super(ResNetBlock, self).__init__()

		# padding
		if padding_type == 'reflect':
			self.padding = nn.ReflectionPad2d(1)
		elif padding_type == 'replicate':
			self.padding = nn.Replicationpad2d(1)
		elif padding_type == 'zero':
			self.padding = nn.ZeroPad2d(1)
		else:
			raise NotImplementedError('Check padding type')

		self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
		self.norm = norm_layer(dim)

		#
	def forward(self, x):
		out = self.padding(x)
		out = self.conv(out)
		out = self.norm(out)
		return out + x


if __name__ == "__main__":
	resnet9 = ResNet(3, 3, 50, n_blocks = 9)
	print(resnet9)

	batch_size = 3
	W, H, C = 256, 256, 3
	x = torch.Tensor(batch_size, C, W, H)
	
	with torch.no_grad():
		out = resnet9(x)
		print(out.shape)