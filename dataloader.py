import os
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


def load_horse2zebra(opt, transform=None):
	"""
		horse2zebra dataset loader
	"""
	train_dataset = horse2zebra(mode='train',size=opt.size, num=opt.num_traindata, transform=transform)
	test_dataset = horse2zebra(mode='test',num=opt.num_validdata, transform=transform)

	train_loader = data.DataLoader(train_dataset, opt.batch_size, shuffle=True)
	test_loader = data.DataLoader(test_dataset, opt.batch_size, shuffle=False)

	return train_loader, test_loader


class horse2zebra(data.Dataset):
	"""
		horse2zebra pytorch Dataset class for CycleGAN 
	"""
	def __init__(self, 
		root = 'data/horse2zebra',
		transform = None,
		mode='train',
		num=99999,
		size=256):
		super(horse2zebra, self).__init__()
		"""	
		Params:
			root : root of dataset
			transform : pytorch transform
			mode(str) : [train|test]
		"""

		self.root = root
		self.transform = transform
		self.mode = mode

		# file list
		self.Afiles = list(os.listdir(root+'/{}A'.format(self.mode)))
		self.Bfiles = list(os.listdir(root+'/{}B'.format(self.mode)))
		self.num = min(len(self.Afiles), len(self.Bfiles),num)
		
		self.default_transform = transforms.Compose([
			transforms.Resize(int(size*1.2), Image.BICUBIC), 
            transforms.RandomCrop(size), 
            transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
	        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	        ])

	def __len__(self):
		return self.num

	def __getitem__(self, idx):
		A = Image.open(self.root + '/{}A/{}'.format(self.mode, self.Afiles[idx])).convert('RGB')
		B = Image.open(self.root + '/{}B/{}'.format(self.mode, self.Bfiles[idx])).convert('RGB')

		A = self.default_transform(A)
		B = self.default_transform(B)

		if self.transform:
			A = self.transform(A)
			B = self.transform(B)

		return A, B


if __name__ == "__main__":

	from config import get_arguments
	import matplotlib.pyplot as plt
	parser = get_arguments()
	opt = parser.parse_args()

	train_dataset, valid_dataset = load_horse2zebra(opt)
	print(len(train_dataset), len(valid_dataset))

	## class test
	train_dataset= horse2zebra(mode='train')
	test_dataset = horse2zebra(mode='test')
	num = 100
	A,B= train_dataset[num]
	a,b = test_dataset[num]
	print(A.shape, B.shape)

	plt.figure(figsize=(10,5))

	plt.subplot(2,2,1)
	plt.imshow(A.numpy().transpose((1,2,0)))

	plt.subplot(2,2,2)
	plt.imshow(B.numpy().transpose((1,2,0)))

	plt.subplot(2,2,3)
	plt.imshow(a.numpy().transpose((1,2,0)))

	plt.subplot(2,2,4)
	plt.imshow(b.numpy().transpose((1,2,0)))

	plt.show()