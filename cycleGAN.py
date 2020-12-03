import torch
import pytorch_lightning as pl
import itertools
import wandb
from model_loader import getG, getD
from losses import vanillaGANLoss, LSGANLoss, IdentityLoss, CycleLoss
from data_buffer import ReplayBuffer


class CycleGANModel(pl.LightningModule):

	def __init__(self, opt):
		super(CycleGANModel, self).__init__()
		self.opt = opt

		# G and D
		self.G_ab = getG(opt).to(self.device)
		self.G_ba = getG(opt).to(self.device)
		self.D_a = getD(opt).to(self.device)
		self.D_b = getD(opt).to(self.device)

		# data replay buffer
		self.buffer_A = ReplayBuffer() # buffer size 50
		self.buffer_B = ReplayBuffer()

		# loss functions
		if opt.G_mode == "vanilla":
			self.ganloss = vanillaGANLoss()
		elif opt.G_mode == "lsgan":
			self.ganloss = LSGANLoss()
		else:
			raise NotImplementedError('check G loss type')

		self.identityloss = IdentityLoss()
		self.cycleloss = CycleLoss()

		if self.opt.use_wandb:
			self.save_hyperparameters()
			# self.logger.log_hyperparams(self.opt)
			# self.logger.watch(self.G_ab)
			# self.logger.watch(self.G_ba)
			# self.logger.watch(self.D_a)
			# self.logger.watch(self.D_b)


	def configure_optimizers(self):
		opt_g = torch.optim.Adam(itertools.chain(self.G_ab.parameters(), self.G_ba.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		opt_d_a = torch.optim.Adam(self.D_a.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		opt_d_b = torch.optim.Adam(self.D_b.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		return [opt_g, opt_d_a, opt_d_b], []

	def forward(self, A,B):
		return self.G_ba(A), self.G_ba(B)	

	def training_step(self, batch, batch_idx, optimizer_idx):
		
		A, B = batch
		A = A.to(self.device)
		B = B.to(self.device)

		# train G
		if optimizer_idx == 0:

			# identity loss
			id_A = self.G_ba(A)
			id_B = self.G_ab(B)

			## Identity loss
			loss_id_A = self.identityloss(id_A, A)
			loss_id_B = self.identityloss(id_B, B)

			loss_identity = (loss_id_A + loss_id_B)*0.5*self.opt.lambda_id

			## GAN loss #detach()?
			fake_B = self.G_ab(A)
			fake_A = self.G_ba(B)
			
			D_fake_B = self.D_b(fake_B)
			D_fake_A = self.D_a(fake_A)
			
			ones = torch.ones_like(D_fake_B).to(self.device)
			zeros = torch.zeros_like(D_fake_B).to(self.device)

			loss_gan_ab = self.ganloss(D_fake_B, ones)
			loss_gan_ba = self.ganloss(D_fake_A, ones)

			loss_gan = (loss_gan_ab + loss_gan_ba)*0.5

			## Cycle loss
			recon_A = self.G_ba(fake_B)
			recon_B = self.G_ab(fake_A)

			loss_cycle_A = self.cycleloss(recon_A, A)
			loss_cycle_B = self.cycleloss(recon_B, B)

			loss_cycle = (loss_cycle_A + loss_cycle_B)*0.5*self.opt.lambda_cycle

			## total G loss
			loss_G = loss_gan + loss_identity + loss_cycle
			
			#log
			logs = {"loss_G":loss_G,
				"loss_gan":loss_gan,
				"loss_id":loss_identity,
				"loss_cycle":loss_cycle,
				}
			if self.logger:
				self.logger.experiment.log(logs)

			return loss_G

		# D_a train
		if optimizer_idx == 1:

			fake_A = self.G_ba(B)
			fake_A = self.buffer_A.push_and_pop(fake_A).to(self.device)

			D_A = self.D_a(A)
			D_fake_A = self.D_a(fake_A)

			ones = torch.ones_like(D_A).to(self.device)
			zeros = torch.zeros_like(D_A).to(self.device)

			loss_D_a_real = self.ganloss(D_A, ones)
			loss_D_a_fake = self.ganloss(D_fake_A, zeros)

			loss_Da = (loss_D_a_real + loss_D_a_fake) * 0.5

			# log
			logs = {"loss_Da" : loss_Da}
			if self.logger:
				self.logger.experiment.log(logs)

			return loss_Da

		# D_b 
		if optimizer_idx == 2:

			fake_B = self.G_ab(A)
			fake_B = self.buffer_B.push_and_pop(fake_B).to(self.device)

			D_B = self.D_b(B)
			D_fake_B = self.D_b(fake_B)

			ones = torch.ones_like(D_B).to(self.device)
			zeros = torch.zeros_like(D_B).to(self.device)

			loss_D_b_real = self.ganloss(D_B, ones)
			loss_D_b_fake = self.ganloss(D_fake_B, zeros)

			loss_Db = (loss_D_b_real + loss_D_b_fake) * 0.5

			# log
			logs ={"loss_Db" : loss_Db}
			if self.logger:
				self.logger.experiment.log(logs)
			
			return loss_Db


	def validation_step(self, batch, batch_idx):
		A,B = batch

		# G
		# id loss
		id_B, id_A = self.forward(B,A)
		loss_id_A = self.identityloss(id_A, A)
		loss_id_B = self.identityloss(id_B, B)
		loss_identity = (loss_id_A + loss_id_B )*0.5 * self.opt.lambda_id

		# gan loss
		fake_B, fake_A = self.forward(A,B)
		D_fake_B, D_fake_A = self.D_b(fake_B), self.D_a(fake_A)

		ones = torch.ones_like(D_fake_B).to(self.device)
		zeros = torch.zeros_like(D_fake_A).to(self.device)

		loss_gan_ab = self.ganloss(D_fake_B, ones)
		loss_gan_ba = self.ganloss(D_fake_A, ones)
		loss_gan = (loss_gan_ab + loss_gan_ba) * 0.5

		# cycle loss
		recon_B, recon_A = self.forward(fake_A, fake_B)
		loss_cycle_B = self.cycleloss(recon_B, B)
		loss_cycle_A = self.cycleloss(recon_A, A)
		loss_cycle = (loss_cycle_A + loss_cycle_B) * 0.5 * self.opt.lambda_cycle

		# total loss
		loss_G = loss_gan + loss_identity + loss_cycle

		#Da
		loss_D_a_real = self.ganloss(self.D_a(A), ones)
		loss_D_a_fake = self.ganloss(self.D_b(fake_A), zeros)
		loss_Da = (loss_D_a_real + loss_D_a_fake) * 0.5

		#Db
		loss_D_b_real = self.ganloss(self.D_b(B), ones)
		loss_D_b_fake = self.ganloss(self.D_b(fake_B), ones)
		loss_Db = (loss_D_b_real + loss_D_b_fake) * 0.5


		logs = {"valid_loss_G":loss_G,
				"valid_loss_gan":loss_gan,
				"valid_loss_id":loss_identity,
				"valid_loss_cycle":loss_cycle,
				"valid_loss_Da":loss_Da,
				"valid_loss_Db":loss_Db
				}

		if self.logger:
			self.logger.experiment.log(logs)


if __name__ == "__main__":
	from config import get_arguments

	# import matplotlib.pyplot as plt
	parser = get_arguments()
	opt = parser.parse_args()
	cyclegan = CycleGANModel(opt)
	A = torch.randn(3, 3, 52, 52)
	B = torch.randn(3, 3, 52, 52)
	fake_A, fake_B = cyclegan.forward(A,B)
	print(fake_A.shape)
	print(fake_B.shape)