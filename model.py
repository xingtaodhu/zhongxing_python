
import torch.nn as nn

class latent_interval_cross(nn.Module):
	def __init__(self,num_latent=3,hidden_dim=16,num_labels=4):
		super(latent_interval_cross,self).__init__()
		self.num_latent = num_latent
		self.hidden_dim = hidden_dim
		self.previous_embedding = nn.Embedding(num_labels,1)
		self.latent = nn.Embedding(1,1)
		self.emb = nn.Embedding(num_labels+1,hidden_dim)
		self.rnn = nn.GRUCell(hidden_dim,hidden_dim)
		self.pred = nn.Linear(hidden_dim,num_labels+1)

	def forward(self,labels,hidden,dts=None):
		out = self.emb(labels)
		hidden = self.rnn(out,hidden)
		out = self.pred(hidden)
		return out, hidden

class ano(nn.Module):
	def __init__(self):
		super().__init__()
		self.pred = nn.Linear(1,1)

	def forward(self,dts):
		out = self.pred(dts)
		return out	
