import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader

neg_prob = np.load("neg_prob.npy")
dataloader = DataLoader(neg_prob, batch_size=64, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(5533, 1024)
        self.fc21 = nn.Linear(1024, 32)
        self.fc22 = nn.Linear(1024, 32)
        self.fc3 = nn.Linear(32, 1024)
        self.fc4 = nn.Linear(1024, 5533)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = Variable(data.float())
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
    print('Train Epoch {}; Loss {:.6f}'.format(epoch, train_loss / len(dataloader.dataset)))

model.eval()
a, b = model.encode(torch.from_numpy(neg_prob).float().cuda())
encoded_neg_prob = np.hstack((a.cpu().detach().numpy(), b.cpu().detach().numpy()))
print(encoded_neg_prob.shape)
np.save('encoded_neg_prob.npy', encoded_neg_prob)
