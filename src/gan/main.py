import argparse
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.5)
# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
# parser.add_argument('--channels', type=int, default=1, help='number of image channels')
# parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)
print(dataloader)

# class Generator(nn.Module):

# class Discriminator(nn.Module):

generator = Generator().cuda()
discriminator = Discriminator().cuda()
loss = nn.CrossEntropyLoss().cuda() # (?) BCELoss()
generator_Optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr)
discriminator_Optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
