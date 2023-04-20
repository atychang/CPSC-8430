import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "DCGAN"
batch_size = 128
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
num_epochs = 30
learning_rate = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


def create_dir(path, remove=True):
    if os.path.exists(path) and remove:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


create_dir("./models", remove=False)
create_dir(f"./models/{model_name}")
create_dir("./generated_images", remove=False)
create_dir(f"./generated_images/{model_name}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset = datasets.CIFAR10(
    root="./data",
    download=True,
    transform=transform,
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()

generator_optimizer = optim.Adam(
    generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
)
discriminator_optimizer = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
)

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    real_label = 0.9
    fake_label = 0.1
    for batch_idx, data in enumerate(dataloader):
        discriminator.zero_grad()
        images = data[0].to(device)
        b_size = images.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = discriminator(images).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_images.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        discriminator_optimizer.step()

        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        generator_optimizer.step()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )
    for i, image in enumerate(fake_images):
        save_image(
            image,
            f"./generated_images/{model_name}/{epoch+1}-{i}.png",
        )
    save_image(
        fake_images,
        f"./generated_images/{model_name}/{epoch+1}.png",
        nrows=8,
    )

    if (epoch + 1) % 10 == 0:
        torch.save(
            generator.state_dict(), f"./models/{model_name}/generator-{epoch+1}.pt"
        )
