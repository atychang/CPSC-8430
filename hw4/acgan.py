import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ACGAN"
batch_size = 100
image_size = 64
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
        transforms.Resize(image_size),
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
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.embedding = nn.Embedding(10, 100)

    def forward(self, noise, label):
        label_embedding = self.embedding(label)
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.validity_layer = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

        self.label_layer = nn.Sequential(
            nn.Conv2d(512, 11, 4, 1, 0, bias=False), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, 11)
        return validity, plabel


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

real_labels = 0.7 + 0.5 * torch.rand(10, device=device)
fake_labels = 0.3 * torch.rand(10, device=device)

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.shape[0]

        real_label = real_labels[batch_idx % 10]
        fake_label = fake_labels[batch_idx % 10]

        fake_class_labels = 10 * torch.ones(
            (batch_size,), dtype=torch.long, device=device
        )

        if batch_idx % 25 == 0:
            real_label, fake_label = fake_label, real_label

        discriminator_optimizer.zero_grad()

        # real
        validity_label = torch.full((batch_size,), real_label, device=device)

        pvalidity, plabels = discriminator(images)

        errD_real_val = criterion(pvalidity, validity_label)
        errD_real_label = F.nll_loss(plabels, labels)

        errD_real = errD_real_val + errD_real_label
        errD_real.backward()

        D_x = pvalidity.mean().item()

        # fake
        noise = torch.randn(batch_size, 100, device=device)
        sample_labels = torch.randint(
            0, 10, (batch_size,), device=device, dtype=torch.long
        )

        fakes = generator(noise, sample_labels)

        validity_label.fill_(fake_label)

        pvalidity, plabels = discriminator(fakes.detach())

        errD_fake_val = criterion(pvalidity, validity_label)
        errD_fake_label = F.nll_loss(plabels, fake_class_labels)

        errD_fake = errD_fake_val + errD_fake_label
        errD_fake.backward()

        D_G_z1 = pvalidity.mean().item()

        # finally update the params!
        errD = errD_real + errD_fake

        discriminator_optimizer.step()

        generator_optimizer.zero_grad()

        noise = torch.randn(batch_size, 100, device=device)
        sample_labels = torch.randint(
            0, 10, (batch_size,), device=device, dtype=torch.long
        )

        validity_label.fill_(1)

        fakes = generator(noise, sample_labels)
        pvalidity, plabels = discriminator(fakes)

        errG_val = criterion(pvalidity, validity_label)
        errG_label = F.nll_loss(plabels, sample_labels)

        errG = errG_val + errG_label
        errG.backward()

        D_G_z2 = pvalidity.mean().item()

        generator_optimizer.step()

        print(
            f"[{epoch+1}/{num_epochs}] [{batch_idx+1}/{len(dataloader)}] \
              D_x: [{D_x:.4f}] D_G: [{D_G_z1:.4f}/{D_G_z2:.4f}] G_loss: [{errG:.4f}] \
              D_loss: [{errD:.4f}] D_label: [{errD_real_label + errD_fake_label + errG_label:.4f}]"
        )

        if batch_idx % 100 == 0:
            noise = torch.randn(10, 100, device=device)
            labels = torch.arange(0, 10, dtype=torch.long, device=device)

            gen_images = generator(noise, labels).detach()
            save_image(
                gen_images, f"generated_images/{model_name}/{epoch+1}_{batch_idx+1}.png"
            )
