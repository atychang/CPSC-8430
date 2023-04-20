import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "WGAN"
batch_size = 64
# number of channels in the training images
nc = 3
# lower and upper clip value for disc. weights
weight_clipping_limit = 0.01
# number of training steps for discriminator per iter
n_critic = 5
num_epochs = 10000
learning_rate = 0.00005


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
    def __init__(self, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0
            ),
        )

    def forward(self, x):
        return self.main(x)

    def feature_extraction(self, x):
        return self.main(x).view(-1, 1024 * 4 * 4)


generator = Generator(nc).to(device)
discriminator = Discriminator(nc).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

generator_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

one = torch.FloatTensor([1])
mone = one * -1
one, mone = one.to(device), mone.to(device)


def get_infinite_batches(dataloader):
    while True:
        for _, (images, _) in enumerate(dataloader):
            yield images.to(device)


data = get_infinite_batches(dataloader)

for epoch in range(num_epochs):
    for p in discriminator.parameters():
        p.requires_grad = True

    for d_iter in range(n_critic):
        discriminator.zero_grad()

        for p in discriminator.parameters():
            p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

        images = data.__next__()
        if images.size()[0] != batch_size:
            continue

        z = torch.rand((batch_size, 100, 1, 1), device=device)

        d_loss_real = discriminator(images)
        d_loss_real = d_loss_real.mean(0).view(1)
        d_loss_real.backward(one)

        z = torch.rand((batch_size, 100, 1, 1), device=device)
        fake_images = generator(z)
        d_loss_fake = discriminator(fake_images)
        d_loss_fake = d_loss_fake.mean(0).view(1)
        d_loss_fake.backward(mone)

        d_loss = d_loss_fake - d_loss_real
        Wasserstein_D = d_loss_real - d_loss_fake
        discriminator_optimizer.step()
        print(
            f"  Discriminator iteration: {d_iter+1}/{n_critic}, \
                loss_fake: {d_loss_fake.item()}, loss_real: {d_loss_real.item()}"
        )

    for p in discriminator.parameters():
        p.requires_grad = False

    generator.zero_grad()

    z = torch.rand((batch_size, 100, 1, 1), device=device)
    fake_images = generator(z)
    g_loss = discriminator(fake_images)
    g_loss = g_loss.mean().mean(0).view(1)
    g_loss.backward(one)
    g_cost = -g_loss
    generator_optimizer.step()
    print(f"Generator iteration: {epoch+1}/{num_epochs}, g_loss: {g_loss.item()}")

    if (epoch + 1) % 200 == 0:
        torch.save(
            generator.state_dict(), f"./models/{model_name}/generator-{epoch+1}.pt"
        )
        torch.save(
            discriminator.state_dict(),
            f"./models/{model_name}/discriminator-{epoch+1}.pt",
        )

        z = torch.rand((800, 100, 1, 1), device=device)
        samples = generator(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = make_grid(samples)
        save_image(grid, f"generated_images/{model_name}/{epoch+1}.png")
