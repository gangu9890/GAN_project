"""
model_definitions.py
Architectures must EXACTLY match what was trained in Colab.
"""
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────
LATENT_SIZE  = 128
IMAGE_SIZE   = 64
NUM_CHANNELS = 3
IMG_DIM      = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS   # for Vanilla


# ─────────────────────────────────────────────────────────────
# DCGAN
# Saved in Colab as bare nn.Sequential (no self.model wrapper)
# → state_dict keys:  "0.weight", "1.weight", ...
# ─────────────────────────────────────────────────────────────
class DCGAN_Discriminator(nn.Sequential):
    """Matches: discriminator = nn.Sequential(...) in GAN_ASS1.ipynb"""
    def __init__(self):
        super().__init__(
            # 3 x 64 x 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),           # 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),           # 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),           # 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),           # 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()                                # 1 x 1 x 1
        )


class DCGAN_Generator(nn.Sequential):
    """Matches: generator = nn.Sequential(...) in GAN_ASS1.ipynb"""
    def __init__(self):
        super().__init__(
            # latent_size x 1 x 1
            nn.ConvTranspose2d(LATENT_SIZE, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),                             # 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),                             # 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),                             # 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),                             # 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()                                  # 3 x 64 x 64
        )


# ─────────────────────────────────────────────────────────────
# Vanilla GAN
# Saved with self.model wrapper → keys: "model.0.weight", ...
# ─────────────────────────────────────────────────────────────
class VanillaGAN_Discriminator(nn.Module):
    """Matches: Discriminator_vanilla in GAN_ASS1.ipynb"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_DIM, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class VanillaGAN_Generator(nn.Module):
    """Matches: Generator_vanilla in GAN_ASS1.ipynb"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_SIZE, 256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(1024, IMG_DIM),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        img = self.model(z)
        img = img.view(z.size(0), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        return img


# ─────────────────────────────────────────────────────────────
# CGAN  (GAN_ASS2.ipynb)
# Generator(latent_size, num_classes) / Discriminator(num_classes)
# Both use self.model + self.label_emb
# ─────────────────────────────────────────────────────────────
class CGAN_Discriminator(nn.Module):
    """Matches: Discriminator(num_classes) in GAN_ASS2.ipynb"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 64 * 64)
        self.model = nn.Sequential(
            # 4 channels: RGB(3) + label map(1)
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(labels.size(0), 1, 64, 64)
        x = torch.cat([img, label_map], dim=1)
        return self.model(x)


class CGAN_Generator(nn.Module):
    """Matches: Generator(latent_size, num_classes) in GAN_ASS2.ipynb"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
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
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, label_vec], dim=1)
        return self.model(x)