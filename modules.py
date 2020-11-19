import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import csv
#------------------ Latent space visualistion -------------------#


def getImage(path, size):
    image = Image.open(path)
    image = image.resize(size)
    return OffsetImage(image)


def vis_tsne(latent, img_size, fig_size):
    img_paths = [latent[i][0] for i in range(len(latent))]
    latent_rep = [latent[i][1] for i in range(len(latent))]

    tene = TSNE().fit_transform(latent_rep[:200])
    x = tene[:, 0]
    y = tene[:, 1]

    plt.rcParams["figure.figsize"] = (fig_size, fig_size)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for x0, y0, path in zip(x, y, img_paths):
        ab = AnnotationBbox(
            getImage(path, (img_size, img_size)), (x0, y0), frameon=False)
        ax.add_artist(ab)
    fig.show()


def vis_tb(latent, img_size):
    img_paths = [latent[i][0] for i in range(len(latent))]
    latent_rep = [latent[i][1] for i in range(len(latent))]
    # create tsv with feature vec arrays
    with open('vis/feature_vecs.tsv', 'w') as fw:
        csv_writer = csv.writer(fw, delimiter='\t')
        csv_writer.writerow(latent_rep)
    # create sprite image
    images = [Image.open(img_path).resize((img_size, img_size))
              for img_path in img_paths]
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = img_size * one_square_size
    master_height = img_size * one_square_size

    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))  # fully transparent

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = img_size*div
        w_loc = img_size*mod
        spriteimage.paste(image, (w_loc, h_loc))
    spriteimage.convert("RGB").save('vis/sprite.jpg', transparency=0)

    #------------------ Dataset --------------------#


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None):
        super().__init__()
        self.df = pd.read_csv(csv)  # read the data csv
        self.transform = transform  # save the transform variable as part of the class object

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get the image filepath and label from at that index from the csv
        filepath = self.df['file_path'][idx]
        # open with PIL and convert to rgb
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)  # apply transforms
        return img, img

#------------------ model --------------------#


class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.architechture_encode = {
            'channels': (8, 16, 32, 64, 128, 256),
            'kernel_size': (3, 3, 3, 3, 3, 3),
            'stride': (2, 2, 2, 1, 1, 1,),
            'padding': (0, 0, 0, 1, 1, 1)
        }

        self.architechture_decode = {
            'channels': (128, 64, 32, 16, 8, 3),
            'kernel_size': (3, 5, 7, 7, 9, 10),
            'stride': (1, 2, 2, 2, 2, 2),
            'padding': (2, 2, 2, 2, 3, 3)
        }
        self.z = None
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                self.architechture_encode['channels'][0],
                self.architechture_encode['kernel_size'][0],
                self.architechture_encode['stride'][0],
                self.architechture_encode['padding'][0]
            ),
            # nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][0]),
            nn.Conv2d(
                self.architechture_encode['channels'][0],
                self.architechture_encode['channels'][1],
                self.architechture_encode['kernel_size'][1],
                self.architechture_encode['stride'][1],
                self.architechture_encode['padding'][1]
            ),
            # nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][1]),
            nn.Conv2d(
                self.architechture_encode['channels'][1],
                self.architechture_encode['channels'][2],
                self.architechture_encode['kernel_size'][2],
                self.architechture_encode['stride'][2],
                self.architechture_encode['padding'][2]
            ),
            # nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][2]),
            nn.Conv2d(
                self.architechture_encode['channels'][2],
                self.architechture_encode['channels'][3],
                self.architechture_encode['kernel_size'][3],
                self.architechture_encode['stride'][3],
                self.architechture_encode['padding'][3]
            ),
            # nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][3]),
            nn.Conv2d(
                self.architechture_encode['channels'][3],
                self.architechture_encode['channels'][4],
                self.architechture_encode['kernel_size'][4],
                self.architechture_encode['stride'][4],
                self.architechture_encode['padding'][4]
            ),
            # nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][4]),
            nn.Conv2d(
                self.architechture_encode['channels'][4],
                self.architechture_encode['channels'][5],
                self.architechture_encode['kernel_size'][5],
                self.architechture_encode['stride'][5],
                self.architechture_encode['padding'][5]
            ),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_encode['channels'][5])
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                self.architechture_decode['channels'][0],
                self.architechture_decode['kernel_size'][0],
                self.architechture_decode['stride'][0],
                self.architechture_decode['padding'][0]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_decode['channels'][0]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][0],
                self.architechture_decode['channels'][1],
                self.architechture_decode['kernel_size'][1],
                self.architechture_decode['stride'][1],
                self.architechture_decode['padding'][1]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_decode['channels'][1]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][1],
                self.architechture_decode['channels'][2],
                self.architechture_decode['kernel_size'][2],
                self.architechture_decode['stride'][2],
                self.architechture_decode['padding'][2]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_decode['channels'][2]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][2],
                self.architechture_decode['channels'][3],
                self.architechture_decode['kernel_size'][3],
                self.architechture_decode['stride'][3],
                self.architechture_decode['padding'][3]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_decode['channels'][3]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][3],
                self.architechture_decode['channels'][4],
                self.architechture_decode['kernel_size'][4],
                self.architechture_decode['stride'][4],
                self.architechture_decode['padding'][4]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.architechture_decode['channels'][4]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][4],
                self.architechture_decode['channels'][5],
                self.architechture_decode['kernel_size'][5],
                self.architechture_decode['stride'][5],
                self.architechture_decode['padding'][5]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][5]),
            nn.Sigmoid()
        )

    def encode(self, x):
        self.z = self.encoder(x)
        return self.z

    def decode(self):
        return self.decoder(self.z)

    def forward(self, x):
        self.z = self.encoder(x)
        print(self.z.shape)
        return self.decoder(self.z)


#------------------------- ResNet AE ----------------------------#


class ResNet_AE(nn.Module):
    def __init__(self):
        super(ResNet_VAE, self).__init__()

        self.architechture_encode = {
            'channels': (1024, 256, 64, 16),
            'kernel_size': (1, 1, 1, 1),
            'stride': (1, 1, 1, 1),
            'padding': (1, 1, 1, 1)
        }

        self.architechture_decode = {
            'channels': (512, 128, 32, 3),
            'kernel_size': (9, 5, 5, 4),
            'stride': (3, 2, 2, 2),
            'padding': (1, 1, 1, 0)
        }

        resnet = models.resnet152(pretrained=True)
        list_of_conv = list(resnet.children())[:-1]
        self.z = None
        self.encoder_resnet = nn.Sequential(*list_of_conv)
        for param in self.encoder_resnet.parameters():
            param.requires_grad = False
        self.encoder_bn = nn.Sequential(
            nn.Conv2d(
                2048,
                self.architechture_encode['channels'][0],
                self.architechture_encode['kernel_size'][0],
                self.architechture_encode['stride'][0],
                self.architechture_encode['padding'][0]
            ),
            nn.BatchNorm2d(self.architechture_encode['channels'][0]),
            nn.Conv2d(
                self.architechture_encode['channels'][0],
                self.architechture_encode['channels'][1],
                self.architechture_encode['kernel_size'][1],
                self.architechture_encode['stride'][1],
                self.architechture_encode['padding'][1]
            ),
            nn.BatchNorm2d(self.architechture_encode['channels'][1]),
            nn.Conv2d(
                self.architechture_encode['channels'][1],
                self.architechture_encode['channels'][2],
                self.architechture_encode['kernel_size'][2],
                self.architechture_encode['stride'][2],
                self.architechture_encode['padding'][2]
            ),
            nn.BatchNorm2d(self.architechture_encode['channels'][2]),
            nn.Conv2d(
                self.architechture_encode['channels'][2],
                self.architechture_encode['channels'][3],
                self.architechture_encode['kernel_size'][3],
                self.architechture_encode['stride'][3],
                self.architechture_encode['padding'][3]
            ),
            nn.BatchNorm2d(self.architechture_encode['channels'][3]),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16,
                self.architechture_decode['channels'][0],
                self.architechture_decode['kernel_size'][0],
                self.architechture_decode['stride'][0],
                self.architechture_decode['padding'][0]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][0]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][0],
                self.architechture_decode['channels'][1],
                self.architechture_decode['kernel_size'][1],
                self.architechture_decode['stride'][1],
                self.architechture_decode['padding'][1]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][1]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][1],
                self.architechture_decode['channels'][2],
                self.architechture_decode['kernel_size'][2],
                self.architechture_decode['stride'][2],
                self.architechture_decode['padding'][2]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][2]),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][2],
                self.architechture_decode['channels'][3],
                self.architechture_decode['kernel_size'][3],
                self.architechture_decode['stride'][3],
                self.architechture_decode['padding'][3]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][3]),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_resnet(x)
        self.z = self.encoder_bn(x)
        return self.z

    def decode(self):
        return self.decoder(self.z)

    def forward(self, x):
        x = self.encoder_resnet(x)
        print(x.shape)
        self.z = self.encoder_bn(x)
        # print(self.z.shape)
        return self.decoder(self.z)

    def freeze(self):
        for param in self.encoder_resnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder_resnet.parameters():
            param.requires_grad = True

# --------------------------- VAE ------------------------------#
# Variational autoencoder


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.architechture_encode = {
            'channels': (8, 16, 32, 64, 128, 256),
            'kernel_size': (3, 3, 3, 3, 3, 3),
            'stride': (2, 2, 2, 1, 1, 1,),
            'padding': (0, 0, 0, 1, 1, 1)
        }

        self.architechture_decode = {
            'channels': (128, 64, 32, 16, 8, 3),
            'kernel_size': (3, 5, 7, 7, 9, 10),
            'stride': (1, 2, 2, 2, 2, 2),
            'padding': (2, 2, 2, 2, 3, 3)
        }
        self.z = None
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                self.architechture_encode['channels'][0],
                self.architechture_encode['kernel_size'][0],
                self.architechture_encode['stride'][0],
                self.architechture_encode['padding'][0]
            ),
            # nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][0]),
            nn.ReLU(),
            nn.Conv2d(
                self.architechture_encode['channels'][0],
                self.architechture_encode['channels'][1],
                self.architechture_encode['kernel_size'][1],
                self.architechture_encode['stride'][1],
                self.architechture_encode['padding'][1]
            ),
            # nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][1]),
            nn.ReLU(),
            nn.Conv2d(
                self.architechture_encode['channels'][1],
                self.architechture_encode['channels'][2],
                self.architechture_encode['kernel_size'][2],
                self.architechture_encode['stride'][2],
                self.architechture_encode['padding'][2]
            ),
            # nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][2]),
            nn.ReLU(),
            nn.Conv2d(
                self.architechture_encode['channels'][2],
                self.architechture_encode['channels'][3],
                self.architechture_encode['kernel_size'][3],
                self.architechture_encode['stride'][3],
                self.architechture_encode['padding'][3]
            ),
            # nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][3]),
            nn.ReLU(),
            nn.Conv2d(
                self.architechture_encode['channels'][3],
                self.architechture_encode['channels'][4],
                self.architechture_encode['kernel_size'][4],
                self.architechture_encode['stride'][4],
                self.architechture_encode['padding'][4]
            ),
            # nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][4]),
            nn.ReLU(),
            nn.Conv2d(
                self.architechture_encode['channels'][4],
                self.architechture_encode['channels'][5],
                self.architechture_encode['kernel_size'][5],
                self.architechture_encode['stride'][5],
                self.architechture_encode['padding'][5]
            ),
            nn.MaxPool2d(3),
            nn.BatchNorm2d(self.architechture_encode['channels'][5]),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                self.architechture_decode['channels'][0],
                self.architechture_decode['kernel_size'][0],
                self.architechture_decode['stride'][0],
                self.architechture_decode['padding'][0]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][0]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][0],
                self.architechture_decode['channels'][1],
                self.architechture_decode['kernel_size'][1],
                self.architechture_decode['stride'][1],
                self.architechture_decode['padding'][1]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][1]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][1],
                self.architechture_decode['channels'][2],
                self.architechture_decode['kernel_size'][2],
                self.architechture_decode['stride'][2],
                self.architechture_decode['padding'][2]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][2]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][2],
                self.architechture_decode['channels'][3],
                self.architechture_decode['kernel_size'][3],
                self.architechture_decode['stride'][3],
                self.architechture_decode['padding'][3]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][3]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][3],
                self.architechture_decode['channels'][4],
                self.architechture_decode['kernel_size'][4],
                self.architechture_decode['stride'][4],
                self.architechture_decode['padding'][4]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][4]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.architechture_decode['channels'][4],
                self.architechture_decode['channels'][5],
                self.architechture_decode['kernel_size'][5],
                self.architechture_decode['stride'][5],
                self.architechture_decode['padding'][5]
            ),
            nn.BatchNorm2d(self.architechture_decode['channels'][5]),
            nn.Sigmoid()
        )

    def encode(self, x):
        self.z = self.encoder(x)
        return self.z

    def decode(self):
        return self.decoder(self.z)

    def forward(self, x):
        self.z = self.encoder(x)
        print(self.z.shape)
        return self.decoder(self.z)
