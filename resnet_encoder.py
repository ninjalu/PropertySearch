# %%
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from time import time
import numpy as np
from modules import AEDataset, SimpleAE
import pandas as pd
import torchvision
# %%

# %%


class ResnetEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet152(pretrained=True)
        list_of_conv = list(resnet.children())[:-1]
        self.layers = nn.Sequential(*list_of_conv)
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.layers(x)


model = ResnetEncoder()
model = model.eval()
# %%
data_dir = 'propertyimages/exterior/image_list.csv'
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405],
                         std=[0.229, 0.224, 0.225])
])


def get_latent_rep(model, csv, transform):
    df = pd.read_csv(csv)
    latent = []
    for i in range(len(df)):
        file_path = df['file_path'][i]
        img = Image.open(file_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        # print(img.shape)
        features = model(img).squeeze().flatten()
        latent.append((file_path, features.detach().numpy()))
    return latent


latent = get_latent_rep(model, data_dir, data_transform)

# %%
img_paths = [latent[i][0] for i in range(len(latent))]
latent_rep = [latent[i][1] for i in range(len(latent))]
tene = TSNE().fit_transform(latent_rep[:100])
x = tene[:, 0]
y = tene[:, 1]
# %%


def getImage(path, size):
    image = Image.open(path)
    image = image.resize(size)
    return OffsetImage(image)


# %%
plt.rcParams["figure.figsize"] = (30, 30)
fig, ax = plt.subplots()
ax.scatter(x, y)
for x0, y0, path in zip(x, y, img_paths):
    ab = AnnotationBbox(getImage(path, (80, 80)), (x0, y0), frameon=False)
    ax.add_artist(ab)

# %%
